import copy
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch


def epoch_learning_rate(epoch, learning_rate, epochs, warmup_epochs=1, warmup_lr=1e-5):
    if (
        not isinstance(epoch, int)
        or not isinstance(epochs, int)
        or not isinstance(warmup_epochs, int)
    ):
        raise TypeError("Epoch and warmup counts must be integers")
    if epochs < 1 or epoch < 0 or warmup_epochs < 0 or warmup_epochs > epochs:
        raise ValueError("Require positive epochs and 0 <= warmup_epochs <= epochs")
    if (
        not math.isfinite(learning_rate)
        or learning_rate <= 0
        or not math.isfinite(warmup_lr)
        or warmup_lr < 0
    ):
        raise ValueError("Learning rates must be finite with a positive base rate")
    if epoch < warmup_epochs:
        return warmup_lr
    decay_steps = max(epochs - warmup_epochs - 1, 0)
    progress = min(epoch - warmup_epochs, decay_steps) / max(decay_steps, 1)
    return 0.5 * learning_rate * (1 + math.cos(math.pi * progress))


def build_optimizer(model, config):
    settings = config.get("training", {})
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("No trainable prompt or projector parameters")
    learning_rate = float(settings.get("learning_rate", 0.002))
    momentum = float(settings.get("momentum", 0.9))
    weight_decay = float(settings.get("weight_decay", 0.0005))
    if (
        not math.isfinite(learning_rate)
        or learning_rate <= 0
        or not 0 <= momentum < 1
        or not math.isfinite(weight_decay)
        or weight_decay < 0
    ):
        raise ValueError("Invalid SGD hyperparameters")
    return torch.optim.SGD(
        parameters,
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=bool(settings.get("nesterov", False)),
    )


def build_scheduler(optimizer, config):
    settings = config.get("training", {})
    epochs = int(settings.get("epochs", 200))
    warmup = int(settings.get("warmup_epochs", 1))
    learning_rate = float(settings.get("learning_rate", 0.002))
    warmup_lr = float(settings.get("warmup_lr", 1e-5))
    epoch_learning_rate(0, learning_rate, epochs, warmup, warmup_lr)
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: (
            epoch_learning_rate(epoch, learning_rate, epochs, warmup, warmup_lr)
            / learning_rate
        ),
    )


def _rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        if (
            not torch.cuda.is_available()
            or len(state["cuda"]) != torch.cuda.device_count()
        ):
            raise ValueError(
                "Checkpoint CUDA RNG topology differs from the current environment"
            )
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state:
        if not torch.backends.mps.is_available():
            raise ValueError(
                "Checkpoint MPS RNG state requires an available MPS backend"
            )
        torch.mps.set_rng_state(state["mps"])


def seed_loader_epoch(loader, seed, epoch):
    if (
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or not isinstance(epoch, int)
        or isinstance(epoch, bool)
        or epoch < 0
    ):
        raise ValueError("Loader seed and nonnegative epoch must be integers")
    if getattr(loader, "persistent_workers", False):
        raise ValueError(
            "Epoch-reproducible training requires persistent_workers=False"
        )
    generator = getattr(loader, "generator", None)
    if generator is not None:
        generator.manual_seed(seed + epoch)
    sampler = getattr(loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
    return seed + epoch


def synchronize_device(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class EpochTimer:
    def __init__(self, device, enabled=False):
        self.device = torch.device(device)
        self.enabled = bool(enabled)
        self.started = None

    def start(self):
        if self.enabled:
            if self.started is not None:
                raise RuntimeError("Epoch timer is already running")
            synchronize_device(self.device)
            self.started = time.perf_counter()
        return self

    def stop(self):
        if not self.enabled:
            return None
        if self.started is None:
            raise RuntimeError("Epoch timer has not been started")
        synchronize_device(self.device)
        seconds = time.perf_counter() - self.started
        self.started = None
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid monotonic epoch timer result")
        return {
            "train_seconds_per_epoch": seconds,
            "scope": "epoch_train_batches",
            "device": str(self.device),
            "synchronized": self.device.type in {"cuda", "mps"},
            "includes_data_loading": True,
            "includes_optimizer": True,
            "includes_validation": False,
            "includes_knowledge_extraction": False,
        }


def accumulate_transport_diagnostics(accumulator, diagnostics):
    required = {
        "row_residual",
        "col_residual",
        "converged_plans",
        "total_plans",
        "iterations",
    }
    if not isinstance(diagnostics, dict) or required - diagnostics.keys():
        raise ValueError("Missing per-image transport diagnostics")
    values = {key: diagnostics[key].detach() for key in required}
    shape = values["row_residual"].shape
    if (
        len(shape) != 1
        or not shape[0]
        or any(value.shape != shape for value in values.values())
    ):
        raise ValueError("Transport diagnostics must have one value per image")
    if any(
        not torch.isfinite(value).all() or torch.any(value < 0)
        for value in values.values()
    ):
        raise ValueError("Transport diagnostics must be finite and nonnegative")
    if torch.any(values["converged_plans"] > values["total_plans"]):
        raise ValueError("Converged plans cannot exceed total transport plans")
    result = (
        dict(accumulator)
        if accumulator is not None
        else {
            "row_residual": 0.0,
            "col_residual": 0.0,
            "converged_plans": 0,
            "total_plans": 0,
            "iterations": 0,
            "examples": 0,
        }
    )
    for key in ("row_residual", "col_residual"):
        result[key] = max(result[key], float(values[key].amax().cpu()))
    result["iterations"] = max(
        result["iterations"], int(values["iterations"].amax().cpu())
    )
    for key in ("converged_plans", "total_plans"):
        result[key] += int(values[key].sum().cpu())
    result["examples"] += shape[0]
    result["converged_fraction"] = (
        result["converged_plans"] / result["total_plans"]
        if result["total_plans"]
        else 0.0
    )
    return result


def save_checkpoint(
    path,
    model,
    config,
    optimizer=None,
    scheduler=None,
    epoch=0,
    best_value=None,
    history=None,
    scaler=None,
):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": 1,
        "model": model.trainable_state_dict(),
        "config": copy.deepcopy(config),
        "class_names": list(model.class_names),
        "frozen_clip_sha256": model.frozen_fingerprint(),
        "architecture": model.architecture(),
        "epoch": int(epoch),
        "best_value": best_value,
        "history": list(history or []),
        "rng": _rng_state(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            torch.save(payload, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return payload


def _canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    expected_config=None,
    restore_rng=False,
    scaler=None,
):
    payload = torch.load(
        Path(path).expanduser().resolve(), map_location="cpu", weights_only=False
    )
    required = {
        "schema",
        "model",
        "config",
        "class_names",
        "frozen_clip_sha256",
        "architecture",
        "epoch",
    }
    if (
        not isinstance(payload, dict)
        or required - payload.keys()
        or payload["schema"] != 1
    ):
        raise ValueError("Invalid PLIKD checkpoint schema")
    if payload["frozen_clip_sha256"] != model.frozen_fingerprint():
        raise ValueError("Frozen CLIP weight fingerprint differs from the checkpoint")
    if payload["architecture"] != model.architecture():
        raise ValueError("Prompt/projector architecture differs from the checkpoint")
    if expected_config is not None and _canonical(payload["config"]) != _canonical(
        expected_config
    ):
        raise ValueError("Checkpoint configuration or training protocol changed")
    if expected_config is not None and list(payload["class_names"]) != list(
        model.class_names
    ):
        raise ValueError("Training candidate class names changed while resuming")
    model.load_trainable_state_dict(payload["model"])
    for target, key in (
        (optimizer, "optimizer"),
        (scheduler, "scheduler"),
        (scaler, "scaler"),
    ):
        if target is not None:
            if payload.get(key) is None:
                raise ValueError(f"Checkpoint has no {key} state")
            target.load_state_dict(payload[key])
    if restore_rng:
        if "rng" not in payload:
            raise ValueError("Checkpoint has no RNG state")
        _restore_rng(payload["rng"])
    return payload


def _batch(batch, device, knowledge):
    if not isinstance(batch, dict) or "images" not in batch or "labels" not in batch:
        raise ValueError("Each batch must contain images and local candidate labels")
    images = batch["images"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    features = batch.get("knowledge_features") if knowledge else None
    if features is not None:
        features = features.to(device, non_blocking=True)
    if "split" in batch:
        splits = (
            batch["split"]
            if isinstance(batch["split"], (list, tuple))
            else [batch["split"]]
        )
        allowed = {"train"} if knowledge else {"val"}
        if not set(splits) <= allowed:
            raise ValueError(
                "Training/validation batch split does not match its loader"
            )
    return images, labels, features


def _autocast(device, precision):
    enabled = precision != "float32"
    if precision not in {"float32", "amp_fp16", "amp_bf16"}:
        raise ValueError("training.precision must be float32, amp_fp16 or amp_bf16")
    if precision == "amp_fp16" and device.type != "cuda":
        raise ValueError("FP16 mixed precision training requires CUDA")
    if enabled and device.type not in {"cpu", "cuda"}:
        raise ValueError(
            "Mixed precision is supported on CPU/CUDA; use float32 on this device"
        )
    dtype = torch.float16 if precision == "amp_fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _validation(model, loader, device, precision):
    model.eval()
    count = correct = 0
    losses = 0.0
    diagnostics = None
    with torch.no_grad():
        for batch in loader:
            images, labels, _ = _batch(batch, device, knowledge=False)
            with _autocast(device, precision):
                output = model(images, labels=labels)
            loss = output["losses"]["classification"]
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite validation classification loss")
            count += len(images)
            correct += int((output["logits"].argmax(-1) == labels).sum())
            losses += float(loss) * len(images)
            diagnostics = accumulate_transport_diagnostics(
                diagnostics, output["transport_diagnostics"]
            )
    if not count:
        raise ValueError("The validation loader contains no examples")
    return {
        "classification_loss": losses / count,
        "accuracy": 100.0 * correct / count,
        "examples": count,
        "transport_diagnostics": diagnostics,
    }


def train_model(config, train_loader, val_loader, model, output, resume=None):
    settings = config.get("training", {})
    epochs = int(settings.get("epochs", 200))
    selection = settings.get("selection", "last")
    if selection not in {"last", "val_loss"}:
        raise ValueError("training.selection must be last or val_loss")
    if selection == "val_loss" and val_loader is None:
        raise ValueError(
            "Validation-loss selection needs an independent validation loader"
        )
    if len(train_loader) < 1:
        raise ValueError("Training loader is empty")
    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(
            "Training output exists; use an explicit matching checkpoint to resume"
        )
    output.mkdir(parents=True, exist_ok=True)
    seed = int(settings.get("seed", config.get("protocol", {}).get("seed", 1)))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    device = next(model.parameters()).device
    precision = settings.get("precision", "float32")
    _autocast(device, precision)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "amp_fp16")
    start = 0
    best_value = None
    history = []
    if resume:
        checkpoint = output / "last.pt" if resume is True else Path(resume)
        payload = load_checkpoint(
            checkpoint,
            model,
            optimizer,
            scheduler,
            expected_config=config,
            restore_rng=True,
            scaler=scaler,
        )
        start = int(payload["epoch"])
        best_value = payload.get("best_value")
        history = list(payload.get("history", []))
        if not 0 <= start <= epochs:
            raise ValueError(
                "Checkpoint epoch is outside the configured training schedule"
            )
    save_every = int(settings.get("save_every", 1))
    if save_every < 1:
        raise ValueError("training.save_every must be positive")
    for epoch in range(start, epochs):
        loader_seed = seed_loader_epoch(train_loader, seed, epoch)
        model.train()
        totals = {}
        diagnostics = None
        count = correct = 0
        learning_rate = optimizer.param_groups[0]["lr"]
        timer = EpochTimer(device, enabled=settings.get("measure_time", False)).start()
        for batch in train_loader:
            images, labels, features = _batch(batch, device, knowledge=True)
            if (
                features is None
                and float(config.get("loss", {}).get("knowledge_weight", 0.7)) > 0
            ):
                raise ValueError(
                    "Training knowledge features are missing; no live MLLM fallback is permitted"
                )
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, precision):
                result = model(images, knowledge_features=features, labels=labels)
                losses = result["losses"]
                loss = losses["total"]
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            scaler.scale(loss).backward()
            if settings.get("max_grad_norm") is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    float(settings["max_grad_norm"]),
                    error_if_nonfinite=True,
                )
            scaler.step(optimizer)
            scaler.update()
            count += len(images)
            correct += int((result["logits"].argmax(-1) == labels).sum())
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * len(
                    images
                )
            diagnostics = accumulate_transport_diagnostics(
                diagnostics, result["transport_diagnostics"]
            )
        timing = timer.stop()
        if not count:
            raise ValueError("Training loader yielded no examples")
        record = {
            "epoch": epoch + 1,
            "learning_rate": learning_rate,
            "train_examples": count,
            "train_accuracy": 100.0 * correct / count,
            "train_losses": {name: value / count for name, value in totals.items()},
            "loader_seed": loader_seed,
            "transport_diagnostics": diagnostics,
        }
        if timing is not None:
            record["timing"] = timing
        if val_loader is not None:
            validation = _validation(model, val_loader, device, precision)
            record["validation"] = validation
            if selection == "val_loss" and (
                best_value is None or validation["classification_loss"] < best_value
            ):
                best_value = validation["classification_loss"]
                improved = True
            else:
                improved = False
        else:
            improved = False
        history.append(record)
        scheduler.step()
        if improved:
            save_checkpoint(
                output / "best.pt",
                model,
                config,
                optimizer,
                scheduler,
                epoch + 1,
                best_value,
                history,
                scaler,
            )
        if (epoch + 1) % save_every == 0 or epoch + 1 == epochs:
            save_checkpoint(
                output / "last.pt",
                model,
                config,
                optimizer,
                scheduler,
                epoch + 1,
                best_value,
                history,
                scaler,
            )
        history_path = output / "history.json"
        temporary = history_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(history, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(history_path)
        print(json.dumps(record), flush=True)
    model.eval()
    return {
        "epochs": epochs,
        "best_value": best_value,
        "history": history,
        "last_checkpoint": str(output / "last.pt"),
        "best_checkpoint": str(output / "best.pt") if selection == "val_loss" else None,
        "selected_checkpoint": str(
            output / ("best.pt" if selection == "val_loss" else "last.pt")
        ),
    }
