import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

import yaml

from knowledge import Journal, canonical, digest, file_digest, write_json


DATASETS = [
    "imagenet",
    "caltech101",
    "oxford_pets",
    "stanford_cars",
    "flowers102",
    "food101",
    "fgvc_aircraft",
    "sun397",
    "dtd",
    "eurosat",
    "ucf101",
]


def load_config(path, overrides=()):
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    for item in overrides:
        key, separator, value = item.partition("=")
        if not separator or any(not part for part in key.split(".")):
            raise ValueError(f"Expected key=value override: {item}")
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
            if not isinstance(target, dict):
                raise ValueError(f"Invalid nested configuration key: {key}")
        target[parts[-1]] = yaml.safe_load(value)
    teacher = config.get("teacher", {})
    endpoint = (
        os.environ.get(teacher.get("base_url_env", "OPENAI_BASE_URL"))
        or "https://api.openai.com/v1"
    )
    teacher["endpoint_sha256"] = hashlib.sha256(
        endpoint.rstrip("/").encode()
    ).hexdigest()
    return config


def save_config(path, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    temporary.replace(path)


def resolve_training(config, dataset):
    config = copy.deepcopy(config)
    training = config["training"]
    shots = int(config["protocol"]["shots"])
    training.setdefault("epoch_policy", training.get("epochs", "auto"))
    training.setdefault("batch_policy", training.get("batch_size", "auto"))
    if training.get("epochs", "auto") == "auto":
        training["epochs"] = (
            50 if dataset == "imagenet" or shots == 1 else 100 if shots <= 4 else 200
        )
    if training.get("batch_size", "auto") == "auto":
        training["batch_size"] = 128 if dataset in {"imagenet", "sun397"} else 32
    training["seed"] = int(config["protocol"]["seed"])
    return config


def metadata(workdir):
    path = Path(workdir) / "protocol.json"
    if not path.is_file():
        raise FileNotFoundError(f"Run prepare first: {path}")
    return json.loads(path.read_text())


def validate_protocol(config, workdir):
    prepared = metadata(workdir)
    for key in ("mode", "shots", "seed"):
        if config["protocol"][key] != prepared[key]:
            raise ValueError(
                f"Configuration differs from prepared data at protocol.{key}"
            )
    return prepared


def validate_checkpoint_protocol(payload, prepared, config=None):
    recorded = payload["config"].get("data_fingerprint", {}).get("protocol")
    if recorded != prepared:
        raise ValueError(
            "Checkpoint training protocol or split differs from the prepared data"
        )
    for key in ("mode", "shots", "seed"):
        if payload["config"]["protocol"][key] != prepared[key]:
            raise ValueError(
                f"Checkpoint protocol.{key} differs from the prepared data"
            )
    names = [prepared["class_names"][index] for index in prepared["training_classes"]]
    if payload["class_names"] != names:
        raise ValueError(
            "Checkpoint training candidate class names differ from the prepared data"
        )
    if config is not None:
        trained = payload["config"]
        if config.get("variant", "full") != trained.get(
            "variant", "full"
        ) or config.get("experiment") != trained.get("experiment"):
            raise ValueError(
                "Evaluation variant or experiment identity differs from the trained checkpoint"
            )


def device_for(config):
    import torch

    device = config["clip"].get("device", "auto")
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    return torch.device(device)


def make_model(config, workdir):
    import clip
    import numpy as np
    import torch
    from model import PLIKD

    prepared = validate_protocol(config, workdir)
    seed = int(config["protocol"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    backbone, preprocess = clip.load(
        config["clip"]["backbone"],
        device="cpu",
        jit=False,
        download_root=config["clip"].get("download_root"),
    )
    names = [prepared["class_names"][index] for index in prepared["training_classes"]]
    model = PLIKD(backbone, clip.tokenize, names, config).to(device_for(config))
    return model, preprocess, clip.tokenize


def prepare(args, config):
    from data import prepare_protocol

    selected = copy.deepcopy(config)
    if args.mode:
        selected["protocol"]["mode"] = args.mode
    targets = {}
    for value in args.target_manifest:
        name, separator, path = value.partition("=")
        if not separator:
            raise ValueError("Target manifests use domain=path syntax")
        if name in targets:
            raise ValueError("Duplicate target domain")
        targets[name] = path
    prepared = prepare_protocol(
        args.workdir,
        mode=selected["protocol"]["mode"],
        shots=int(selected["protocol"]["shots"]),
        seed=int(selected["protocol"]["seed"]),
        root=args.root,
        dataset=args.dataset,
        manifest=args.manifest,
        target_manifests=targets or None,
        domains=args.domains,
    )
    save_config(Path(args.workdir) / "config.yaml", selected)
    print(
        canonical(
            {
                "dataset": prepared["dataset"],
                "mode": prepared["mode"],
                "counts": prepared["counts"],
            }
        )
    )


def extract(args, config):
    from data import load_prepared
    from knowledge import extract_knowledge, import_knowledge

    validate_protocol(config, args.workdir)
    records = load_prepared(args.workdir, "train")
    output = Path(args.workdir) / "knowledge.jsonl"
    if args.import_file:
        import_knowledge(records, args.import_file, output, config)
    else:
        extract_knowledge(records, output, config)


def encode(args, config):
    from data import load_prepared
    from knowledge import encode_knowledge

    model, _, tokenize = make_model(config, args.workdir)
    records = load_prepared(args.workdir, "train")
    encode_knowledge(
        records,
        Path(args.workdir) / "knowledge.jsonl",
        Path(args.workdir) / "knowledge.npz",
        model.clip_model,
        tokenize,
        config,
        model.frozen_fingerprint(),
    )


def seed_worker(worker_id):
    import numpy as np
    import torch

    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def image_loader(
    records, preprocess, candidate_ids, config, training=False, knowledge=None
):
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    from data import ImageDataset, collate_images

    if training:
        resolution = int(getattr(preprocess.transforms[0], "size", 224))
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    resolution, interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        settings = config["training"]
    else:
        transform = preprocess
        settings = config["evaluation"]
    dataset = ImageDataset(records, transform, candidate_ids, knowledge)
    generator = torch.Generator().manual_seed(int(config["protocol"]["seed"]))
    workers = int(settings.get("workers", 4))
    return DataLoader(
        dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=training,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_images,
        generator=generator,
        worker_init_fn=seed_worker,
        persistent_workers=workers > 0 and not training,
    )


def train(args, config):
    from data import load_prepared
    from knowledge import (
        FIELDS,
        load_knowledge_features,
        training_signature,
        validate_knowledge_source,
    )
    from train import train_model

    prepared = validate_protocol(config, args.workdir)
    config = resolve_training(config, prepared["dataset"])
    model, preprocess, _ = make_model(config, args.workdir)
    records = load_prepared(args.workdir, "train")
    validate_knowledge_source(records, Path(args.workdir) / "knowledge.jsonl", config)
    features, provenance = load_knowledge_features(
        Path(args.workdir) / "knowledge.npz", records
    )
    expected = training_signature(
        records,
        config,
        source_sha256=file_digest(Path(args.workdir) / "knowledge.jsonl"),
        backbone_fingerprint=model.frozen_fingerprint(),
        dimensions=config["knowledge"].get("dimensions", list(FIELDS)),
        content_tokens=int(config["knowledge"].get("content_tokens", 64)),
    )
    if provenance != expected:
        raise ValueError(
            "Training knowledge features have stale source/model/config provenance"
        )
    config["data_fingerprint"] = {
        "protocol": prepared,
        "knowledge": file_digest(Path(args.workdir) / "knowledge.npz"),
        "images": digest(
            [(record.id, file_digest(record.image)) for record in records]
        ),
    }
    training = image_loader(
        records, preprocess, prepared["training_classes"], config, True, features
    )
    validation_records = load_prepared(args.workdir, "val")
    validation = (
        image_loader(
            validation_records, preprocess, prepared["training_classes"], config
        )
        if validation_records
        else None
    )
    output = Path(args.workdir) / "checkpoints"
    result = train_model(
        config, training, validation, model, output, resume=args.resume
    )
    write_json(Path(args.workdir) / "training_summary.json", result)


def comparison_config(config):
    training = config["training"]
    omitted = {
        "seed",
        "workers",
        "save_every",
        "epochs",
        "batch_size",
        "epoch_policy",
        "batch_policy",
        "measure_time",
    }
    teacher = {
        key: value
        for key, value in config["teacher"].items()
        if key not in {"cache", "api_key_env", "base_url_env", "retries"}
    }
    return {
        "backbone": config["clip"]["backbone"],
        "model": config["model"],
        "loss": config["loss"],
        "transport": config["transport"],
        "knowledge": {
            key: value
            for key, value in config["knowledge"].items()
            if key != "batch_size"
        },
        "teacher": teacher,
        "training": {
            key: value for key, value in training.items() if key not in omitted
        },
        "epochs": training.get("epoch_policy", training.get("epochs", "auto")),
        "batch": training.get("batch_policy", training.get("batch_size", "auto")),
    }


def prediction_records(identifiers, logits, candidate_ids, diagnostics):
    import torch
    from train import accumulate_transport_diagnostics

    if (
        logits.shape != (len(identifiers), len(candidate_ids))
        or not torch.isfinite(logits).all()
    ):
        raise ValueError(
            "Classification logits must be finite and aligned with images and candidates"
        )
    accumulate_transport_diagnostics(None, diagnostics)
    if len(diagnostics["row_residual"]) != len(identifiers):
        raise ValueError("Transport diagnostics do not cover the predicted images")
    vectors = {key: value.detach().cpu().tolist() for key, value in diagnostics.items()}
    predictions = logits.argmax(-1).cpu().tolist()
    return [
        {
            "id": identifier,
            "prediction": candidate_ids[prediction],
            "transport": {key: value[index] for key, value in vectors.items()},
        }
        for index, (identifier, prediction) in enumerate(zip(identifiers, predictions))
    ]


def prediction_transport(rows):
    import torch
    from train import accumulate_transport_diagnostics

    if not rows or any("transport" not in row for row in rows):
        raise ValueError("Prediction records must include transport diagnostics")
    keys = (
        "row_residual",
        "col_residual",
        "converged_plans",
        "total_plans",
        "iterations",
    )
    return accumulate_transport_diagnostics(
        None,
        {key: torch.tensor([row["transport"][key] for row in rows]) for key in keys},
    )


def checkpoint_timing(payload):
    observations = [
        (record["epoch"], record["timing"])
        for record in payload.get("history", [])
        if record.get("timing") is not None
    ]
    if not observations:
        return None
    protocol = {
        key: value
        for key, value in observations[0][1].items()
        if key != "train_seconds_per_epoch"
    }
    seconds = []
    for _, timing in observations:
        value = timing["train_seconds_per_epoch"]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("Recorded training timing must be finite and positive")
        if {
            key: item
            for key, item in timing.items()
            if key != "train_seconds_per_epoch"
        } != protocol:
            raise ValueError("Recorded training timing protocols differ across epochs")
        seconds.append(value)
    return {
        "train_seconds_per_epoch": statistics.fmean(seconds),
        "train_timing": {
            **protocol,
            "epochs": [epoch for epoch, _ in observations],
            "observations": len(observations),
            "complete_training_history": len(observations) == payload["epoch"],
        },
    }


def cached_inference_timing(metadata, parameters, prediction_hash, device, batch_size):
    evidence = metadata.get("inference_timing")
    if evidence is None:
        return None
    if not isinstance(evidence, dict):
        raise ValueError("Invalid cached inference timing evidence")
    body = {key: value for key, value in evidence.items() if key != "sha256"}
    if evidence.get("sha256") != digest(body):
        raise ValueError("Cached inference timing evidence changed")
    timing = body.get("timing")
    if not isinstance(timing, dict) or any(
        isinstance(timing.get(key), bool)
        or not isinstance(timing.get(key), (int, float))
        or not math.isfinite(timing[key])
        or timing[key] <= 0
        for key in ("seconds", "images", "inference_images_per_second")
    ):
        raise ValueError("Invalid cached inference measurement")
    if timing.get("scope") != "model_and_dataloader" or not math.isclose(
        timing["inference_images_per_second"], timing["images"] / timing["seconds"]
    ):
        raise ValueError("Inconsistent cached inference measurement")
    if (
        body.get("parameters_sha256") != digest(parameters)
        or body.get("prediction_sha256") != prediction_hash
        or timing.get("device") != str(device)
        or timing.get("batch_size") != batch_size
    ):
        return None
    return copy.deepcopy(timing)


def evaluate(args, config):
    import torch
    from data import load_prepared
    from evaluate import accuracy
    from train import load_checkpoint, synchronize_device

    prepared = validate_protocol(config, args.workdir)
    selected = (
        "best.pt"
        if config["training"].get("selection", "last") == "val_loss"
        else "last.pt"
    )
    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else Path(args.workdir) / "checkpoints" / selected
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"A trained checkpoint is required: {checkpoint}")
    model, preprocess, _ = make_model(config, args.workdir)
    payload = load_checkpoint(checkpoint, model)
    validate_checkpoint_protocol(payload, prepared, config)
    training_config = payload["config"]
    current_recipe = comparison_config(resolve_training(config, prepared["dataset"]))
    if comparison_config(training_config) != current_recipe:
        raise ValueError(
            "Evaluation recipe differs from the checkpoint training recipe"
        )
    current_recipe["frozen_clip_sha256"] = payload["frozen_clip_sha256"]
    model.eval()
    splits = [args.split] if args.split else list(prepared["evaluation_classes"])
    results = []
    for split in splits:
        if split not in prepared["evaluation_classes"]:
            raise ValueError("Only declared evaluation splits may be classified")
        records = load_prepared(args.workdir, split)
        ids = prepared["evaluation_classes"][split]
        names = [prepared["class_names"][index] for index in ids]
        parameters = {
            "checkpoint_sha256": file_digest(checkpoint),
            "configuration": config,
            "protocol": prepared,
            "split": split,
            "images": digest([(r.id, file_digest(r.image)) for r in records]),
            "candidate_ids": ids,
            "prediction_schema": 2,
        }
        output = Path(args.workdir) / f"predictions_{split}.jsonl"
        journal = Journal(output, parameters)
        pending = (
            records
            if args.measure_time
            else [r for r in records if r.id not in journal.rows]
        )
        loader = image_loader(pending, preprocess, ids, config)
        device = next(model.parameters()).device
        synchronize_device(device)
        start = time.perf_counter()
        with torch.no_grad():
            for batch in loader:
                output_batch = model(batch["images"].to(device), class_names=names)
                predicted = prediction_records(
                    batch["ids"],
                    output_batch["logits"],
                    ids,
                    output_batch["transport_diagnostics"],
                )
                for row in predicted:
                    identifier, label = row["id"], row["prediction"]
                    if identifier in journal.rows:
                        if journal.rows[identifier]["prediction"] != label:
                            raise ValueError(
                                "A repeated timed inference changed a cached class prediction"
                            )
                    else:
                        journal.append(row)
        synchronize_device(device)
        elapsed = time.perf_counter() - start
        result = {
            **accuracy([asdict(r) for r in records], list(journal.rows.values())),
            "dataset": prepared["dataset"],
            "mode": prepared["mode"],
            "split": split,
            "shots": prepared["shots"],
            "seed": prepared["seed"],
            "variant": config.get("variant", "full"),
            "comparison": current_recipe,
            "candidate_ids": ids,
            "checkpoint_sha256": parameters["checkpoint_sha256"],
            "manifest_sha256": prepared["split_hashes"][split],
            "checkpoint_epoch": payload["epoch"],
            "checkpoint_file": checkpoint.name,
            "protocol": prepared,
            "data_identity": {
                "source_sha256": prepared["source_sha256"],
                "class_names": prepared["class_names"],
                "evaluation_classes": prepared["evaluation_classes"],
                "evaluation_split_hashes": {
                    name: prepared["split_hashes"][name]
                    for name in prepared["evaluation_classes"]
                },
            },
            "prediction_sha256": file_digest(output),
            "source": "measured",
            "config": config,
            "transport_diagnostics": prediction_transport(list(journal.rows.values())),
        }
        if config.get("experiment"):
            result["experiment"] = config["experiment"]
        journal.finalize()
        metadata_path = journal.path.with_suffix(journal.path.suffix + ".meta.json")
        cache_metadata = json.loads(metadata_path.read_text())
        training_timing = checkpoint_timing(payload)
        if training_timing is not None:
            result["timing"] = training_timing
        if args.measure_time:
            measured_timing = {
                "inference_images_per_second": len(records) / elapsed,
                "seconds": elapsed,
                "images": len(records),
                "scope": "model_and_dataloader",
                "device": str(device),
                "batch_size": config["evaluation"]["batch_size"],
            }
            evidence = {
                "parameters_sha256": digest(parameters),
                "prediction_sha256": result["prediction_sha256"],
                "timing": measured_timing,
            }
            cache_metadata["inference_timing"] = {
                **evidence,
                "sha256": digest(evidence),
            }
            write_json(metadata_path, cache_metadata)
        measured_timing = cached_inference_timing(
            cache_metadata,
            parameters,
            result["prediction_sha256"],
            device,
            config["evaluation"]["batch_size"],
        )
        if measured_timing is not None:
            result.setdefault("timing", {}).update(measured_timing)
            result["inference_timing_provenance"] = {
                key: value
                for key, value in cache_metadata["inference_timing"].items()
                if key != "timing"
            }
        write_json(output.with_suffix(".metrics.json"), result)
        print(
            canonical(
                {
                    "split": split,
                    "accuracy": result["accuracy"],
                    "total": result["total"],
                }
            )
        )
        results.append(result)
    return results


def expand_suite(config, suite, datasets=None, seeds=None):
    datasets = list(datasets or DATASETS)
    seeds = list(seeds or config["protocol"].get("seeds", [1, 2, 3]))
    if suite == "all":
        return [
            row
            for name in ("main", "knowledge", "dual", "ot", "epsilon", "weights")
            for row in expand_suite(config, name, datasets, seeds)
        ]
    settings = []
    if suite == "main":
        settings = [
            (mode, "full", {}, "protocol", mode)
            for mode in ("base_new", "fewshot", "domain")
        ]
    elif suite == "knowledge":
        settings = [("base_new", "full", {}, "dimension", "all")]
        for removed in ("visual", "functional", "contextual"):
            settings.append(
                (
                    "base_new",
                    f"no_{removed}",
                    {
                        "knowledge.dimensions": [
                            v
                            for v in ("visual", "functional", "contextual")
                            if v != removed
                        ]
                    },
                    "dimension",
                    removed,
                )
            )
    elif suite == "dual":
        for shots in (1, 2, 4, 8, 16):
            settings.extend(
                [
                    ("base_new", "full", {"protocol.shots": shots}, "shots", shots),
                    (
                        "base_new",
                        "mllm_only",
                        {"protocol.shots": shots, "loss.visual_weight": 0.0},
                        "shots",
                        shots,
                    ),
                ]
            )
    elif suite == "ot":
        settings = [
            ("base_new", variant, {"transport.solver": solver}, "solver", solver)
            for variant, solver in (("full", "sparse"), ("entropic", "entropic"))
        ]
    elif suite == "epsilon":
        datasets = ["imagenet"]
        settings = [
            ("base_new", "full", {"transport.epsilon": value}, "epsilon", value)
            for value in (0.5, 1.0, 2.0, 3.0)
        ]
    elif suite == "weights":
        datasets = ["imagenet"]
        settings = [
            (
                "base_new",
                "full",
                {"loss.knowledge_weight": i / 10, "loss.visual_weight": 1 - i / 10},
                "knowledge_weight",
                i / 10,
            )
            for i in range(9, 0, -1)
        ]
        settings += [
            (
                "base_new",
                "full",
                {"loss.classification_weight": i / 10},
                "classification_weight",
                i / 10,
            )
            for i in range(10, 1, -1)
        ]
    else:
        raise ValueError(f"Unknown experiment suite: {suite}")
    rows = []
    for mode, variant, changes, axis, value in settings:
        for dataset in ["imagenet"] if mode == "domain" else datasets:
            for seed in seeds:
                current = copy.deepcopy(config)
                current["protocol"].update(mode=mode, seed=seed)
                current["variant"] = variant
                for key, item in changes.items():
                    parent, child = key.split(".", 1)
                    current[parent][child] = item
                group_config = copy.deepcopy(config)
                group_config["protocol"].pop("seed", None)
                current["experiment"] = {
                    "suite": suite,
                    "axis": axis,
                    "value": value,
                    "group": digest({"suite": suite, "config": group_config})[:16],
                }
                identifier = (
                    f"{dataset}_{mode}_{variant}_s{seed}_{digest(current)[:12]}"
                )
                rows.append(
                    {
                        "id": identifier,
                        "dataset": dataset,
                        "mode": mode,
                        "variant": variant,
                        "config": current,
                    }
                )
    return rows


def sweep(args, config):
    rows = expand_suite(config, args.suite, args.datasets, args.seeds)
    destination = Path(args.workdir)
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "experiment_plan.json", rows)
    for row in rows:
        run = destination / row["id"]
        run.mkdir(parents=True, exist_ok=True)
        run_config = run / "requested.yaml"
        save_config(run_config, row["config"])
        if args.execute:
            if not args.root:
                raise ValueError(
                    "Executing a sweep requires --root with prepared benchmark datasets"
                )
            base = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--config",
                str(run_config.resolve()),
                "--workdir",
                str(run.resolve()),
            ]
            if not (run / "protocol.json").is_file():
                subprocess.run(
                    base
                    + [
                        "prepare",
                        "--root",
                        str(Path(args.root).resolve()),
                        "--dataset",
                        row["dataset"],
                    ],
                    check=True,
                )
            for command in ("extract", "encode"):
                subprocess.run(base + [command], check=True)
            command = base + ["train"]
            if (run / "checkpoints" / "last.pt").is_file():
                command += [
                    "--resume",
                    str((run / "checkpoints" / "last.pt").resolve()),
                ]
            subprocess.run(command, check=True)
            subprocess.run(base + ["evaluate"], check=True)
    print(
        canonical(
            {
                "planned_runs": len(rows),
                "executed": bool(args.execute),
                "plan": str(destination / "experiment_plan.json"),
            }
        )
    )


def parser():
    cli = argparse.ArgumentParser(
        description="PLIKD: instance-aware prompt learning with sparse optimal transport"
    )
    cli.add_argument("--config")
    cli.add_argument("--workdir", default="runs/default")
    cli.add_argument("--set", action="append", default=[])
    commands = cli.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--root")
    p.add_argument("--dataset")
    p.add_argument("--manifest")
    p.add_argument("--mode", choices=("base_new", "fewshot", "domain"))
    p.add_argument("--target-manifest", action="append", default=[])
    p.add_argument("--domains", nargs="+")
    p = commands.add_parser("extract")
    p.add_argument("--import-file")
    commands.add_parser("encode")
    p = commands.add_parser("train")
    p.add_argument("--resume")
    p = commands.add_parser("evaluate")
    p.add_argument("--checkpoint")
    p.add_argument("--split")
    p.add_argument("--measure-time", action="store_true")
    p = commands.add_parser("sweep")
    p.add_argument(
        "--suite",
        choices=("main", "knowledge", "dual", "ot", "epsilon", "weights", "all"),
        default="main",
    )
    p.add_argument("--datasets", nargs="+")
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--root")
    p.add_argument("--execute", action="store_true")
    return cli


def main():
    args = parser().parse_args()
    saved = Path(args.workdir) / "config.yaml"
    config_path = (
        Path(args.config)
        if args.config
        else saved
        if saved.is_file() and args.command != "prepare"
        else Path(__file__).with_name("config.yaml")
    )
    config = load_config(config_path, args.set)
    if args.command not in ("prepare", "sweep"):
        validate_protocol(config, args.workdir)
    return globals()[args.command](args, config)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
