from pathlib import Path

import pytest


def test_experiment_plan_covers_three_protocols_and_all_knowledge_controls():
    from main import load_config, expand_suite

    config = load_config(Path(__file__).parents[1] / "config.yaml")
    main = expand_suite(config, "main", datasets=["imagenet"], seeds=[1])
    assert {row["config"]["protocol"]["mode"] for row in main} == {
        "base_new",
        "fewshot",
        "domain",
    }
    ablations = expand_suite(config, "knowledge", datasets=["imagenet"], seeds=[1])
    assert {row["variant"] for row in ablations} == {
        "full",
        "no_visual",
        "no_functional",
        "no_contextual",
    }
    assert all(row["config"]["protocol"]["mode"] == "base_new" for row in ablations)


def test_solver_and_temperature_variants_have_distinct_run_identifiers():
    from main import load_config, expand_suite

    config = load_config(Path(__file__).parents[1] / "config.yaml")
    rows = expand_suite(config, "ot", datasets=["imagenet"], seeds=[1, 2, 3])
    assert {row["config"]["transport"]["solver"] for row in rows} == {
        "sparse",
        "entropic",
    }
    assert len({row["id"] for row in rows}) == len(rows) == 6


def test_epoch_schedule_follows_shots_and_imagenet_exception():
    from main import resolve_training

    config = {
        "training": {"epochs": "auto", "batch_size": "auto"},
        "protocol": {"shots": 16, "seed": 3},
    }
    assert resolve_training(config, "imagenet")["training"]["epochs"] == 50
    assert resolve_training(config, "fgvc_aircraft")["training"]["epochs"] == 200
    assert resolve_training(config, "imagenet")["training"]["batch_size"] == 128
    assert resolve_training(config, "fgvc_aircraft")["training"]["batch_size"] == 32
    assert resolve_training(config, "imagenet")["training"]["seed"] == 3


def test_encoding_uses_frozen_clip_model_interface(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import data
    import knowledge
    import main

    encoder = object()
    model = SimpleNamespace(clip_model=encoder, frozen_fingerprint=lambda: "weights")
    observed = []
    monkeypatch.setattr(main, "make_model", lambda *args: (model, None, "tokenize"))
    monkeypatch.setattr(data, "load_prepared", lambda *args: ["training-record"])
    monkeypatch.setattr(
        knowledge, "encode_knowledge", lambda *args: observed.append(args)
    )
    main.encode(SimpleNamespace(workdir=tmp_path), {})
    assert observed[0][3] is encoder
    assert observed[0][-1] == "weights"


def test_evaluation_rejects_a_checkpoint_from_another_protocol():
    from main import validate_checkpoint_protocol

    protocol = {
        "mode": "base_new",
        "shots": 16,
        "seed": 1,
        "class_names": ["one", "two"],
        "training_classes": [0],
        "source_sha256": "source",
    }
    payload = {
        "class_names": ["one"],
        "config": {
            "protocol": {"mode": "base_new", "shots": 16, "seed": 1},
            "data_fingerprint": {"protocol": protocol},
        },
    }
    validate_checkpoint_protocol(payload, protocol)
    changed = {**protocol, "seed": 2}
    with pytest.raises(ValueError, match="protocol|split"):
        validate_checkpoint_protocol(payload, changed)
    with pytest.raises(ValueError, match="class|candidate"):
        validate_checkpoint_protocol({**payload, "class_names": ["two"]}, protocol)
    with pytest.raises(ValueError, match="variant|experiment"):
        validate_checkpoint_protocol(payload, protocol, {"variant": "mllm_only"})
    with pytest.raises(ValueError, match="variant|experiment"):
        validate_checkpoint_protocol(payload, protocol, {"experiment": {"suite": "ot"}})


def test_offline_cli_prepares_imports_and_plans_without_loading_models(tmp_path):
    import json
    import subprocess
    import sys
    from PIL import Image

    root = Path(__file__).parents[1]
    manifest = tmp_path / "source.jsonl"
    records = []
    for label in range(3):
        for split in ("train", "val", "test"):
            image = tmp_path / f"{label}_{split}.png"
            Image.new("RGB", (4, 4), (label * 50, 0, 0)).save(image)
            records.append(
                {
                    "id": f"{label}_{split}",
                    "image": str(image),
                    "label": label,
                    "classname": f"class {label}",
                    "split": split,
                }
            )
    manifest.write_text("".join(json.dumps(row) + "\n" for row in records))
    run = tmp_path / "run"
    base = [sys.executable, str(root / "main.py"), "--workdir", str(run)]
    subprocess.run(
        base + ["--set", "protocol.shots=1", "prepare", "--manifest", str(manifest)],
        check=True,
        capture_output=True,
    )
    train = [
        json.loads(line) for line in (run / "train.jsonl").read_text().splitlines()
    ]
    imported = tmp_path / "knowledge.jsonl"
    imported.write_text(
        "".join(
            json.dumps(
                {
                    "id": row["id"],
                    "visual": "shape",
                    "functional": "use",
                    "contextual": "habitat",
                }
            )
            + "\n"
            for row in train
        )
    )
    subprocess.run(
        base + ["extract", "--import-file", str(imported)],
        check=True,
        capture_output=True,
    )
    assert len((run / "knowledge.jsonl").read_text().splitlines()) == 2
    assert not (run / "knowledge.npz").exists()
    plan = tmp_path / "plan"
    subprocess.run(
        [
            sys.executable,
            str(root / "main.py"),
            "--workdir",
            str(plan),
            "sweep",
            "--suite",
            "ot",
            "--datasets",
            "imagenet",
            "--seeds",
            "1",
        ],
        check=True,
        capture_output=True,
    )
    rows = json.loads((plan / "experiment_plan.json").read_text())
    assert len(rows) == 2
    assert not list(plan.rglob("*.pt"))


def test_native_clip_interface_and_prompt_gradient_without_pretrained_weights():
    import clip
    from clip.model import CLIP
    import torch
    from torch.nn import functional as F
    from main import load_config
    from model import PLIKD

    torch.manual_seed(7)
    backbone = CLIP(
        embed_dim=8,
        image_resolution=8,
        vision_layers=1,
        vision_width=64,
        vision_patch_size=4,
        context_length=77,
        vocab_size=49408,
        transformer_width=8,
        transformer_heads=1,
        transformer_layers=1,
    ).eval()
    config = load_config(Path(__file__).parents[1] / "config.yaml")
    config["model"].update(
        visual_prompt_length=2,
        context_tokens=2,
        projector_hidden=8,
        activation_checkpointing=False,
    )
    model = PLIKD(backbone, clip.tokenize, ["cat", "dog"], config)
    images = torch.randn(2, 3, 8, 8)
    result = model(
        images, knowledge_features=torch.randn(2, 8), labels=torch.tensor([0, 1])
    )
    with torch.no_grad():
        expected = F.normalize(backbone.encode_image(images).float(), dim=-1)
    assert torch.allclose(result["visual_global"], expected, atol=1e-6)
    assert result["logits"].shape == (2, 2)
    result["losses"]["total"].backward()
    assert model.context.grad is not None and model.context.grad.norm() > 0
    assert model.visual_prompt.grad is not None and model.visual_prompt.grad.norm() > 0
    assert all(p.grad is None for p in backbone.parameters())


def test_prediction_records_preserve_transport_convergence_and_global_labels():
    import torch
    from main import prediction_records, prediction_transport

    diagnostics = {
        "row_residual": torch.tensor([1e-7, 2e-5]),
        "col_residual": torch.zeros(2),
        "converged_plans": torch.tensor([2, 1]),
        "total_plans": torch.tensor([2, 2]),
        "iterations": torch.tensor([50, 100]),
    }
    rows = prediction_records(
        ["one", "two"], torch.tensor([[2.0, 1.0], [0.0, 3.0]]), [7, 9], diagnostics
    )
    assert [row["prediction"] for row in rows] == [7, 9]
    assert rows[1]["transport"]["iterations"] == 100
    summary = prediction_transport(rows)
    assert summary["converged_fraction"] == 0.75
    assert summary["row_residual"] == pytest.approx(2e-5)
    with pytest.raises(ValueError, match="finite|logits"):
        prediction_records(
            ["one", "two"], torch.full((2, 2), float("nan")), [7, 9], diagnostics
        )


def test_checkpoint_timing_uses_only_recorded_training_observations():
    from main import checkpoint_timing

    assert checkpoint_timing({"epoch": 2, "history": [{"epoch": 1}]}) is None
    rows = [
        {
            "epoch": epoch,
            "timing": {
                "train_seconds_per_epoch": seconds,
                "scope": "epoch_train_batches",
                "device": "cpu",
            },
        }
        for epoch, seconds in [(1, 2.0), (2, 4.0)]
    ]
    result = checkpoint_timing({"epoch": 2, "history": rows})
    assert result["train_seconds_per_epoch"] == 3
    assert result["train_timing"]["complete_training_history"]
    assert result["train_timing"]["epochs"] == [1, 2]


@pytest.fixture
def offline_evaluation(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import torch
    import data
    import main
    import train

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.calls = 0

        def forward(self, images, class_names=None):
            self.calls += 1
            return {
                "logits": torch.tensor([[2.0, 1.0], [1.0, 2.0]]),
                "transport_diagnostics": {
                    "row_residual": torch.zeros(2),
                    "col_residual": torch.zeros(2),
                    "converged_plans": torch.full((2,), 2),
                    "total_plans": torch.full((2,), 2),
                    "iterations": torch.ones(2, dtype=torch.long),
                },
            }

    records = []
    for label, identifier in enumerate(("a", "b")):
        image = tmp_path / f"{identifier}.png"
        image.write_bytes(identifier.encode())
        records.append(data.Record(identifier, str(image), label, identifier, "test"))
    config = main.load_config(Path(__file__).parents[1] / "config.yaml")
    config["protocol"]["mode"] = "fewshot"
    protocol = {
        "dataset": "imagenet",
        "mode": "fewshot",
        "shots": 16,
        "seed": 1,
        "class_names": ["a", "b"],
        "training_classes": [0, 1],
        "evaluation_classes": {"test": [0, 1]},
        "source_sha256": "source",
        "split_hashes": {"test": "testhash"},
    }
    payload = {
        "config": main.resolve_training(config, "imagenet"),
        "class_names": ["a", "b"],
        "frozen_clip_sha256": "weights",
        "epoch": 50,
        "history": [],
    }
    payload["config"]["data_fingerprint"] = {"protocol": protocol}
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"software integration checkpoint")
    model = TinyModel()
    args = SimpleNamespace(
        workdir=tmp_path,
        checkpoint=str(checkpoint),
        split=None,
        measure_time=False,
    )

    def loader(rows, *rest):
        return (
            [{"images": torch.zeros(len(rows), 3, 2, 2), "ids": [r.id for r in rows]}]
            if rows
            else []
        )

    monkeypatch.setattr(main, "validate_protocol", lambda *args: protocol)
    monkeypatch.setattr(main, "make_model", lambda *args: (model, None, None))
    monkeypatch.setattr(train, "load_checkpoint", lambda *args: payload)
    monkeypatch.setattr(data, "load_prepared", lambda *args: records)
    monkeypatch.setattr(main, "image_loader", loader)
    return SimpleNamespace(args=args, config=config, model=model, directory=tmp_path)


def test_completed_prediction_cache_is_finalized_and_tampering_rejected(
    offline_evaluation,
):
    import json
    import main

    run = offline_evaluation
    main.evaluate(run.args, run.config)
    predictions = run.directory / "predictions_test.jsonl"
    metadata = json.loads(Path(str(predictions) + ".meta.json").read_text())
    assert metadata["content_sha256"] == main.file_digest(predictions)
    assert metadata["row_count"] == 2
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    rows[0]["prediction"] = 1
    predictions.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="content changed"):
        main.evaluate(run.args, run.config)
    assert run.model.calls == 1


def test_cached_evaluation_preserves_verified_inference_measurement(offline_evaluation):
    import json
    import main

    run = offline_evaluation
    run.args.measure_time = True
    measured = main.evaluate(run.args, run.config)[0]
    path = run.directory / "predictions_test.metrics.json"
    tampered = json.loads(path.read_text())
    tampered["timing"]["seconds"] = 1234567.0
    path.write_text(json.dumps(tampered))
    run.args.measure_time = False
    cached = main.evaluate(run.args, run.config)[0]
    assert cached["timing"] == measured["timing"]
    assert run.model.calls == 1


def test_cached_measurement_rejects_modified_timing_evidence(offline_evaluation):
    import json
    import main

    run = offline_evaluation
    run.args.measure_time = True
    main.evaluate(run.args, run.config)
    metadata_path = run.directory / "predictions_test.jsonl.meta.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["inference_timing"]["timing"]["seconds"] += 1.0
    metadata_path.write_text(json.dumps(metadata))
    run.args.measure_time = False
    with pytest.raises(ValueError, match="timing|measurement"):
        main.evaluate(run.args, run.config)


@pytest.mark.parametrize(
    "changed", ["parameters", "predictions", "device", "batch_size"]
)
def test_cached_measurement_is_not_inherited_across_changed_inputs(
    offline_evaluation, changed
):
    import json
    import main

    run = offline_evaluation
    run.args.measure_time = True
    result = main.evaluate(run.args, run.config)[0]
    metadata_path = run.directory / "predictions_test.jsonl.meta.json"
    metadata = json.loads(metadata_path.read_text())
    parameters = metadata["parameters"]
    prediction_hash = result["prediction_sha256"]
    device = "cpu"
    batch_size = run.config["evaluation"]["batch_size"]
    if changed == "parameters":
        parameters = {**parameters, "checkpoint_sha256": "other"}
    elif changed == "predictions":
        prediction_hash = "other"
    elif changed == "device":
        device = "cuda:0"
    else:
        batch_size += 1
    assert (
        main.cached_inference_timing(
            metadata, parameters, prediction_hash, device, batch_size
        )
        is None
    )
