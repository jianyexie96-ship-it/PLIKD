import json

import pytest


def metric(
    dataset="imagenet",
    seed=1,
    split="test",
    mode="fewshot",
    value=80.0,
    variant="full",
    **kwargs,
):
    return {
        "dataset": dataset,
        "mode": mode,
        "split": split,
        "seed": seed,
        "shots": 16,
        "accuracy": value,
        "total": 100,
        "variant": variant,
        "comparison": {},
        **kwargs,
    }


def test_accuracy_uses_global_identity_and_exact_image_coverage():
    from evaluate import accuracy

    truth = [{"id": "a", "label": 7}, {"id": "b", "label": 9}]
    result = accuracy(
        truth, [{"id": "b", "prediction": 7}, {"id": "a", "prediction": 7}]
    )
    assert result["accuracy"] == 50 and result["total"] == 2
    with pytest.raises(ValueError, match="coverage"):
        accuracy(truth, [{"id": "a", "prediction": 7}])


def test_base_new_pairs_report_harmonic_mean_per_seed():
    from evaluate import summarize

    rows = []
    for seed, base, new in [(1, 80.0, 60.0), (2, 90.0, 50.0)]:
        for split, value in [("test_base", base), ("test_new", new)]:
            rows.append(
                {
                    "dataset": "fixture",
                    "mode": "base_new",
                    "split": split,
                    "seed": seed,
                    "shots": 16,
                    "accuracy": value,
                    "total": 100,
                    "variant": "full",
                    "comparison": {"temperature": 0.07},
                }
            )
    result = summarize(rows)
    paired = result["base_new"]
    assert len(paired) == 1
    assert paired[0]["base"] == 85 and paired[0]["new"] == 55
    assert paired[0]["harmonic_of_means"] == pytest.approx(2 * 85 * 55 / 140)
    assert paired[0]["mean_harmonic"] == pytest.approx(
        (2 * 80 * 60 / 140 + 2 * 90 * 50 / 140) / 2
    )


def test_missing_new_class_results_are_never_presented_as_complete():
    from evaluate import summarize

    row = {
        "dataset": "fixture",
        "mode": "base_new",
        "split": "test_base",
        "seed": 1,
        "shots": 16,
        "accuracy": 80,
        "total": 100,
        "variant": "full",
        "comparison": {},
    }
    result = summarize([row])
    assert not result["base_new"]
    assert result["incomplete_pairs"]


def test_distinct_solver_definitions_are_not_merged():
    from evaluate import summarize

    rows = []
    for regularized in [False, True]:
        rows.append(
            {
                "dataset": "fixture",
                "mode": "fewshot",
                "split": "test",
                "seed": 1,
                "shots": 16,
                "accuracy": 80,
                "total": 100,
                "variant": "full",
                "comparison": {"regularized": regularized},
            }
        )
    assert len(summarize(rows)["accuracy"]) == 2


def test_different_dataset_releases_are_not_merged_as_repeated_seeds():
    from evaluate import summarize

    rows = [
        metric(seed=1, data_identity={"source_sha256": "first"}),
        metric(seed=2, data_identity={"source_sha256": "second"}),
    ]
    groups = summarize(rows)["accuracy"]
    assert len(groups) == 2
    assert all(group["n_seeds"] == 1 for group in groups)


def test_cross_dataset_average_rejects_mixed_releases_across_seeds():
    from evaluate import summarize

    rows = [
        metric(seed=1, data_identity={"source_sha256": "first"}),
        metric(seed=2, data_identity={"source_sha256": "second"}),
        metric("caltech101", seed=1, data_identity={"source_sha256": "stable"}),
        metric("caltech101", seed=2, data_identity={"source_sha256": "stable"}),
    ]
    with pytest.raises(ValueError, match="data identity|evaluation identity|release"):
        summarize(rows)


def test_base_new_derived_rows_preserve_both_splits_evaluation_identity(monkeypatch):
    import evaluate

    captured = []
    original = evaluate._cross_dataset

    def capture(rows, paired=False):
        rows = list(rows)
        if paired:
            captured.extend(rows)
        return original(rows, paired)

    monkeypatch.setattr(evaluate, "_cross_dataset", capture)
    rows = []
    for dataset in ("imagenet", "caltech101"):
        for seed in (1, 2):
            for split, candidates in (("test_base", [0]), ("test_new", [1])):
                rows.append(
                    metric(
                        dataset,
                        seed=seed,
                        mode="base_new",
                        split=split,
                        candidate_ids=candidates,
                        manifest_sha256=f"{dataset}_{split}",
                        protocol={
                            "source_sha256": dataset,
                            "evaluation_classes": {"test_base": [0], "test_new": [1]},
                            "split_hashes": {
                                "train": f"seed_{seed}",
                                "test_base": "base",
                                "test_new": "new",
                            },
                        },
                    )
                )
    evaluate.summarize(rows)
    assert len(captured) == 4
    for row in captured:
        identity = json.loads(evaluate._evaluation_identity(row))
        pair = identity["paired_evaluation_identity"]
        assert set(pair) == {"test_base", "test_new"}
        for split, candidates in (("test_base", [0]), ("test_new", [1])):
            assert pair[split]["candidate_ids"] == candidates
            assert pair[split]["manifest_sha256"] == f"{row['dataset']}_{split}"
            assert pair[split]["evaluation_protocol"]["source_sha256"] == row["dataset"]
            assert (
                "train"
                not in pair[split]["evaluation_protocol"]["evaluation_split_hashes"]
            )
    captured[-1]["paired_evaluation_identity"]["test_new"]["manifest_sha256"] = "other"
    with pytest.raises(ValueError, match="evaluation identity|release"):
        original(captured, paired=True)


@pytest.mark.parametrize("identity_field", ["data_identity", "protocol"])
def test_ablation_plots_reject_release_changes_across_seeds(tmp_path, identity_field):
    from evaluate import generate_report

    rows = []
    for seed in (1, 2):
        for variant, value in (("full", "all"), ("no_visual", "visual")):
            for split in ("test_base", "test_new"):
                rows.append(
                    metric(
                        seed=seed,
                        mode="base_new",
                        split=split,
                        variant=variant,
                        experiment={
                            "suite": "knowledge",
                            "axis": "dimension",
                            "value": value,
                            "group": "fixture",
                        },
                        **{identity_field: {"source_sha256": f"release_{seed}"}},
                    )
                )
    result = generate_report(rows, tmp_path)
    assert result["comparison_coverage"]
    assert all(not value["matched"] for value in result["comparison_coverage"])
    assert all("identity" in value["reason"] for value in result["comparison_coverage"])
    assert not (tmp_path / "figure3_measured.png").exists()


def test_ablation_plots_allow_training_changes_across_seeds(tmp_path):
    from evaluate import generate_report

    rows = []
    for seed in (1, 2):
        for variant, value in (("full", "all"), ("no_visual", "visual")):
            for split in ("test_base", "test_new"):
                rows.append(
                    metric(
                        seed=seed,
                        mode="base_new",
                        split=split,
                        variant=variant,
                        checkpoint_sha256=f"{variant}_{seed}",
                        protocol={
                            "seed": seed,
                            "source_sha256": "same",
                            "split_hashes": {
                                "train": f"{variant}_{seed}",
                                "val": f"validation_{seed}",
                                "test_base": "base",
                                "test_new": "new",
                            },
                        },
                        experiment={
                            "suite": "knowledge",
                            "axis": "dimension",
                            "value": value,
                            "group": "fixture",
                        },
                    )
                )
    result = generate_report(rows, tmp_path)
    assert all(value["matched"] for value in result["comparison_coverage"])
    assert (tmp_path / "figure3_measured.png").exists()


@pytest.mark.parametrize("changed", ["data_identity", "candidates", "evaluation_split"])
def test_ablation_plots_require_matching_evaluation_identity(tmp_path, changed):
    from evaluate import generate_report

    rows = []
    for variant, value in (("full", "all"), ("no_visual", "visual")):
        for split in ("test_base", "test_new"):
            row = metric(
                mode="base_new",
                split=split,
                variant=variant,
                candidate_ids=[0, 1],
                data_identity={"source_sha256": "same"},
                protocol={
                    "source_sha256": "same",
                    "evaluation_classes": {"test_base": [0, 1], "test_new": [2, 3]},
                    "split_hashes": {
                        "train": variant,
                        "test_base": "base",
                        "test_new": "new",
                    },
                },
                experiment={
                    "suite": "knowledge",
                    "axis": "dimension",
                    "value": value,
                    "group": "fixture",
                },
            )
            if variant == "no_visual":
                if changed == "data_identity":
                    row["data_identity"] = {"source_sha256": "different"}
                elif changed == "candidates":
                    row["candidate_ids"] = [1, 0]
                else:
                    row["protocol"]["split_hashes"]["test_base"] = "different"
            rows.append(row)
    result = generate_report(rows, tmp_path)
    assert result["comparison_coverage"]
    assert all(not value["matched"] for value in result["comparison_coverage"])
    assert all("identity" in value["reason"] for value in result["comparison_coverage"])
    assert not (tmp_path / "figure3_measured.png").exists()


def test_ablation_identity_does_not_compare_training_checkpoint_or_sample_hashes(
    tmp_path,
):
    from evaluate import generate_report

    rows = []
    for variant, value in (("full", "all"), ("no_visual", "visual")):
        for split in ("test_base", "test_new"):
            rows.append(
                metric(
                    mode="base_new",
                    split=split,
                    variant=variant,
                    data_identity={"source_sha256": "same"},
                    checkpoint_sha256=variant,
                    protocol={
                        "source_sha256": "same",
                        "evaluation_classes": {"test_base": [0], "test_new": [1]},
                        "split_hashes": {
                            "train": variant,
                            "test_base": "base",
                            "test_new": "new",
                        },
                    },
                    experiment={
                        "suite": "knowledge",
                        "axis": "dimension",
                        "value": value,
                        "group": "fixture",
                    },
                )
            )
    result = generate_report(rows, tmp_path)
    assert all(value["matched"] for value in result["comparison_coverage"])
    assert (tmp_path / "figure3_measured.png").exists()


def test_invalid_predictions_are_wrong_and_duplicate_ids_are_rejected():
    from evaluate import accuracy

    truth = [{"id": "a", "label": 7}, {"id": "b", "label": 9}]
    result = accuracy(
        truth, [{"id": "a", "prediction": None}, {"id": "b", "prediction": True}]
    )
    assert result["accuracy"] == 0 and result["invalid"] == 2
    with pytest.raises(ValueError, match="coverage|duplicate"):
        accuracy(truth, [{"id": "a", "prediction": 7}, {"id": "a", "prediction": 9}])


def test_base_new_mismatched_protocols_do_not_form_a_pair():
    from evaluate import summarize

    rows = [
        metric(mode="base_new", split="test_base", protocol={"source_sha256": "one"}),
        metric(mode="base_new", split="test_new", protocol={"source_sha256": "two"}),
    ]
    result = summarize(rows)
    assert not result["base_new"]
    assert result["incomplete_pairs"][0]["reason"] == "protocol mismatch"


def test_dataset_averages_are_arithmetic_and_seeds_are_aligned():
    from evaluate import summarize

    rows = [
        metric(value=50.0, total=10000),
        metric("caltech101", value=100.0, total=10),
        metric(seed=2, value=90.0, total=10000),
    ]
    result = summarize(rows)["cross_dataset_accuracy"][0]
    assert result["mean"] == 75
    assert result["seeds"] == [1]
    assert result["same_seed_coverage"] is False
    assert result["complete_11_datasets"] is False
    assert result["dataset_count"] == 2


def test_duplicate_seed_metrics_and_nonfinite_accuracy_are_rejected():
    from evaluate import summarize

    with pytest.raises(ValueError, match="duplicate|Duplicate"):
        summarize([metric(), metric()])
    with pytest.raises(ValueError, match="accuracy"):
        summarize([metric(value=float("nan"))])


def test_domain_source_and_four_targets_are_separate_from_eleven_dataset_average():
    from evaluate import summarize

    names = ["test", "imagenet_v2", "imagenet_sketch", "imagenet_a", "imagenet_r"]
    rows = [
        metric(mode="domain", split=name, value=float(60 + index))
        for index, name in enumerate(names)
    ]
    result = summarize(rows)
    assert not result["cross_dataset_accuracy"]
    assert result["domain"][0]["complete_source_and_four_targets"] is True
    assert result["domain"][0]["target_mean"] == pytest.approx(62.5)


def test_domain_summary_rejects_mixed_releases_across_seeds():
    from evaluate import summarize

    rows = [
        metric(
            mode="domain",
            split=split,
            seed=seed,
            data_identity={"source_sha256": f"release_{seed}"},
        )
        for seed in (1, 2)
        for split in (
            "test",
            "imagenet_v2",
            "imagenet_sketch",
            "imagenet_a",
            "imagenet_r",
        )
    ]
    with pytest.raises(ValueError, match="evaluation identity|release"):
        summarize(rows)


def test_report_exports_only_measured_rows_and_isolates_paper_reference(tmp_path):
    from evaluate import generate_report

    source = tmp_path / "metrics.json"
    source.write_text(json.dumps(metric(value=42.0)))
    output = tmp_path / "report"
    report = generate_report([str(source)], output)
    assert report["accuracy"][0]["mean"] == 42
    assert (output / "accuracy.csv").is_file()
    reference = json.loads((output / "paper_reference.json").read_text())
    assert reference["table5_fewshot"]["imagenet"] == 67.39
    assert len(reference["table1_base_new"]["datasets"]) == 11
    measured = json.loads((output / "measured_rows.json").read_text())
    assert measured[0]["accuracy"] == 42
    assert not report["timing"]
    assert not list(output.glob("figure*.png"))


def test_empty_report_never_draws_paper_reference_as_measured_points(tmp_path):
    from evaluate import generate_report

    report = generate_report([], tmp_path / "report")
    assert not report["accuracy"]
    assert not list((tmp_path / "report").glob("*.png"))


def test_experiment_plots_require_identical_dataset_seed_coverage(tmp_path):
    from evaluate import generate_report

    rows = []
    for variant in ("full", "no_visual"):
        for seed in (1, 2):
            rows.append(
                metric(
                    seed=seed,
                    variant=variant,
                    experiment={
                        "suite": "knowledge",
                        "axis": "variant",
                        "value": variant,
                        "group": "fixture",
                    },
                )
            )
    result = generate_report(rows, tmp_path / "balanced")
    assert (tmp_path / "balanced" / "figure3_measured.png").is_file()
    assert all(item["matched"] for item in result["comparison_coverage"])
    result = generate_report(rows[:-1], tmp_path / "unbalanced")
    assert not (tmp_path / "unbalanced" / "figure3_measured.png").exists()
    assert any(not item["matched"] for item in result["comparison_coverage"])


def test_main_suite_names_route_dual_ot_and_weight_outputs(tmp_path):
    from evaluate import generate_report

    rows = []
    for suite, axis, settings in [
        (
            "dual",
            "shots",
            [("full", 1), ("mllm_only", 1), ("full", 2), ("mllm_only", 2)],
        ),
        ("ot", "solver", [("full", "sparse"), ("entropic", "entropic")]),
        ("weights", "knowledge_weight", [("full", 0.7), ("full", 0.8)]),
        ("weights", "classification_weight", [("full", 0.8), ("full", 0.9)]),
    ]:
        for variant, value in settings:
            for split in ("test_base", "test_new"):
                rows.append(
                    metric(
                        mode="base_new",
                        split=split,
                        variant=variant,
                        shots=value if axis == "shots" else 16,
                        experiment={
                            "suite": suite,
                            "axis": axis,
                            "value": value,
                            "group": "fixture",
                        },
                    )
                )
    generate_report(rows, tmp_path / "report")
    assert (tmp_path / "report" / "figure4_measured.png").is_file()
    assert (tmp_path / "report" / "figure5_measured.png").is_file()
    assert (
        len((tmp_path / "report" / "table6_measured.csv").read_text().splitlines()) == 3
    )
    assert (
        len((tmp_path / "report" / "table7_measured.csv").read_text().splitlines()) == 3
    )
    assert (tmp_path / "report" / "weights_knowledge_weight_measured.png").is_file()
    assert (
        tmp_path / "report" / "weights_classification_weight_measured.png"
    ).is_file()


def test_experiment_group_does_not_override_unrelated_comparison_settings(tmp_path):
    from evaluate import generate_report

    rows = [
        metric(
            variant=variant,
            comparison={"temperature": temperature},
            experiment={
                "suite": "knowledge",
                "axis": "dimension",
                "value": variant,
                "group": "fixture",
            },
        )
        for variant, temperature in (("full", 0.07), ("no_visual", 0.5))
    ]
    result = generate_report(rows, tmp_path / "report")
    assert not (tmp_path / "report" / "figure3_measured.png").exists()
    assert all(not row["matched"] for row in result["comparison_coverage"])


def test_timing_aggregates_only_actual_observations_with_the_same_scope():
    from evaluate import summarize

    rows = [
        metric(
            seed=seed,
            timing={
                "train_seconds_per_epoch": value,
                "scope": "epoch",
                "device": "cpu",
                "seconds": value,
            },
        )
        for seed, value in ((1, 3.0), (2, 5.0))
    ]
    result = summarize(rows)["timing"]
    assert len(result) == 1
    assert result[0]["train_seconds_per_epoch"] == 4
    assert result[0]["train_seconds_per_epoch_std"] == 1


def test_full_eleven_dataset_coverage_is_checked_per_seed():
    from evaluate import DATASETS, summarize

    rows = [
        metric(dataset, seed=seed, value=float(index * 4 + seed))
        for index, dataset in enumerate(DATASETS)
        for seed in (1, 2)
    ]
    complete = summarize(rows)["cross_dataset_accuracy"][0]
    assert complete["complete_11_datasets"]
    assert complete["mean"] == 21.5 and complete["std"] == 0.5
    incomplete = summarize(rows[1:])["cross_dataset_accuracy"][0]
    assert not incomplete["complete_11_datasets"]
    assert incomplete["seeds"] == [2] and incomplete["excluded_seeds"] == [1]
    assert incomplete["mean"] == 22


def test_cross_dataset_base_new_preserves_mean_harmonic_and_harmonic_of_means():
    from evaluate import summarize

    rows = [
        metric(dataset, mode="base_new", split=split, value=value)
        for dataset, base, new in (("imagenet", 90.0, 50.0), ("caltech101", 60.0, 80.0))
        for split, value in (("test_base", base), ("test_new", new))
    ]
    aggregate = summarize(rows)["cross_dataset_base_new"][0]
    assert aggregate["base"] == 75 and aggregate["new"] == 65
    assert aggregate["mean_harmonic"] == pytest.approx(
        (2 * 90 * 50 / 140 + 2 * 60 * 80 / 140) / 2
    )
    assert aggregate["harmonic_of_means"] == pytest.approx(2 * 75 * 65 / 140)


def test_base_new_figures_include_harmonic_values_derived_from_paired_seeds(tmp_path):
    import csv

    from evaluate import generate_report

    rows = [
        metric(
            mode="base_new",
            split=split,
            value=value,
            variant=variant,
            experiment={
                "suite": "knowledge",
                "axis": "dimension",
                "value": variant,
                "group": "fixture",
            },
        )
        for variant in ("full", "no_visual")
        for split, value in (("test_base", 90.0), ("test_new", 50.0))
    ]
    generate_report(rows, tmp_path / "report")
    with (tmp_path / "report" / "figure3_measured.csv").open() as handle:
        points = list(csv.DictReader(handle))
    harmonics = [point for point in points if point["split"] == "harmonic"]
    assert len(harmonics) == 2
    assert all(
        float(point["mean"]) == pytest.approx(2 * 90 * 50 / 140) for point in harmonics
    )
