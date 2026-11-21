import argparse
import csv
import glob
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from numbers import Integral, Real
from pathlib import Path

DATASETS = (
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
)
DOMAINS = ("imagenet", "imagenet_v2", "imagenet_sketch", "imagenet_a", "imagenet_r")
ALIASES = {
    "oxfordpets": "oxford_pets",
    "oxfordpet": "oxford_pets",
    "stanfordcars": "stanford_cars",
    "cars": "stanford_cars",
    "fgvcaircraft": "fgvc_aircraft",
    "aircraft": "fgvc_aircraft",
    "imagenetv2": "imagenet_v2",
    "imagenetsketch": "imagenet_sketch",
    "imageneta": "imagenet_a",
    "imagenetr": "imagenet_r",
}


def _canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _dataset(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("dataset must be nonempty text")
    name = value.strip().lower().replace("-", "_")
    return ALIASES.get(name.replace("_", ""), name)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _percentage(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or not 0 <= value <= 100
    ):
        raise ValueError("accuracy must be a finite percentage in [0,100]")
    return float(value)


def accuracy(truth, preds):
    truth, preds = list(truth), list(preds)
    if not truth:
        raise ValueError("Accuracy requires nonempty ground truth")
    for rows in (truth, preds):
        if any(
            not isinstance(row, dict)
            or not isinstance(row.get("id"), str)
            or not row["id"].strip()
            for row in rows
        ):
            raise ValueError("Prediction coverage requires nonempty string ids")
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError("Prediction coverage contains duplicate ids")
    mapping = {row["id"]: row for row in preds}
    if set(mapping) != {row["id"] for row in truth}:
        raise ValueError("Prediction coverage does not exactly match ground truth ids")
    labels = {_integer(row["label"], "global label") for row in truth}
    counts = defaultdict(lambda: {"total": 0, "correct": 0})
    correct = invalid = 0
    for row in truth:
        label = int(row["label"])
        predicted = mapping[row["id"]].get("prediction")
        valid = (
            isinstance(predicted, Integral)
            and not isinstance(predicted, bool)
            and int(predicted) in labels
        )
        hit = valid and int(predicted) == label
        correct += int(hit)
        invalid += int(not valid)
        counts[label]["total"] += 1
        counts[label]["correct"] += int(hit)
    per_class = [
        {
            "label": label,
            **values,
            "accuracy": 100 * values["correct"] / values["total"],
        }
        for label, values in sorted(counts.items())
    ]
    return {
        "accuracy": 100 * correct / len(truth),
        "total": len(truth),
        "correct": correct,
        "invalid": invalid,
        "per_class": per_class,
    }


def _row(value):
    if not isinstance(value, dict):
        raise TypeError("Each metric row must be an object")
    required = {"dataset", "mode", "split", "seed", "shots", "accuracy", "total"}
    if not required.issubset(value):
        raise ValueError(f"Metric row is missing {sorted(required - value.keys())}")
    row = dict(value)
    row["dataset"] = _dataset(row["dataset"])
    if row["mode"] not in {"fewshot", "base_new", "domain"}:
        raise ValueError("Unknown evaluation mode")
    if not isinstance(row["split"], str) or not row["split"]:
        raise ValueError("split must be nonempty text")
    row["seed"] = _integer(row["seed"], "seed")
    row["shots"] = _integer(row["shots"], "shots", 1)
    row["total"] = _integer(row["total"], "total", 1)
    row["accuracy"] = _percentage(row["accuracy"])
    row.setdefault("variant", "full")
    row.setdefault("comparison", {})
    if (
        not isinstance(row["variant"], str)
        or not row["variant"]
        or not isinstance(row["comparison"], dict)
    ):
        raise ValueError("variant must be text and comparison must be a JSON object")
    if row.get("experiment") is not None:
        experiment = row["experiment"]
        if not isinstance(experiment, dict) or not {
            "suite",
            "axis",
            "value",
            "group",
        }.issubset(experiment):
            raise ValueError("experiment requires suite, axis, value and group")
    if row.get("measured") is False or row.get("origin") == "paper_reference":
        raise ValueError("Unsupported metric source")
    _canonical(row)
    return row


def _meta(row, dataset=True, split=True):
    result = {key: row[key] for key in ("mode", "shots", "variant", "comparison")}
    if dataset:
        result["dataset"] = row["dataset"]
        if "data_identity" in row:
            result["data_identity"] = row["data_identity"]
    if split:
        result["split"] = row["split"]
    if row.get("experiment") is not None:
        result["experiment"] = row["experiment"]
    return result


def _stats(values):
    values = list(values)
    return {
        "mean": statistics.fmean(values) if values else None,
        "std": statistics.pstdev(values) if values else None,
    }


def _harmonic(base, new):
    return 2 * base * new / (base + new) if base + new else 0.0


def _compatible(first, second):
    for field in (
        "protocol",
        "checkpoint_sha256",
        "checkpoint_hash",
        "adapter_sha256",
        "training_provenance_sha256",
        "checkpoint_epoch",
        "checkpoint_file",
    ):
        if (field in first or field in second) and _canonical(
            first.get(field)
        ) != _canonical(second.get(field)):
            return False
    return True


def _evaluation_identity(row):
    identity = {
        key: row[key]
        for key in (
            "data_identity",
            "candidate_ids",
            "manifest_sha256",
            "paired_evaluation_identity",
        )
        if key in row
    }
    protocol = row.get("protocol")
    if isinstance(protocol, dict):
        selected = {
            key: protocol[key]
            for key in (
                "dataset",
                "source_sha256",
                "class_names",
                "base_classes",
                "new_classes",
                "evaluation_classes",
            )
            if key in protocol
        }
        split_hashes = protocol.get("split_hashes")
        if isinstance(split_hashes, dict):
            selected["evaluation_split_hashes"] = {
                key: value
                for key, value in split_hashes.items()
                if key not in {"train", "val"}
            }
        if selected:
            identity["evaluation_protocol"] = selected
    return _canonical(identity)


def _require_same_evaluation_identity(samples):
    if len({_evaluation_identity(row) for row in samples}) > 1:
        raise ValueError(
            "Dataset evaluation identity or data release differs across seeds"
        )


def _cross_dataset(rows, paired=False):
    groups = defaultdict(list)
    for row in rows:
        if row["mode"] != "domain":
            groups[_canonical(_meta(row, dataset=False, split=not paired))].append(row)
    output = []
    for key, items in sorted(groups.items()):
        samples = defaultdict(dict)
        for row in items:
            if row["seed"] in samples[row["dataset"]]:
                raise ValueError("Duplicate dataset/seed in cross-dataset aggregation")
            samples[row["dataset"]][row["seed"]] = row
        if len(samples) < 2:
            continue
        for values in samples.values():
            _require_same_evaluation_identity(values.values())
        seed_sets = [set(values) for values in samples.values()]
        common = sorted(set.intersection(*seed_sets))
        union = set.union(*seed_sets)
        same = all(seeds == seed_sets[0] for seeds in seed_sets)
        per_seed = []
        for seed in common:
            values = [samples[dataset][seed] for dataset in sorted(samples)]
            if paired:
                base, new = (
                    statistics.fmean(value["base"] for value in values),
                    statistics.fmean(value["new"] for value in values),
                )
                per_seed.append(
                    {
                        "seed": seed,
                        "base": base,
                        "new": new,
                        "mean_harmonic": statistics.fmean(
                            value["harmonic"] for value in values
                        ),
                        "harmonic_of_means": _harmonic(base, new),
                    }
                )
            else:
                per_seed.append(
                    {
                        "seed": seed,
                        "accuracy": statistics.fmean(
                            value["accuracy"] for value in values
                        ),
                    }
                )
        result = {
            **json.loads(key),
            "datasets": sorted(samples),
            "dataset_count": len(samples),
            "seeds": common,
            "n_seeds": len(common),
            "same_seed_coverage": same,
            "excluded_seeds": sorted(union - set(common)),
            "complete_11_datasets": set(samples) == set(DATASETS)
            and same
            and bool(common),
            "missing_datasets": sorted(set(DATASETS) - set(samples)),
            "per_seed": per_seed,
        }
        if paired:
            base, new = (
                _stats(value["base"] for value in per_seed),
                _stats(value["new"] for value in per_seed),
            )
            harmonic = _stats(value["mean_harmonic"] for value in per_seed)
            result.update(
                {
                    "base": base["mean"],
                    "base_std": base["std"],
                    "new": new["mean"],
                    "new_std": new["std"],
                    "mean_harmonic": harmonic["mean"],
                    "harmonic_std": harmonic["std"],
                    "harmonic_of_means": _harmonic(base["mean"], new["mean"])
                    if per_seed
                    else None,
                }
            )
            result["mean_H"] = result["mean_harmonic"]
            result["H_of_means"] = result["harmonic_of_means"]
        else:
            result.update(_stats(value["accuracy"] for value in per_seed))
        output.append(result)
    return output


def _domain_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["mode"] == "domain":
            groups[_canonical(_meta(row, dataset=False, split=False))].append(row)
    results = []
    for key, items in sorted(groups.items()):
        domains = defaultdict(dict)
        for row in items:
            name = _dataset(
                row.get(
                    "domain", row["dataset"] if row["split"] == "test" else row["split"]
                )
            )
            if row["seed"] in domains[name]:
                raise ValueError("Duplicate domain/seed result")
            domains[name][row["seed"]] = row
        for values in domains.values():
            _require_same_evaluation_identity(values.values())
        sets = [set(domains[name]) for name in DOMAINS if name in domains]
        common = sorted(set.intersection(*sets)) if len(sets) == len(DOMAINS) else []
        compatible = [
            seed
            for seed in common
            if all(
                _compatible(domains["imagenet"][seed], domains[name][seed])
                for name in DOMAINS[1:]
            )
        ]
        stats = {
            name: {
                **_stats(value["accuracy"] for value in samples.values()),
                "seeds": sorted(samples),
            }
            for name, samples in sorted(domains.items())
        }
        paired = [
            {
                "seed": seed,
                **{name: domains[name][seed]["accuracy"] for name in DOMAINS},
                "target_mean": statistics.fmean(
                    domains[name][seed]["accuracy"] for name in DOMAINS[1:]
                ),
            }
            for seed in compatible
        ]
        complete = (
            len(sets) == len(DOMAINS)
            and all(values == sets[0] for values in sets)
            and len(compatible) == len(common)
            and bool(common)
        )
        results.append(
            {
                **json.loads(key),
                "source": stats.get("imagenet"),
                "targets": {
                    name: value for name, value in stats.items() if name != "imagenet"
                },
                "complete_source_and_four_targets": complete,
                "missing_domains": sorted(set(DOMAINS) - set(domains)),
                "seeds": compatible,
                "per_seed": paired,
                "target_mean": _stats(value["target_mean"] for value in paired)["mean"],
                "target_std": _stats(value["target_mean"] for value in paired)["std"],
            }
        )
    return results


def _timing_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        timing = row.get("timing")
        if timing is None:
            continue
        if not isinstance(timing, dict):
            raise TypeError("timing must be an object with measured values")
        values = {}
        for field in ("train_seconds_per_epoch", "inference_images_per_second"):
            if field in timing:
                value = timing[field]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError(
                        f"Measured timing {field} must be finite and positive"
                    )
                values[field] = float(value)
        if values:
            protocol = {
                key: value
                for key, value in timing.items()
                if key
                not in {
                    "train_seconds_per_epoch",
                    "inference_images_per_second",
                    "seconds",
                    "images",
                }
            }
            groups[_canonical({**_meta(row), "timing_protocol": protocol})].append(
                {
                    "seed": row["seed"],
                    **values,
                    **{
                        key: value
                        for key, value in timing.items()
                        if key in {"seconds", "images"}
                    },
                }
            )
    output = []
    for key, observations in sorted(groups.items()):
        result = {
            **json.loads(key),
            "seeds": sorted(value["seed"] for value in observations),
            "n_seeds": len(observations),
            "per_seed": sorted(observations, key=lambda value: value["seed"]),
            "field_seeds": {},
        }
        for field in ("train_seconds_per_epoch", "inference_images_per_second"):
            samples = [value for value in observations if field in value]
            if samples:
                stats = _stats(value[field] for value in samples)
                result[field] = stats["mean"]
                result[field + "_std"] = stats["std"]
                result["field_seeds"][field] = sorted(
                    value["seed"] for value in samples
                )
        output.append(result)
    return output


def summarize(rows):
    rows = [_row(row) for row in rows]
    groups = defaultdict(dict)
    pair_groups = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        key = _canonical(_meta(row))
        if row["seed"] in groups[key]:
            raise ValueError(
                "Duplicate seed metric for the same dataset, split and comparison"
            )
        groups[key][row["seed"]] = row
        if row["mode"] == "base_new" and row["split"] in {"test_base", "test_new"}:
            pair_groups[_canonical(_meta(row, split=False))][row["seed"]][
                row["split"]
            ] = row
    aggregated = []
    for key, samples in sorted(groups.items()):
        stats = _stats(row["accuracy"] for row in samples.values())
        aggregated.append(
            {
                **json.loads(key),
                **stats,
                "accuracy": stats["mean"],
                "accuracy_std": stats["std"],
                "seeds": sorted(samples),
                "n_seeds": len(samples),
                "per_seed": [
                    {
                        "seed": seed,
                        "accuracy": samples[seed]["accuracy"],
                        "total": samples[seed]["total"],
                    }
                    for seed in sorted(samples)
                ],
            }
        )
    paired, incomplete, flat_pairs = [], [], []
    for key, samples in sorted(pair_groups.items()):
        pairs = []
        for seed, values in sorted(samples.items()):
            if set(values) != {"test_base", "test_new"}:
                incomplete.append(
                    {
                        **json.loads(key),
                        "seed": seed,
                        "reason": "missing base or new split",
                        "present_splits": sorted(values),
                    }
                )
            elif not _compatible(values["test_base"], values["test_new"]):
                incomplete.append(
                    {
                        **json.loads(key),
                        "seed": seed,
                        "reason": "protocol mismatch",
                        "present_splits": sorted(values),
                    }
                )
            else:
                base, new = (
                    values["test_base"]["accuracy"],
                    values["test_new"]["accuracy"],
                )
                pair = {
                    "seed": seed,
                    "base": base,
                    "new": new,
                    "harmonic": _harmonic(base, new),
                }
                pairs.append(pair)
                flat_pairs.append(
                    {
                        **json.loads(key),
                        **pair,
                        "paired_evaluation_identity": {
                            split: json.loads(_evaluation_identity(row))
                            for split, row in values.items()
                        },
                    }
                )
        if pairs:
            base, new = (
                _stats(pair["base"] for pair in pairs),
                _stats(pair["new"] for pair in pairs),
            )
            harmonic = _stats(pair["harmonic"] for pair in pairs)
            paired.append(
                {
                    **json.loads(key),
                    "base": base["mean"],
                    "base_std": base["std"],
                    "new": new["mean"],
                    "new_std": new["std"],
                    "mean_harmonic": harmonic["mean"],
                    "harmonic_std": harmonic["std"],
                    "harmonic_of_means": _harmonic(base["mean"], new["mean"]),
                    "mean_H": harmonic["mean"],
                    "H_of_means": _harmonic(base["mean"], new["mean"]),
                    "seeds": [pair["seed"] for pair in pairs],
                    "n_seeds": len(pairs),
                    "complete_seed_pairs": len(pairs) == len(samples),
                    "per_seed": pairs,
                }
            )
    return {
        "accuracy": aggregated,
        "base_new": paired,
        "incomplete_pairs": incomplete,
        "cross_dataset_accuracy": _cross_dataset(rows),
        "cross_dataset_base_new": _cross_dataset(flat_pairs, paired=True),
        "domain": _domain_summary(rows),
        "timing": _timing_summary(rows),
        "comparison_coverage": [],
        "std_definition": "population standard deviation (ddof=0)",
        "accuracy_unit": "percent",
    }


def paper_reference():
    triples = (
        (83.15, 77.28, 80.11),
        (98.96, 97.95, 98.45),
        (96.33, 98.50, 97.40),
        (83.15, 85.32, 84.22),
        (98.92, 84.38, 91.07),
        (93.40, 95.76, 94.57),
        (50.07, 43.39, 46.49),
        (84.82, 82.93, 83.86),
        (87.19, 73.83, 79.96),
        (96.94, 85.82, 91.04),
        (91.33, 84.15, 87.59),
    )
    fewshot = (
        67.39,
        92.58,
        87.62,
        76.52,
        95.19,
        79.02,
        35.76,
        74.19,
        69.93,
        85.86,
        78.39,
    )
    lambda12 = (
        (0.9, 0.1, 82.05, 75.83, 78.82),
        (0.8, 0.2, 82.20, 76.34, 79.16),
        (0.7, 0.3, 83.15, 77.28, 80.11),
        (0.6, 0.4, 82.63, 76.50, 79.45),
        (0.5, 0.5, 82.12, 76.30, 79.10),
        (0.4, 0.6, 81.90, 76.05, 78.87),
        (0.3, 0.7, 81.77, 75.63, 78.58),
        (0.2, 0.8, 81.52, 75.39, 78.34),
        (0.1, 0.9, 81.15, 74.89, 77.89),
    )
    lambda3 = (
        (1.0, 82.70, 77.06, 79.78),
        (0.9, 82.95, 77.13, 79.93),
        (0.8, 83.15, 77.28, 80.11),
        (0.7, 82.86, 77.08, 79.87),
        (0.6, 82.63, 76.92, 79.67),
        (0.5, 82.59, 76.83, 79.61),
        (0.4, 82.50, 76.35, 79.31),
        (0.3, 82.46, 76.30, 79.26),
        (0.2, 82.31, 76.23, 79.15),
    )
    return {
        "origin": "paper_reference",
        "measured": False,
        "paper": "PLIKD, WWW 2026",
        "doi": "10.1145/3774904.3792293",
        "source_pages": {
            "table1": 7,
            "table2": 7,
            "table5": 12,
            "table6": 12,
            "table7": 12,
        },
        "table1_base_new": {
            "datasets": {
                name: {"base": values[0], "new": values[1], "harmonic": values[2]}
                for name, values in zip(DATASETS, triples)
            },
            "reported_average": {"base": 87.66, "new": 82.66, "harmonic": 85.09},
        },
        "table2_domain": dict(zip(DOMAINS, (74.39, 69.20, 52.93, 53.85, 80.90))),
        "table5_fewshot": {**dict(zip(DATASETS, fewshot)), "reported_average": 76.59},
        "table6_lambda12": [
            {"lambda1": a, "lambda2": b, "base": base, "new": new, "harmonic": harmonic}
            for a, b, base, new, harmonic in lambda12
        ],
        "table7_lambda3": [
            {"lambda3": a, "base": base, "new": new, "harmonic": harmonic}
            for a, base, new, harmonic in lambda3
        ],
        "figures": {
            str(
                number
            ): "No numeric plot coordinates transcribed; measured plots require actual input metrics"
            for number in (3, 4, 5, 6)
        },
    }


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path, rows):
    rows = list(rows)
    fields = sorted({key for row in rows for key in row}) or ["no_measured_rows"]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: _canonical(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def _read_inputs(inputs):
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    rows, files = [], {}
    for value in inputs:
        if isinstance(value, dict):
            rows.append(value)
            continue
        names = sorted(glob.glob(str(value), recursive=True))
        if not names:
            raise FileNotFoundError(f"No metric files matched {value}")
        for name in names:
            path = Path(name).resolve()
            if str(path) in files:
                continue
            content = path.read_bytes()
            files[str(path)] = hashlib.sha256(content).hexdigest()
            parsed = json.loads(content)
            if isinstance(parsed, dict) and "rows" in parsed:
                parsed = parsed["rows"]
            rows.extend(parsed if isinstance(parsed, list) else [parsed])
    return [_row(row) for row in rows], files


def _figure(experiment):
    suite = str(experiment["suite"]).lower()
    axis = str(experiment["axis"]).lower().replace("_", "")
    if suite in {"knowledge", "instance_knowledge", "figure3"}:
        return 3
    if suite in {"dual", "dual_teacher", "distillation", "figure4"}:
        return 4
    if suite in {"transport", "ot", "figure5"}:
        return 5
    if suite in {"epsilon", "sparsity", "figure6"} or axis == "epsilon":
        return 6
    if suite == "weights" and axis in {"knowledgeweight", "classificationweight"}:
        return "weights_" + experiment["axis"]
    return None


def _controlled_comparison(row):
    comparison = json.loads(_canonical(row["comparison"]))
    experiment = row["experiment"]
    suite, axis = experiment["suite"], experiment["axis"]
    paths = []
    if suite in {"knowledge", "instance_knowledge", "figure3"}:
        paths = [("knowledge", "dimensions")]
    elif suite in {"dual", "dual_teacher", "distillation", "figure4"}:
        paths = [("loss", "visual_weight")]
    elif suite in {"ot", "transport", "figure5"}:
        paths = [("transport", "solver")]
    elif axis == "epsilon":
        paths = [("transport", "epsilon")]
    elif suite == "weights" and axis == "knowledge_weight":
        paths = [("loss", "knowledge_weight"), ("loss", "visual_weight")]
    elif suite == "weights" and axis == "classification_weight":
        paths = [("loss", "classification_weight")]
    for parent, child in paths:
        if isinstance(comparison.get(parent), dict):
            comparison[parent].pop(child, None)
    if axis in {"variant", "dimension", "solver"}:
        comparison.pop("variant", None)
    return comparison


def _plot_label(value, axis):
    if axis == "dimension":
        return {
            "all": "All knowledge",
            "full": "All knowledge",
            "visual": "Without visual",
            "functional": "Without functional",
            "contextual": "Without contextual",
            "no_visual": "Without visual",
            "no_functional": "Without functional",
            "no_contextual": "Without contextual",
        }.get(str(value), str(value))
    if axis == "solver":
        return {"sparse": "SSOT", "entropic": "Entropic OT"}.get(str(value), str(value))
    return str(value)


def _plot_experiments(rows, output):
    paired = defaultdict(dict)
    for row in rows:
        if (
            row["mode"] == "base_new"
            and row["split"] in {"test_base", "test_new"}
            and row.get("experiment")
            and _figure(row["experiment"]) not in {None, 4}
        ):
            paired[(_canonical(_meta(row, split=False)), row["seed"])][row["split"]] = (
                row
            )
    derived = []
    for values in paired.values():
        if set(values) == {"test_base", "test_new"} and _compatible(
            values["test_base"], values["test_new"]
        ):
            derived.append(
                {
                    **values["test_base"],
                    "split": "harmonic",
                    "accuracy": _harmonic(
                        values["test_base"]["accuracy"], values["test_new"]["accuracy"]
                    ),
                }
            )
    cohorts = defaultdict(list)
    for row in rows + derived:
        experiment = row.get("experiment")
        if not experiment or _figure(experiment) is None:
            continue
        key = {
            "figure": _figure(experiment),
            "suite": experiment["suite"],
            "group": experiment["group"],
            "axis": experiment["axis"],
            "mode": row["mode"],
            "split": row["split"],
            "comparison": _controlled_comparison(row),
        }
        if experiment["axis"] != "shots":
            key["shots"] = row["shots"]
        cohorts[_canonical(key)].append(row)
    coverage, panels, points = [], defaultdict(list), defaultdict(list)
    for key, items in sorted(cohorts.items()):
        metadata = json.loads(key)
        cells, values, duplicates = defaultdict(dict), {}, False
        identities = defaultdict(set)
        for row in items:
            experiment = row["experiment"]
            series = (
                "PLIKD"
                if experiment["axis"] in {"variant", "dimension", "solver"}
                else row["variant"]
            )
            cell = (series, _canonical(experiment["value"]))
            unit = (row["dataset"], row["seed"])
            identities[row["dataset"]].add(_evaluation_identity(row))
            duplicates |= unit in cells[cell]
            cells[cell][unit] = row["accuracy"]
            values[cell] = experiment["value"]
        unit_sets = [set(cell) for cell in cells.values()]
        series_values = defaultdict(set)
        for series, value in cells:
            series_values[series].add(value)
        matched_axes = all(
            value == next(iter(series_values.values()))
            for value in series_values.values()
        )
        matched = (
            len(cells) >= 2
            and not duplicates
            and matched_axes
            and all(units == unit_sets[0] for units in unit_sets)
        )
        same_identity = all(len(values) == 1 for values in identities.values())
        matched = matched and same_identity
        coverage.append(
            {
                **metadata,
                "matched": matched,
                "reason": None
                if matched
                else (
                    "Evaluation identity differs across comparison cells or seeds"
                    if not same_identity
                    else "At least two unambiguous cells with identical dataset/seed coverage are required"
                ),
                "cells": [
                    {
                        "series": cell[0],
                        "value": values[cell],
                        "units": [list(unit) for unit in sorted(samples)],
                    }
                    for cell, samples in sorted(cells.items())
                ],
            }
        )
        if not matched:
            continue
        units = unit_sets[0]
        datasets = sorted({dataset for dataset, seed in units})
        seed_sets = [
            {seed for dataset, seed in units if dataset == name} for name in datasets
        ]
        scopes = ["imagenet"] if "imagenet" in datasets else datasets
        if set(datasets) == set(DATASETS) and all(
            seeds == seed_sets[0] for seeds in seed_sets
        ):
            scopes.insert(0, "AVG11")
        for scope in scopes:
            selected = set(DATASETS) if scope == "AVG11" else {scope}
            seeds = sorted({seed for dataset, seed in units if dataset in selected})
            chart = []
            for cell, samples in sorted(cells.items()):
                observations = [
                    statistics.fmean(samples[(dataset, seed)] for dataset in selected)
                    for seed in seeds
                ]
                point = {
                    **metadata,
                    "dataset": scope,
                    "series": cell[0],
                    "value": values[cell],
                    **_stats(observations),
                    "seeds": seeds,
                    "metric": "paired-seed harmonic mean"
                    if metadata["split"] == "harmonic"
                    else "accuracy",
                }
                chart.append(point)
                points[metadata["figure"]].append(point)
            panels[metadata["figure"]].append((f"{scope}: {metadata['split']}", chart))
    if not panels:
        return coverage
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for figure_number, contents in sorted(
        panels.items(), key=lambda item: str(item[0])
    ):
        columns = min(2, len(contents))
        figure, axes = plt.subplots(
            math.ceil(len(contents) / columns),
            columns,
            figsize=(6 * columns, 4 * math.ceil(len(contents) / columns)),
            squeeze=False,
        )
        for axis, (title, chart) in zip(axes.flat, contents):
            series_names = sorted({point["series"] for point in chart})
            numeric = all(
                isinstance(point["value"], Real)
                and not isinstance(point["value"], bool)
                for point in chart
            )
            for index, series in enumerate(series_names):
                values = sorted(
                    (point for point in chart if point["series"] == series),
                    key=lambda point: (
                        point["value"] if numeric else _canonical(point["value"])
                    ),
                )
                x = (
                    [point["value"] for point in values]
                    if numeric
                    else list(range(len(values)))
                )
                if numeric:
                    label = (
                        {
                            "full": "Dual supervision",
                            "mllm_only": "MLLM supervision only",
                        }.get(series, series)
                        if chart[0]["suite"] in {"dual", "dual_teacher"}
                        else series
                    )
                    axis.errorbar(
                        x,
                        [point["mean"] for point in values],
                        yerr=[point["std"] for point in values],
                        marker="o",
                        capsize=3,
                        label=label,
                    )
                else:
                    width = 0.8 / len(series_names)
                    axis.bar(
                        [
                            value + (index - (len(series_names) - 1) / 2) * width
                            for value in x
                        ],
                        [point["mean"] for point in values],
                        width=width,
                        yerr=[point["std"] for point in values],
                        capsize=3,
                        label=series,
                    )
                    axis.set_xticks(
                        x,
                        [
                            _plot_label(point["value"], point["axis"])
                            for point in values
                        ],
                        rotation=20,
                    )
            axis.set_title(title)
            axis.set_xlabel(chart[0]["axis"])
            axis.set_ylabel(
                "H (%)" if chart[0]["split"] == "harmonic" else "Accuracy (%)"
            )
            axis.set_ylim(0, 100)
            if len(series_names) > 1:
                axis.legend()
        for axis in list(axes.flat)[len(contents) :]:
            axis.set_visible(False)
        prefix = (
            f"figure{figure_number}"
            if isinstance(figure_number, int)
            else figure_number
        )
        title = (
            f"Figure {figure_number}"
            if isinstance(figure_number, int)
            else figure_number.replace("_", " ").title()
        )
        figure.suptitle(f"{title}: measured results only")
        figure.tight_layout()
        figure.savefig(output / f"{prefix}_measured.png", dpi=160)
        plt.close(figure)
        _write_csv(output / f"{prefix}_measured.csv", points[figure_number])
    return coverage


def generate_report(inputs, output):
    rows, files = _read_inputs(inputs)
    result = summarize(rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for prefix in (
        "figure3",
        "figure4",
        "figure5",
        "figure6",
        "weights_knowledge_weight",
        "weights_classification_weight",
    ):
        for suffix in ("png", "csv"):
            (output / f"{prefix}_measured.{suffix}").unlink(missing_ok=True)
    result["comparison_coverage"] = _plot_experiments(rows, output)
    result["input_sha256"] = files
    result["measured_row_count"] = len(rows)
    _write_json(output / "measured_rows.json", rows)
    _write_json(output / "paper_reference.json", paper_reference())
    _write_json(output / "summary.json", result)
    for key in (
        "accuracy",
        "base_new",
        "incomplete_pairs",
        "cross_dataset_accuracy",
        "cross_dataset_base_new",
        "domain",
        "timing",
        "comparison_coverage",
    ):
        _write_csv(output / f"{key}.csv", result[key])
    _write_csv(
        output / "table1_measured.csv",
        result["base_new"]
        + [{"dataset": "AVG", **row} for row in result["cross_dataset_base_new"]],
    )
    _write_csv(output / "table2_measured.csv", result["domain"])
    _write_csv(output / "table3_measured.csv", result["timing"])
    _write_csv(
        output / "table5_measured.csv",
        [row for row in result["accuracy"] if row["mode"] == "fewshot"]
        + [
            {"dataset": "AVG", **row}
            for row in result["cross_dataset_accuracy"]
            if row["mode"] == "fewshot"
        ],
    )
    for table, suite, axis in (
        (6, "lambda12", "knowledge_weight"),
        (7, "lambda3", "classification_weight"),
    ):
        _write_csv(
            output / f"table{table}_measured.csv",
            [
                row
                for row in result["base_new"]
                if row.get("experiment", {}).get("suite") == suite
                or (
                    row.get("experiment", {}).get("suite") == "weights"
                    and row["experiment"]["axis"] == axis
                )
            ],
        )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate measured PLIKD classification results"
    )
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    generate_report(args.input, args.output)


if __name__ == "__main__":
    main()
