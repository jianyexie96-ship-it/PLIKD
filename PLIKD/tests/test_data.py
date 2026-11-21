from PIL import Image
import pytest


def records(tmp_path, classes=5, samples=6):
    from data import Record

    result = []
    for label in range(classes):
        for split in ("train", "val", "test"):
            for index in range(samples):
                path = tmp_path / f"{label}_{split}_{index}.png"
                Image.new("RGB", (4, 4), (label * 20, index, 0)).save(path)
                result.append(
                    Record(
                        f"{label}-{split}-{index}",
                        str(path),
                        label,
                        f"class {label}",
                        split,
                    )
                )
    return result


def test_base_new_protocol_preserves_global_class_ids_and_prevents_new_training(
    tmp_path,
):
    from data import prepare_protocol, load_manifest, save_manifest

    path = tmp_path / "input.jsonl"
    save_manifest(records(tmp_path), path)
    output = tmp_path / "run"
    protocol = prepare_protocol(
        output, mode="base_new", shots=2, seed=1, manifest=path, dataset="fixture"
    )
    train = load_manifest(output / "train.jsonl")
    new = load_manifest(output / "test_new.jsonl")
    assert {row.label for row in train} == {0, 1, 2}
    assert {row.label for row in new} == {3, 4}
    assert len(train) == 6
    assert len(load_manifest(output / "val.jsonl")) == 6
    assert protocol["base_classes"] == [0, 1, 2]
    assert protocol["new_classes"] == [3, 4]
    assert protocol["evaluation_classes"]["test_new"] == [3, 4]


def test_fewshot_protocol_uses_all_classes_and_frozen_manifests(tmp_path):
    from data import prepare_protocol, load_prepared, save_manifest

    path = tmp_path / "input.jsonl"
    save_manifest(records(tmp_path), path)
    output = tmp_path / "run"
    prepare_protocol(
        output, mode="fewshot", shots=1, seed=2, manifest=path, dataset="fixture"
    )
    assert len(load_prepared(output, "train")) == 5
    assert len(load_prepared(output, "test")) == 30
    (output / "test.jsonl").write_text("")
    with pytest.raises(ValueError, match="modified|hash|changed"):
        load_prepared(output, "test")


def test_domain_import_remaps_target_classes_by_source_names(tmp_path):
    from data import Record, prepare_protocol, save_manifest, load_prepared

    path = tmp_path / "input.jsonl"
    save_manifest(records(tmp_path, classes=3), path)
    image = tmp_path / "variant.png"
    Image.new("RGB", (4, 4), "blue").save(image)
    target = tmp_path / "target.jsonl"
    save_manifest([Record("variant-1", str(image), 0, "class 2", "test")], target)
    output = tmp_path / "run"
    metadata = prepare_protocol(
        output,
        mode="domain",
        shots=1,
        seed=1,
        manifest=path,
        dataset="fixture",
        target_manifests={"imagenet_a": str(target)},
    )
    imported = load_prepared(output, "imagenet_a")
    assert imported[0].label == 2
    assert metadata["evaluation_classes"]["imagenet_a"] == [2]
    assert metadata["mode"] == "domain"


def test_prepare_requires_complete_train_and_test_class_identity(tmp_path):
    from data import prepare_protocol, save_manifest

    values = records(tmp_path, classes=2)
    values = [r for r in values if not (r.label == 1 and r.split == "test")]
    path = tmp_path / "bad.jsonl"
    save_manifest(values, path)
    with pytest.raises(ValueError, match="class|classes"):
        prepare_protocol(
            tmp_path / "run",
            mode="base_new",
            shots=1,
            seed=1,
            manifest=path,
            dataset="fixture",
        )


@pytest.fixture
def duplicate_imagenet(tmp_path):
    pairs = [
        ("n03773504", "missile"),
        ("n04008634", "missile"),
        ("n04355933", "sunglasses"),
        ("n04356056", "sunglasses"),
    ]
    directory = tmp_path / "imagenet"
    for split in ("train", "val"):
        for synset, name in pairs:
            folder = directory / "images" / split / synset
            folder.mkdir(parents=True)
            Image.new("RGB", (4, 4), "red").save(folder / "sample.png")
    ordered = [pairs[index] for index in (3, 1, 0, 2)]
    (directory / "classnames.txt").write_text(
        "".join(f"{synset} {name}\n" for synset, name in ordered)
    )
    return tmp_path, pairs, ordered


def test_imagenet_preserves_duplicate_names_as_distinct_global_labels(
    duplicate_imagenet,
):
    from data import class_names, prepare_dataset, load_manifest

    root, pairs, ordered = duplicate_imagenet
    manifest = root / "source.jsonl"
    values = prepare_dataset(root, "imagenet", manifest)
    assert class_names(values) == [name for synset, name in pairs]
    assert class_names(load_manifest(manifest)) == [name for synset, name in pairs]
    assert {r.label for r in values if r.classname == "missile"} == {0, 1}
    assert {r.label for r in values if r.classname == "sunglasses"} == {2, 3}


@pytest.mark.parametrize(
    "variant", ["imagenet_v2", "imagenet_sketch", "imagenet_a", "imagenet_r"]
)
def test_imagenet_variants_map_synsets_without_merging_duplicate_names(
    duplicate_imagenet, variant
):
    from pathlib import Path

    from data import _imagenet_variant

    root, pairs, ordered = duplicate_imagenet
    directories = {
        "imagenet_v2": root / "imagenetv2" / "imagenetv2-matched-frequency-format-val",
        "imagenet_sketch": root / "imagenet-sketch" / "images",
        "imagenet_a": root / "imagenet-adversarial" / "imagenet-a",
        "imagenet_r": root / "imagenet-rendition" / "imagenet-r",
    }
    labels = {synset: index for index, (synset, name) in enumerate(pairs)}
    expected = {}
    for index, (synset, name) in enumerate(
        ordered if variant == "imagenet_v2" else pairs
    ):
        folder = directories[variant] / (
            str(index) if variant == "imagenet_v2" else synset
        )
        folder.mkdir(parents=True)
        Image.new("RGB", (4, 4), "blue").save(folder / "sample.png")
        expected[folder.name] = labels[synset]
    values = _imagenet_variant(root, variant, [name for synset, name in pairs])
    assert {Path(r.image).parent.name: r.label for r in values} == expected
    assert {r.label for r in values} == {0, 1, 2, 3}
    with pytest.raises(ValueError, match="source|mapping|class"):
        _imagenet_variant(
            root, variant, ["missile", "sunglasses", "missile", "sunglasses"]
        )


def test_custom_domain_duplicate_names_require_matching_global_label(tmp_path):
    from dataclasses import replace
    from data import Record, prepare_protocol, save_manifest, load_prepared

    names = ["other", "missile", "missile"]
    source = [
        replace(row, classname=names[row.label])
        for row in records(tmp_path, classes=3, samples=1)
    ]
    manifest = tmp_path / "source.jsonl"
    save_manifest(source, manifest)
    image = tmp_path / "target.png"
    Image.new("RGB", (4, 4), "blue").save(image)
    target = tmp_path / "target.jsonl"
    save_manifest([Record("target", str(image), 1, "missile", "test")], target)
    output = tmp_path / "run"
    prepare_protocol(
        output,
        mode="domain",
        shots=1,
        manifest=manifest,
        dataset="fixture",
        target_manifests={"variant": target},
    )
    assert load_prepared(output, "variant")[0].label == 1
    save_manifest([Record("target", str(image), 0, "missile", "test")], target)
    with pytest.raises(ValueError, match="ambiguous|Ambiguous|global label"):
        prepare_protocol(
            tmp_path / "ambiguous",
            mode="domain",
            shots=1,
            manifest=manifest,
            dataset="fixture",
            target_manifests={"variant": target},
        )
