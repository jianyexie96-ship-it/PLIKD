from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile


DATA_SOURCE = "https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md"
DATA_MIRROR = "https://huggingface.co/zhengli97/prompt_learning_dataset/tree/main"
SPLITS = ("train", "val", "test")
DATASETS = {
    "imagenet": ("imagenet", "images", None),
    "caltech101": ("caltech-101", "101_ObjectCategories", "Caltech101"),
    "oxford_pets": ("oxford_pets", "images", "OxfordPets"),
    "stanford_cars": ("stanford_cars", "", "StanfordCars"),
    "flowers102": ("oxford_flowers", "jpg", "OxfordFlowers"),
    "food101": ("food-101", "images", "Food101"),
    "fgvc_aircraft": ("fgvc_aircraft", "images", None),
    "sun397": ("sun397", "SUN397", "SUN397"),
    "dtd": ("dtd", "images", "DescribableTextures"),
    "eurosat": ("eurosat", "2750", "EuroSAT"),
    "ucf101": ("ucf101", "UCF-101-midframes", "UCF101"),
}
ALIASES = {
    "imagenet1k": "imagenet",
    "imagenet1000": "imagenet",
    "oxfordpet": "oxford_pets",
    "oxfordpets": "oxford_pets",
    "pets": "oxford_pets",
    "stanfordcars": "stanford_cars",
    "cars": "stanford_cars",
    "oxfordflowers": "flowers102",
    "flowers": "flowers102",
    "fgvcaircraft": "fgvc_aircraft",
    "aircraft": "fgvc_aircraft",
    "describabletextures": "dtd",
}


@dataclass(frozen=True)
class Record:
    id: str
    image: str
    label: int
    classname: str
    split: str

    def __post_init__(self):
        object.__setattr__(self, "image", str(Path(self.image).expanduser().resolve()))


def _class_mapping(records):
    mapping = {}
    for item in records:
        if (
            not isinstance(item.label, int)
            or isinstance(item.label, bool)
            or item.label < 0
        ):
            raise ValueError(f"Invalid nonnegative integer label: {item.label!r}")
        if not isinstance(item.classname, str) or not item.classname.strip():
            raise ValueError(f"Empty or invalid classname for {item.id}")
        if item.label in mapping and mapping[item.label] != item.classname:
            raise ValueError(f"Conflicting classnames for label {item.label}")
        mapping[item.label] = item.classname
    return mapping


def class_names(records):
    mapping = _class_mapping(records)
    if sorted(mapping) != list(range(len(mapping))):
        raise ValueError("Class labels must be contiguous from zero")
    return [mapping[label] for label in range(len(mapping))]


def validate_manifest(records):
    records = list(records)
    _class_mapping(records)
    ids = set()
    images = {}
    for item in records:
        if not isinstance(item.id, str) or not item.id.strip():
            raise ValueError("Record id must be a nonempty string")
        if item.id in ids:
            raise ValueError(f"Duplicate record id: {item.id}")
        ids.add(item.id)
        if item.split not in SPLITS:
            raise ValueError(f"Invalid split {item.split!r} for {item.id}")
        image = Path(item.image).expanduser().resolve()
        if not image.is_file():
            raise FileNotFoundError(f"Image does not exist: {image}")
        if image in images:
            previous = images[image]
            raise ValueError(
                f"Duplicate image or cross-split overlap: {image} ({previous}, {item.split})"
            )
        images[image] = item.split
    return records


def _record_from_json(value, base):
    if not isinstance(value, dict):
        raise ValueError("Each manifest record must be a JSON object")
    missing = {"id", "image", "label", "classname", "split"} - value.keys()
    if missing:
        raise ValueError(f"Manifest record is missing fields: {sorted(missing)}")
    image = Path(value["image"]).expanduser()
    if not image.is_absolute():
        image = base / image
    return Record(
        value["id"], str(image), value["label"], value["classname"], value["split"]
    )


def load_manifest(path):
    path = Path(path).expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    try:
        values = json.loads(text)
    except json.JSONDecodeError:
        values = []
        for number, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    values.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid manifest JSON on line {number}: {error}"
                    ) from error
    if isinstance(values, dict):
        values = values["records"] if "records" in values else [values]
    if not isinstance(values, list):
        raise ValueError(
            "Manifest must be JSONL, a list, or an object containing records"
        )
    records = [_record_from_json(value, path.parent) for value in values]
    return validate_manifest(records)


def save_manifest(records, path):
    records = validate_manifest(records)
    path = Path(path).expanduser().resolve()
    if any(Path(item.image) == path for item in records):
        raise ValueError("Manifest output cannot overwrite an image")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as output:
            temporary = Path(output.name)
            for item in records:
                output.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _sample(records, shots, rng):
    records = list(records)
    if not isinstance(shots, int) or isinstance(shots, bool):
        raise ValueError("shots must be an integer")
    if shots < 1:
        return records
    groups = defaultdict(list)
    for item in records:
        groups[item.label].append(item)
    result = []
    for items in groups.values():
        result.extend(rng.sample(items, shots) if len(items) >= shots else items)
    return result


def sample_fewshot(records, shots, seed):
    return _sample(records, shots, random.Random(seed))


def split_records(records, shots=0, seed=1):
    records = list(records)
    if any(item.split not in SPLITS for item in records):
        raise ValueError("Records contain an unknown split")
    result = {
        split: [item for item in records if item.split == split] for split in SPLITS
    }
    rng = random.Random(seed)
    result["train"] = _sample(result["train"], shots, rng)
    result["val"] = _sample(result["val"], min(shots, 4), rng)
    return result


def _dataset_name(dataset):
    key = "".join(
        character for character in str(dataset).lower() if character.isalnum()
    )
    names = {
        "".join(character for character in name if character.isalnum()): name
        for name in DATASETS
    }
    names.update(ALIASES)
    if key not in names:
        raise ValueError(
            f"Unknown dataset {dataset!r}; choose from {', '.join(DATASETS)}"
        )
    return names[key]


def _required(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Required dataset file is missing: {path}. Official setup: {DATA_SOURCE}. Author mirror: {DATA_MIRROR}"
        )
    return path


def _make_record(dataset, dataset_root, image, label, classname, split):
    image = Path(os.path.abspath(image))
    relative = image.relative_to(Path(os.path.abspath(dataset_root))).as_posix()
    digest = hashlib.sha256(f"{dataset}:{relative}".encode("utf-8")).hexdigest()[:24]
    return Record(f"{dataset}-{digest}", str(image.resolve()), label, classname, split)


def _read_zhou(dataset, directory, prefix, suffix):
    path = _required(directory / f"split_zhou_{suffix}.json")
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict) or set(SPLITS) - values.keys():
        raise ValueError(f"Zhou split requires train, val and test lists: {path}")
    image_root = directory / prefix
    records = []
    for split in SPLITS:
        if not isinstance(values[split], list):
            raise ValueError(f"Split {split} must be a list: {path}")
        for entry in values[split]:
            if not isinstance(entry, (list, tuple)) or len(entry) != 3:
                raise ValueError(
                    f"Expected [relative_path, label, classname] in {path}"
                )
            relative, label, classname = entry
            if not isinstance(relative, str) or Path(relative).is_absolute():
                raise ValueError(f"Zhou image path must be relative: {relative!r}")
            image = image_root / relative
            if not image.resolve().is_relative_to(image_root.resolve()):
                raise ValueError(
                    f"Zhou image path escapes outside image directory: {relative}"
                )
            records.append(
                _make_record(dataset, directory, image, label, classname, split)
            )
    return records


def _read_imagenet(dataset, directory):
    names = {}
    for line in (
        _required(directory / "classnames.txt").read_text(encoding="utf-8").splitlines()
    ):
        if not line.strip():
            continue
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2 or fields[0] in names:
            raise ValueError("Invalid or duplicate ImageNet classname entry")
        names[fields[0]] = fields[1]
    train_root = directory / "images/train"
    test_root = directory / "images/val"
    for path in (train_root, test_root):
        if not path.is_dir():
            raise FileNotFoundError(
                f"Required ImageNet directory is missing: {path}. Official setup: {DATA_SOURCE}"
            )
    folders = sorted(
        path.name
        for path in train_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    labels = {folder: label for label, folder in enumerate(folders)}
    records = []
    for split, image_root in (("train", train_root), ("test", test_root)):
        split_folders = sorted(
            path.name
            for path in image_root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        )
        if split_folders != folders:
            raise ValueError("ImageNet train and validation synset folders must match")
        for folder in folders:
            if folder not in names:
                raise ValueError(f"Missing ImageNet classname for synset {folder}")
            for image in sorted((image_root / folder).iterdir()):
                if (
                    image.is_file()
                    and not image.name.startswith(".")
                    and image.suffix.lower()
                    in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                ):
                    records.append(
                        _make_record(
                            dataset,
                            directory,
                            image,
                            labels[folder],
                            names[folder],
                            split,
                        )
                    )
    return records


def _read_aircraft(dataset, directory):
    names = [
        line.strip()
        for line in _required(directory / "variants.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate aircraft variants")
    labels = {name: label for label, name in enumerate(names)}
    records = []
    for split in SPLITS:
        path = _required(directory / f"images_variant_{split}.txt")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            fields = line.strip().split(maxsplit=1)
            if len(fields) != 2 or fields[1] not in labels:
                raise ValueError(f"Invalid aircraft annotation in {path}: {line}")
            filename, name = fields
            if Path(filename).name != filename:
                raise ValueError(f"Invalid aircraft image id: {filename}")
            image = directory / "images" / f"{filename}.jpg"
            records.append(
                _make_record(dataset, directory, image, labels[name], name, split)
            )
    return records


def prepare_dataset(root, dataset, output):
    dataset = _dataset_name(dataset)
    folder, prefix, suffix = DATASETS[dataset]
    directory = Path(root).expanduser().resolve() / folder
    if dataset == "imagenet":
        records = _read_imagenet(dataset, directory)
    elif dataset == "fgvc_aircraft":
        records = _read_aircraft(dataset, directory)
    else:
        records = _read_zhou(dataset, directory, prefix, suffix)
    if not records:
        raise ValueError(f"Dataset {dataset} contains no records")
    validate_manifest(records)
    class_names(records)
    save_manifest(records, output)
    return records


def _file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _write_metadata(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def candidate_classes(records):
    mapping = _class_mapping(records)
    ids = sorted(mapping)
    return ids, [mapping[index] for index in ids]


def _imagenet_variant(root, name, source_names):
    root = Path(root).expanduser().resolve()
    names_file = root / "imagenet" / "classnames.txt"
    _required(names_file)
    synsets = []
    names = {}
    for line in names_file.read_text().splitlines():
        if line.strip():
            fields = line.strip().split(maxsplit=1)
            if len(fields) != 2 or fields[0] in names:
                raise ValueError("Invalid or duplicate ImageNet classname entry")
            key, value = fields
            synsets.append(key)
            names[key] = value
    source_root = root / "imagenet" / "images" / "train"
    if not source_root.is_dir():
        raise FileNotFoundError(f"Missing ImageNet source directory: {source_root}")
    source_synsets = sorted(
        folder.name
        for folder in source_root.iterdir()
        if folder.is_dir() and not folder.name.startswith(".")
    )
    if any(synset not in names for synset in source_synsets) or [
        names[synset] for synset in source_synsets
    ] != list(source_names):
        raise ValueError("ImageNet source class mapping differs from sorted synset ids")
    source_ids = {synset: index for index, synset in enumerate(source_synsets)}
    directories = {
        "imagenet_v2": root / "imagenetv2" / "imagenetv2-matched-frequency-format-val",
        "imagenet_sketch": root / "imagenet-sketch" / "images",
        "imagenet_a": root / "imagenet-adversarial" / "imagenet-a",
        "imagenet_r": root / "imagenet-rendition" / "imagenet-r",
    }
    directory = directories[name]
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Missing {name} image directory: {directory}. See {DATA_SOURCE}"
        )
    records = []
    for folder in sorted(
        p for p in directory.iterdir() if p.is_dir() and not p.name.startswith(".")
    ):
        if name == "imagenet_v2":
            if not folder.name.isdigit() or int(folder.name) >= len(synsets):
                raise ValueError(
                    f"Invalid ImageNetV2 numeric class directory: {folder}"
                )
            synset = synsets[int(folder.name)]
        else:
            synset = folder.name
        if synset not in source_ids:
            raise ValueError(
                f"Target class {synset} does not match the ImageNet source class mapping"
            )
        classname = names[synset]
        for image in sorted(folder.iterdir()):
            if image.is_file() and image.suffix.lower() in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp",
            }:
                key = hashlib.sha256(
                    f"{name}:{image.relative_to(directory).as_posix()}".encode()
                ).hexdigest()
                records.append(
                    Record(key, str(image), source_ids[synset], classname, "test")
                )
    if not records:
        raise ValueError(f"No images found for {name}")
    return validate_manifest(records)


def prepare_protocol(
    output,
    mode="base_new",
    shots=16,
    seed=1,
    root=None,
    dataset=None,
    manifest=None,
    target_manifests=None,
    domains=None,
):
    if mode not in {"base_new", "fewshot", "domain"}:
        raise ValueError("mode must be base_new, fewshot or domain")
    if not isinstance(shots, int) or isinstance(shots, bool) or shots < 1:
        raise ValueError("PLIKD few-shot training requires a positive shot count")
    output = Path(output).expanduser().resolve()
    if (output / "protocol.json").exists():
        raise FileExistsError(f"Prepared protocol already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if manifest:
        values = load_manifest(manifest)
        dataset = dataset or "custom"
    else:
        if root is None or dataset is None:
            raise ValueError("Provide a source manifest, or root and dataset")
        dataset = _dataset_name(dataset)
        values = prepare_dataset(root, dataset, output / "source.jsonl")
    names = class_names(values)
    if not names:
        raise ValueError("No source classes")
    all_ids = list(range(len(names)))
    original = {split: [r for r in values if r.split == split] for split in SPLITS}
    for split in ("train", "test"):
        if sorted({r.label for r in original[split]}) != all_ids:
            raise ValueError(f"Every source class must occur in {split}")
    midpoint = (len(names) + 1) // 2
    base_ids, new_ids = all_ids[:midpoint], all_ids[midpoint:]
    if mode == "base_new" and not new_ids:
        raise ValueError("Base-to-new evaluation needs at least two classes")
    training_ids = set(base_ids if mode == "base_new" else all_ids)
    selected = [r for r in values if r.label in training_ids]
    sampled = split_records(selected, shots, seed)
    outputs = {"train": sampled["train"], "val": sampled["val"]}
    evaluation_classes = {}
    if mode == "base_new":
        outputs["test_base"] = [r for r in original["test"] if r.label in training_ids]
        outputs["test_new"] = [
            r for r in original["test"] if r.label not in training_ids
        ]
        evaluation_classes.update({"test_base": base_ids, "test_new": new_ids})
    else:
        outputs["test"] = original["test"]
        evaluation_classes["test"] = all_ids
    source_names = defaultdict(list)
    for index, name in enumerate(names):
        source_names[name].append(index)
    if mode == "domain":
        if target_manifests:
            for target_name, target_path in sorted(target_manifests.items()):
                if (
                    target_name in outputs
                    or not target_name
                    or not target_name.replace("_", "").isalnum()
                ):
                    raise ValueError(f"Invalid target domain identifier: {target_name}")
                target = load_manifest(target_path)
                if not target or any(r.split != "test" for r in target):
                    raise ValueError(
                        "A target-domain manifest must contain test records only"
                    )
                converted = []
                for record in target:
                    if record.classname not in source_names:
                        raise ValueError(
                            f"Unknown source class in target domain: {record.classname}"
                        )
                    labels = source_names[record.classname]
                    if len(labels) > 1 and record.label not in labels:
                        raise ValueError(
                            f"Ambiguous target classname {record.classname!r}; provide a matching source global label"
                        )
                    label = labels[0] if len(labels) == 1 else record.label
                    identifier = hashlib.sha256(
                        f"{target_name}:{record.id}".encode()
                    ).hexdigest()
                    converted.append(
                        Record(
                            identifier,
                            record.image,
                            label,
                            record.classname,
                            "test",
                        )
                    )
                outputs[target_name] = converted
                evaluation_classes[target_name] = sorted({r.label for r in converted})
        else:
            if dataset != "imagenet" or root is None:
                raise ValueError(
                    "Domain mode requires ImageNet root or explicit target manifests"
                )
            for target_name in domains or (
                "imagenet_v2",
                "imagenet_sketch",
                "imagenet_a",
                "imagenet_r",
            ):
                if target_name not in {
                    "imagenet_v2",
                    "imagenet_sketch",
                    "imagenet_a",
                    "imagenet_r",
                }:
                    raise ValueError(f"Unsupported ImageNet target: {target_name}")
                target = _imagenet_variant(root, target_name, names)
                outputs[target_name] = target
                evaluation_classes[target_name] = sorted({r.label for r in target})
        validate_manifest(
            values
            + [
                r
                for key, rows in outputs.items()
                if key not in {"train", "val", "test"}
                for r in rows
            ]
        )
    save_manifest(values, output / "source.jsonl")
    for split, rows in outputs.items():
        save_manifest(rows, output / f"{split}.jsonl")
    metadata = {
        "schema": 1,
        "dataset": dataset,
        "mode": mode,
        "shots": shots,
        "seed": seed,
        "class_names": names,
        "training_classes": sorted(training_ids),
        "base_classes": base_ids,
        "new_classes": new_ids,
        "evaluation_classes": evaluation_classes,
        "source_sha256": _file_hash(output / "source.jsonl"),
        "split_hashes": {
            name: _file_hash(output / f"{name}.jsonl") for name in outputs
        },
        "counts": {name: len(rows) for name, rows in outputs.items()},
        "class_sampling": "sorted source ids; first ceil(C/2) base; one Python RNG for train then val",
    }
    _write_metadata(output / "protocol.json", metadata)
    return metadata


def load_prepared(output, split):
    output = Path(output)
    metadata = json.loads((output / "protocol.json").read_text())
    if split not in metadata["split_hashes"]:
        raise ValueError(f"Split {split} is not part of this prepared protocol")
    path = output / f"{split}.jsonl"
    if _file_hash(path) != metadata["split_hashes"][split]:
        raise ValueError(f"Prepared manifest changed or was modified: {split}")
    return load_manifest(path)


class ImageDataset:
    def __init__(self, records, transform, candidate_ids, knowledge_features=None):
        self.records = list(records)
        self.transform = transform
        self.label_map = {label: index for index, label in enumerate(candidate_ids)}
        if len(self.label_map) != len(candidate_ids) or not candidate_ids:
            raise ValueError("Candidate class ids must be nonempty and unique")
        if any(record.label not in self.label_map for record in self.records):
            raise ValueError("Image class is outside the selected candidate classes")
        self.knowledge = knowledge_features
        if knowledge_features is not None:
            if any(record.split != "train" for record in self.records):
                raise ValueError(
                    "Knowledge features are only permitted for training images"
                )
            if not {record.id for record in self.records}.issubset(knowledge_features):
                raise ValueError("Training knowledge features are incomplete")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from PIL import Image, ImageOps
        import torch

        record = self.records[index]
        with Image.open(record.image) as image:
            pixels = self.transform(ImageOps.exif_transpose(image).convert("RGB"))
        row = {
            "images": pixels,
            "labels": self.label_map[record.label],
            "ids": record.id,
        }
        if self.knowledge is not None:
            row["knowledge_features"] = torch.as_tensor(
                self.knowledge[record.id], dtype=torch.float32
            )
        return row


def collate_images(rows):
    import torch

    batch = {
        "images": torch.stack([r["images"] for r in rows]),
        "labels": torch.tensor([r["labels"] for r in rows], dtype=torch.long),
        "ids": [r["ids"] for r in rows],
    }
    if any("knowledge_features" in row for row in rows):
        if not all("knowledge_features" in row for row in rows):
            raise ValueError("Mixed training knowledge availability in one batch")
        batch["knowledge_features"] = torch.stack(
            [r["knowledge_features"] for r in rows]
        )
    return batch
