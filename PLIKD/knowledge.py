import base64
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import threading
import time


def _hash(value):
    return hashlib.sha256(value).hexdigest()


def _image(image):
    from PIL import Image, ImageOps

    if isinstance(image, (str, os.PathLike)):
        with Image.open(image) as opened:
            return ImageOps.exif_transpose(opened).convert("RGB")
    if isinstance(image, Image.Image):
        return ImageOps.exif_transpose(image).convert("RGB")
    raise TypeError("An image must be a local path or a PIL image")


def _image_payload(image, max_size):
    from PIL import Image

    if isinstance(image, (str, os.PathLike)):
        original_hash = _hash(Path(image).read_bytes())
    else:
        original = _image(image)
        original_hash = _hash(str(original.size).encode() + original.tobytes())
    prepared = _image(image)
    prepared.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    stream = io.BytesIO()
    prepared.save(stream, format="JPEG", quality=95)
    encoded = stream.getvalue()
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii"), {
        "original_sha256": original_hash,
        "upload_sha256": _hash(encoded),
    }


def _response_text(response):
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("Teacher returned no completion choices")
    choice = choices[0]
    message = choice.message
    if getattr(message, "refusal", None):
        raise ValueError("Teacher refused the request")
    if getattr(choice, "finish_reason", None) == "length":
        raise ValueError("Teacher output was truncated by the token limit")
    content = getattr(message, "content", None)
    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                pieces.append(item.get("text", ""))
            elif getattr(item, "type", None) == "text":
                pieces.append(getattr(item, "text", ""))
        content = "".join(pieces)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Teacher returned empty text")
    return content.strip()


class Teacher:
    def __init__(self, config):
        self.config = dict(config.get("teacher", {}))
        self.model = self.config.get("model", "gpt-4o")
        self.revision = self.config.get("revision")
        self.request_model = self.revision or self.model
        self.base_url = (
            os.environ.get(self.config.get("base_url_env", "OPENAI_BASE_URL"))
            or "https://api.openai.com/v1"
        )
        self.endpoint_sha256 = _hash(self.base_url.rstrip("/").encode())
        self.max_image_size = int(self.config.get("max_image_size", 1536))
        self.retries = int(self.config.get("retries", 5))
        if self.max_image_size < 1 or self.retries < 0:
            raise ValueError("Teacher image size and retry settings are invalid")
        cache = self.config.get("cache", "artifacts/teacher.sqlite3")
        if str(cache) != ":memory:":
            Path(cache).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
            cache = str(Path(cache).expanduser())
        self.connection = sqlite3.connect(cache, timeout=60, check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS completions (cache_key TEXT PRIMARY KEY, request TEXT NOT NULL, response TEXT NOT NULL, created_at REAL NOT NULL)"
        )
        self.connection.commit()
        self.lock = threading.RLock()
        self.client = None

    def _client(self):
        if self.client is None:
            api_key_env = self.config.get("api_key_env", "OPENAI_API_KEY")
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise EnvironmentError(
                    f"Set {api_key_env} before calling the teacher API"
                )
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key, base_url=self.base_url, max_retries=0)
        return self.client

    def generate(
        self, prompt, images=(), temperature=0, max_tokens=4096, sample_id=None
    ):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Teacher prompt must be nonempty text")
        temperature = float(temperature)
        max_tokens = int(max_tokens)
        if not 0 <= temperature <= 2 or max_tokens <= 0:
            raise ValueError("Teacher decoding parameters are invalid")
        image_data = [_image_payload(image, self.max_image_size) for image in images]
        request = {
            "schema": 1,
            "model": self.model,
            "revision": self.revision,
            "request_model": self.request_model,
            "endpoint_sha256": self.endpoint_sha256,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt": prompt,
            "images": [item[1] for item in image_data],
            "max_image_size": self.max_image_size,
            "sample_id": sample_id,
        }
        serialized = json.dumps(
            request, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        key = _hash(serialized.encode())
        with self.lock:
            cached = self.connection.execute(
                "SELECT response FROM completions WHERE cache_key = ?", (key,)
            ).fetchone()
        if cached is not None:
            return cached[0]
        content = [{"type": "text", "text": prompt}] + [
            {"type": "image_url", "image_url": {"url": data, "detail": "high"}}
            for data, metadata in image_data
        ]
        for attempt in range(self.retries + 1):
            try:
                response = self._client().chat.completions.create(
                    model=self.request_model,
                    messages=[{"role": "user", "content": content}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                text = _response_text(response)
                break
            except Exception as error:
                status = getattr(error, "status_code", None)
                retryable = (
                    status in (408, 409, 429)
                    or (isinstance(status, int) and status >= 500)
                    or type(error).__name__
                    in {
                        "APIConnectionError",
                        "APITimeoutError",
                        "RateLimitError",
                        "InternalServerError",
                    }
                )
                if not retryable:
                    raise
                if attempt >= self.retries:
                    raise RuntimeError(
                        f"Teacher request failed after {attempt + 1} attempts ({type(error).__name__})"
                    ) from error
                time.sleep(min(30, 0.5 * (2**attempt)))
        with self.lock:
            self.connection.execute(
                "INSERT OR IGNORE INTO completions (cache_key, request, response, created_at) VALUES (?, ?, ?, ?)",
                (key, serialized, text, time.time()),
            )
            self.connection.commit()
        return text

    def close(self):
        with self.lock:
            if self.connection is not None:
                self.connection.close()
                self.connection = None
            if self.client is not None:
                self.client.close()
                self.client = None


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


class Journal:
    def __init__(self, path, parameters):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path = self.path.with_suffix(self.path.suffix + ".meta.json")
        metadata = {
            "schema": 1,
            "signature": digest(parameters),
            "parameters": parameters,
        }
        if metadata_path.exists():
            old = json.loads(metadata_path.read_text())
            if old.get("signature") != metadata["signature"] or digest(
                old.get("parameters")
            ) != old.get("signature"):
                raise ValueError(
                    f"Artifact inputs or settings changed: {self.path}. Use a new work directory."
                )
            if old.get("content_sha256") is not None and (
                not self.path.is_file()
                or file_digest(self.path) != old["content_sha256"]
            ):
                raise ValueError(f"Finalized artifact content changed: {self.path}")
        else:
            if self.path.exists() and self.path.stat().st_size:
                raise ValueError(f"Artifact has no provenance metadata: {self.path}")
            write_json(metadata_path, metadata)
        rows = read_jsonl(self.path) if self.path.exists() else []
        self.rows = {x["id"]: x for x in rows}
        if len(rows) != len(self.rows):
            raise ValueError(f"Duplicate ids in {self.path}")
        self.path.touch(exist_ok=True)

    def append(self, row):
        if row["id"] in self.rows:
            raise ValueError(f"Duplicate artifact id: {row['id']}")
        import fcntl

        with self.path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.rows[row["id"]] = row

    def finalize(self):
        path = self.path.with_suffix(self.path.suffix + ".meta.json")
        metadata = json.loads(path.read_text())
        metadata.update(content_sha256=file_digest(self.path), row_count=len(self.rows))
        write_json(path, metadata)
        return self.path


FIELDS = ("visual", "functional", "contextual")


def question_one(classname):
    return (
        "Based on the given class name, describe its typical visual features, functional attributes, and contextual information. Be concise and ensure accuracy. "
        "Example: Class: airplanes. Visual Features: Wings, fuselage, engines, tail. "
        "Functional Attributes: Used for air transportation. Contextual Information: Airports, blue sky.\n"
        f"Class: {classname}\nReturn a JSON object with exactly visual, functional, contextual keys, each containing concise text."
    )


def question_two(prior):
    return (
        "Given the provided image and instance-aware knowledge, refine the knowledge by supplementing missing but relevant features and removing unrelated information. "
        "Ensure that the descriptions remain concise and aligned with the specific image.\n"
        f"Instance-aware knowledge: {prior}\nReturn a JSON object with exactly visual, functional, contextual keys."
    )


def question_three(classname, refined):
    return (
        "Please combine the refined instance-aware knowledge, the image, and the class name to obtain detailed and precise information about the instance. "
        "Focus on specific instance attributes such as visual details (e.g., skin color, texture) or contextual nuances. Ensure accuracy and keep the response concise.\n"
        f"Class: {classname}\nRefined instance-aware knowledge: {refined}\n"
        "Return a JSON object with exactly visual, functional, contextual keys."
    )


def parse_knowledge(value):
    import re

    if isinstance(value, str):
        text = value.strip()
        if text.startswith("```"):
            match = re.fullmatch(
                r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL | re.IGNORECASE
            )
            if not match:
                raise ValueError("Malformed knowledge output fence")
            text = match.group(1).strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            pattern = re.compile(
                r"(?im)^\s*(?:\d+[.)]\s*)?(?:\*\*)?(Visual(?: Features)?|Functional(?: Attributes)?|Contextual(?: Information)?)(?:\*\*)?\s*:\s*(?:\*\*)?"
            )
            matches = list(pattern.finditer(text))
            if len(matches) != 3 or text[: matches[0].start()].strip():
                raise ValueError("Expected three named knowledge dimensions") from exc
            value = {}
            for i, match in enumerate(matches):
                field = match.group(1).split()[0].lower()
                if field in value:
                    raise ValueError("Duplicate knowledge dimension")
                value[field] = text[
                    match.end() : matches[i + 1].start()
                    if i + 1 < len(matches)
                    else len(text)
                ].strip()
    if not isinstance(value, dict) or set(value) != set(FIELDS):
        raise ValueError(
            "Knowledge must contain exactly visual, functional and contextual fields"
        )
    result = {}
    for field in FIELDS:
        text = value[field]
        if isinstance(text, list):
            if any(not isinstance(part, str) or not part.strip() for part in text):
                raise ValueError(f"Invalid {field} knowledge list")
            text = "; ".join(text)
        if not isinstance(text, str) or not any(c.isalnum() for c in text):
            raise ValueError(f"Empty or invalid {field} knowledge")
        result[field] = text.strip()
    return result


def knowledge_text(fields, keep=FIELDS):
    fields = parse_knowledge(fields)
    if not keep or len(set(keep)) != len(keep) or not set(keep).issubset(FIELDS):
        raise ValueError("Select one or more unique knowledge dimensions")
    labels = {
        "visual": "Visual Features",
        "functional": "Functional Attributes",
        "contextual": "Contextual Information",
    }
    return "\n".join(
        f"{labels[field]}: {fields[field]}" for field in FIELDS if field in keep
    )


def clip_token_batch(texts, tokenize, content_tokens=64):
    import torch

    if not isinstance(content_tokens, int) or not 1 <= content_tokens <= 75:
        raise ValueError("CLIP knowledge content token limit must be in [1,75]")
    tokens = tokenize(texts, truncate=True)
    result = torch.zeros_like(tokens)
    for i, row in enumerate(tokens):
        end = int(row.argmax())
        count = min(max(end - 1, 0), content_tokens)
        result[i, 0] = row[0]
        result[i, 1 : count + 1] = row[1 : count + 1]
        result[i, count + 1] = row[end]
    return result


def training_signature(records, config, **extra):
    if any(record.split != "train" for record in records):
        raise ValueError("Instance knowledge may only be extracted for training images")
    if not records or len({record.id for record in records}) != len(records):
        raise ValueError("A nonempty unique training image manifest is required")
    return {
        "schema": 1,
        "teacher": config.get("teacher", {}),
        "samples": [
            {
                "id": r.id,
                "label": r.label,
                "classname": r.classname,
                "image_sha256": file_digest(r.image),
            }
            for r in records
        ],
        **extra,
    }


def validate_knowledge_source(records, source, config):
    source = Path(source)
    path = source.with_suffix(source.suffix + ".meta.json")
    if not path.is_file():
        raise ValueError("Knowledge source has no provenance metadata; import it first")
    metadata = json.loads(path.read_text())
    parameters = metadata.get("parameters", {})
    if metadata.get("signature") != digest(parameters) or metadata.get(
        "content_sha256"
    ) != file_digest(source):
        raise ValueError("Knowledge source content or provenance changed")
    expected = training_signature(records, config)
    if any(
        parameters.get(key) != value for key, value in expected.items()
    ) or metadata.get("row_count") != len(records):
        raise ValueError(
            "Knowledge source teacher or training image provenance changed"
        )
    return metadata


def extract_knowledge(records, output, config):
    from dataclasses import asdict

    parameters = training_signature(records, config)
    journal = Journal(output, parameters)
    teacher = Teacher(config)
    max_tokens = int(config.get("teacher", {}).get("max_tokens", 4096))
    try:
        priors = {}
        for record in records:
            if record.id in journal.rows:
                continue
            if record.classname not in priors:
                prior = teacher.generate(
                    question_one(record.classname), temperature=0, max_tokens=max_tokens
                )
                parse_knowledge(prior)
                priors[record.classname] = prior
            prior = priors[record.classname]
            refined = teacher.generate(
                question_two(prior),
                [record.image],
                temperature=0,
                max_tokens=max_tokens,
            )
            parse_knowledge(refined)
            final = teacher.generate(
                question_three(record.classname, refined),
                [record.image],
                temperature=0,
                max_tokens=max_tokens,
            )
            fields = parse_knowledge(final)
            journal.append(
                {**asdict(record), **fields, "q": prior, "qr": refined, "qm": final}
            )
            print(f"knowledge {len(journal.rows)}/{len(records)}", flush=True)
    finally:
        teacher.close()
    return journal.finalize()


def import_knowledge(records, source, output, config):
    from dataclasses import asdict

    rows = read_jsonl(source)
    indexed = {row["id"]: row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != {r.id for r in records}:
        raise ValueError(
            "Imported knowledge must exactly cover the selected training ids"
        )
    parsed = []
    for record in records:
        row = indexed[record.id]
        if (
            row.get("classname", record.classname) != record.classname
            or row.get("label", record.label) != record.label
        ):
            raise ValueError(
                "Imported knowledge class differs from the training manifest"
            )
        if row.get("split", "train") != "train":
            raise ValueError("Imported knowledge split must be training")
        if "image" in row:
            image = Path(row["image"]).expanduser()
            if not image.is_absolute():
                image = Path(source).resolve().parent / image
            if image.resolve() != Path(record.image).resolve():
                raise ValueError(
                    "Imported knowledge image differs from the training instance"
                )
        fields = parse_knowledge({field: row[field] for field in FIELDS})
        parsed.append({**asdict(record), **fields, "source": "imported"})
    journal = Journal(
        output, training_signature(records, config, imported_sha256=file_digest(source))
    )
    for row in parsed:
        if row["id"] not in journal.rows:
            journal.append(row)
    return journal.finalize()


def encode_knowledge(
    records, source, output, clip_model, tokenize, config, backbone_fingerprint
):
    import numpy as np
    import torch

    validate_knowledge_source(records, source, config)
    rows = read_jsonl(source)
    mapping = {row["id"]: row for row in rows}
    if len(mapping) != len(rows) or set(mapping) != {r.id for r in records}:
        raise ValueError("Knowledge records do not exactly cover training images")
    keep = config.get("knowledge", {}).get("dimensions", list(FIELDS))
    limit = int(config.get("knowledge", {}).get("content_tokens", 64))
    parameters = training_signature(
        records,
        config,
        source_sha256=file_digest(source),
        backbone_fingerprint=backbone_fingerprint,
        dimensions=keep,
        content_tokens=limit,
    )
    target = Path(output)
    if target.exists():
        features, metadata = load_knowledge_features(target, records)
        if metadata != parameters:
            raise ValueError(
                "Knowledge feature provenance changed; use a new run directory"
            )
        return features
    texts = []
    for record in records:
        row = mapping[record.id]
        if row.get("classname") != record.classname or row.get("label") != record.label:
            raise ValueError("Knowledge feature class identity mismatch")
        texts.append(knowledge_text({field: row[field] for field in FIELDS}, keep))
    device = next(clip_model.parameters()).device
    batch_size = int(config.get("knowledge", {}).get("batch_size", 128))
    if batch_size < 1:
        raise ValueError("Knowledge encoding batch size must be positive")
    features = []
    clip_model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            tokens = clip_token_batch(
                texts[start : start + batch_size], tokenize, limit
            ).to(device)
            encoded = clip_model.encode_text(tokens).float()
            features.append(
                torch.nn.functional.normalize(encoded, dim=-1).cpu().numpy()
            )
    values = np.concatenate(features)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            features=values,
            ids=np.array([r.id for r in records]),
            metadata=np.array(canonical(parameters)),
        )
    temporary.replace(target)
    return {record.id: feature for record, feature in zip(records, values)}


def load_knowledge_features(path, records):
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        ids = data["ids"].tolist()
        features = data["features"]
        metadata = json.loads(str(data["metadata"]))
    if (
        len(ids) != len(set(ids))
        or set(ids) != {r.id for r in records}
        or features.ndim != 2
        or len(features) != len(ids)
        or not np.isfinite(features).all()
    ):
        raise ValueError("Invalid or incomplete training knowledge features")
    return {identifier: feature for identifier, feature in zip(ids, features)}, metadata
