import json

from PIL import Image
import pytest
import torch


def test_three_questions_keep_each_required_evidence_source():
    from knowledge import question_one, question_two, question_three

    first = question_one("panda")
    assert "airplanes" in first and "panda" in first
    assert all(
        value in first.lower() for value in ("visual", "functional", "contextual")
    )
    assert "prior knowledge" in question_two("prior knowledge")
    assert "refined evidence" in question_three("panda", "refined evidence")


def test_knowledge_dimensions_are_removed_before_encoding():
    from knowledge import parse_knowledge, knowledge_text

    parsed = parse_knowledge(
        json.dumps(
            {
                "visual": ["striped fur"],
                "functional": "climbing",
                "contextual": "forest",
            }
        )
    )
    text = knowledge_text(parsed, ["visual", "contextual"])
    assert "striped fur" in text and "forest" in text and "climbing" not in text
    with pytest.raises(ValueError):
        parse_knowledge('{"visual":"wings","functional":"flight"}')


def test_clip_knowledge_truncation_keeps_64_content_tokens_and_eos():
    from knowledge import clip_token_batch

    def tokenize(texts, truncate=False):
        return torch.tensor([[1000, *list(range(1, 76)), 2000] for _ in texts])

    tokens = clip_token_batch(["long knowledge"], tokenize, 64)
    assert tokens.shape == (1, 77)
    assert tokens[0, 0] == 1000
    assert tokens[0, 1:65].tolist() == list(range(1, 65))
    assert tokens[0, 65] == 2000
    assert tokens[0, 66:].tolist() == [0] * 11


def test_extraction_rejects_heldout_images_before_contacting_teacher(tmp_path):
    from data import Record
    from knowledge import extract_knowledge

    image = tmp_path / "test.png"
    Image.new("RGB", (4, 4)).save(image)
    row = Record("test-1", str(image), 0, "panda", "test")
    with pytest.raises(ValueError, match="training|train"):
        extract_knowledge([row], tmp_path / "knowledge.jsonl", {"teacher": {}})


def test_imported_knowledge_requires_exact_training_ids_and_class_names(tmp_path):
    from data import Record
    from knowledge import import_knowledge

    image = tmp_path / "train.png"
    Image.new("RGB", (4, 4)).save(image)
    row = Record("train-1", str(image), 0, "panda", "train")
    source = tmp_path / "import.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "train-1",
                "classname": "wrong class",
                "visual": "fur",
                "functional": "foraging",
                "contextual": "forest",
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="class"):
        import_knowledge([row], source, tmp_path / "knowledge.jsonl", {})


@pytest.mark.parametrize(
    "conflict", [{"split": "test"}, {"image": "another-image.png"}]
)
def test_imported_knowledge_rejects_inconsistent_instance_identity(tmp_path, conflict):
    from data import Record
    from knowledge import import_knowledge

    image = tmp_path / "train.png"
    Image.new("RGB", (4, 4)).save(image)
    row = Record("train-1", str(image), 0, "panda", "train")
    source = tmp_path / "import.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": row.id,
                "visual": "fur",
                "functional": "foraging",
                "contextual": "forest",
                **conflict,
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="instance|image|training|split"):
        import_knowledge([row], source, tmp_path / "knowledge.jsonl", {})


@pytest.mark.parametrize("change", ["teacher", "image", "knowledge"])
def test_encoding_cannot_relabel_old_knowledge_as_a_new_source(tmp_path, change):
    from data import Record
    from knowledge import import_knowledge, encode_knowledge

    image = tmp_path / "train.png"
    Image.new("RGB", (4, 4), "red").save(image)
    records = [Record("train-1", str(image), 0, "panda", "train")]
    config = {"teacher": {"model": "original-teacher"}, "knowledge": {"batch_size": 1}}
    source = tmp_path / "import.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "train-1",
                "visual": "fur",
                "functional": "foraging",
                "contextual": "forest",
            }
        )
        + "\n"
    )
    knowledge = tmp_path / "knowledge.jsonl"
    import_knowledge(records, source, knowledge, config)
    if change == "teacher":
        config["teacher"]["model"] = "another-teacher"
    elif change == "image":
        Image.new("RGB", (4, 4), "blue").save(image)
    else:
        knowledge.write_text(knowledge.read_text().replace("forest", "ocean"))

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

        def encode_text(self, tokens):
            return torch.ones(len(tokens), 4)

    def tokenize(texts, truncate=False):
        return torch.tensor([[10, 1, 20, *([0] * 74)] for _ in texts])

    with pytest.raises(
        ValueError, match="provenance|source|changed|knowledge|Knowledge"
    ):
        encode_knowledge(
            records,
            knowledge,
            tmp_path / "features.npz",
            Encoder(),
            tokenize,
            config,
            "weights",
        )
