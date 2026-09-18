from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.api import datasets
from app.datasets.adapters import DatasetSchemaError, canonicalize, detect_schema, inspect_rows
from app.db.migrate import upgrade_database
from app.db.models import Dataset
from app.db.session import make_engine
from app.main import app


@pytest.mark.parametrize("row,expected", [
    ({"text": "नमस्ते 世界"}, "text"),
    ({"instruction": "Explain", "input": "context", "output": "answer"}, "instruction"),
    ({"prompt": "Hi", "response": "Hello"}, "prompt_response"),
    ({"prompt": "Hi", "completion": "Hello"}, "prompt_completion"),
    ({"question": "Why?", "answer": "Because", "context": "Context"}, "question_answer"),
    ({"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello"}]}, "sharegpt"),
])
def test_supported_adapters_preserve_input(row, expected):
    original = deepcopy(row)
    assert detect_schema(row) == expected
    result = canonicalize(row)
    assert result.source_format == expected
    assert row == original
    if expected == "text":
        assert result.text == "नमस्ते 世界"
    else:
        assert result.messages[-1].role == "assistant"


def test_multiturn_and_whitespace_are_preserved():
    turns = [{"role": "system", "content": "Be helpful"},
             {"role": "user", "content": " first "}, {"role": "assistant", "content": "one"},
             {"role": "user", "content": "second"}, {"role": "assistant", "content": "two"}]
    result = canonicalize({"messages": turns})
    assert [message.model_dump() for message in result.messages] == turns


def test_mapping_subclasses_like_hf_lazy_row_are_supported():
    from collections.abc import Mapping

    class CustomRow(Mapping):
        def __init__(self, d):
            self._d = d

        def __getitem__(self, k):
            return self._d[k]

        def __iter__(self):
            return iter(self._d)

        def __len__(self):
            return len(self._d)

    row = CustomRow({"instruction": "Explain", "output": "answer"})
    assert detect_schema(row) == "instruction"
    result = canonicalize(row)
    assert result.source_format == "instruction"
    assert result.messages[0].content == "Explain"
    assert result.messages[1].content == "answer"


@pytest.mark.parametrize("row", [
    {"prompt": "x", "response": 12},
    {"prompt": "x", "response": "y", "completion": "z"},
    {"messages": []},
    {"messages": [{"role": "assistant", "content": "a"}]},
    {"messages": [{"role": "user", "content": "x"}]},
    {"messages": [{"role": [], "content": "x"}]},
    {"conversations": [{"from": [], "value": "x"}]},
    {"instruction": "x"},
    {"prompt": "x", "response": "y", "input": "a", "context": "b"},
])
def test_invalid_or_ambiguous_schemas_are_not_guessed(row):
    with pytest.raises(DatasetSchemaError):
        canonicalize(row)


@pytest.mark.parametrize("row,kind", [
    ({"input_ids": [1, 2], "labels": [1, 2]}, "pretokenized"),
    ({"messages": [{"role": "assistant", "tool_calls": []}], "tools": []}, "tool_calling"),
])
def test_advanced_formats_are_recognized_but_worker_gated(row, kind):
    assert detect_schema(row) == kind
    with pytest.raises(DatasetSchemaError):
        canonicalize(row)
    report = inspect_rows([(1, row)])
    assert report.formats == [kind]
    assert not report.valid


def test_preference_format_is_canonical_for_alignment_only():
    row = {"prompt": "x", "chosen": "yes", "rejected": "no"}
    record = canonicalize(row)
    assert record.kind == "preference"
    report = inspect_rows([(1, row)])
    assert report.formats == ["preference"]
    assert report.valid


def test_mapping_requires_existing_fields_and_preview_is_bounded():
    assert canonicalize({"q": "Q", "a": "A"}, {"prompt": "q", "response": "a"}).messages[1].content == "A"
    with pytest.raises(DatasetSchemaError, match="missing"):
        canonicalize({"q": "Q"}, {"prompt": "q", "response": "a"})
    def rows():
        yield 1, {"text": "Only read this"}
        raise AssertionError("Preview consumed beyond its limit")
    assert inspect_rows(rows(), limit=1).sampled_rows == 1


def test_schema_endpoint_is_typed_and_does_not_modify_dataset(tmp_path, monkeypatch):
    engine = make_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    upgrade_database(engine)
    monkeypatch.setattr(datasets, "engine", engine)
    path = tmp_path / "data.jsonl"
    content = '{"prompt": "Hi", "completion": "Hello"}\n'
    path.write_text(content, encoding="utf-8")
    with Session(engine) as db:
        row = Dataset(name="test", kind="instruction", path=str(path), fmt="jsonl")
        db.add(row)
        db.commit()
        dataset_id = row.id
    # Do not start the global app lifespan/queue against the developer database.
    client = TestClient(app)
    try:
        response = client.get(f"/api/datasets/{dataset_id}/schema")
        assert response.status_code == 200
        assert response.json()["formats"] == ["prompt_completion"]
        assert response.json()["sample_only"] is True
        assert client.get(f"/api/datasets/{dataset_id}/schema?limit=101").status_code == 422
        assert client.get("/api/datasets/999/schema").status_code == 404
        assert path.read_text(encoding="utf-8") == content
    finally:
        client.close()
        engine.dispose()
