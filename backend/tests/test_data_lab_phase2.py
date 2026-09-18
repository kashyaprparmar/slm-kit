import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlmodel import Session, select

from app.api import data_lab
from app.api import datasets as datasets_api
from app.datasets.lineage import backfill_legacy_versions, current_version, register_version
from app.datasets.prepare import prepare
from app.datasets.quality import profile as quality_profile
from app.db.migrate import upgrade_database
from app.db.models import Dataset, DatasetVersion
from app.db.session import make_engine
from app.domain import DatasetKind
from app.train_entry.tokenization import TokenizerDataError, render_and_tokenize


class FakeTokenizer:
    eos_token = "</s>"
    eos_token_id = 2
    unk_token_id = 0
    chat_template = "native"

    def __call__(self, text, add_special_tokens=True):
        prefix = [1] if add_special_tokens else []
        return {"input_ids": prefix + [ord(char) for char in text]}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                            return_dict=False, return_assistant_tokens_mask=False, **_kwargs):
        text = "<s>" + "".join(f"[{m['role']}]{m['content']}" for m in messages)
        if add_generation_prompt:
            text += "[assistant]"
        else:
            text += "</s>"
        if not tokenize:
            return text
        ids = [ord(char) for char in text]
        if not return_dict:
            return ids
        mask = [0] * len(ids)
        if return_assistant_tokens_mask:
            cursor = 0
            for message in messages:
                marker = f"[{message['role']}]"
                start = text.index(marker, cursor) + len(marker)
                end = start + len(message["content"])
                if message["role"] == "assistant":
                    mask[start:end] = [1] * (end - start)
                cursor = end
        return {"input_ids": ids, "assistant_masks": mask}


def _rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_streamed_preparation_is_reproducible_and_three_way(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps({"prompt": f"Question {i}", "completion": f"Answer {i}"}) + "\n" for i in range(20)), encoding="utf-8")
    first, dropped = prepare(source, tmp_path / "one", DatasetKind.INSTRUCTION, {}, .2, True, True, False, 42, .2)
    second, _ = prepare(source, tmp_path / "two", DatasetKind.INSTRUCTION, {}, .2, True, True, False, 42, .2)
    assert dropped == 0
    assert [item[0] for item in first] == ["train", "validation", "test"]
    assert [item[4].num_rows for item in first] == [12, 4, 4]
    assert [_rows(item[1]) for item in first] == [_rows(item[1]) for item in second]
    assert all(set(row) == {"messages"} for item in first for row in _rows(item[1]))
    assert not list(tmp_path.rglob("*.part")) and not list(tmp_path.rglob(".*.sqlite3"))


def test_preparation_failure_is_atomic(tmp_path):
    source = tmp_path / "bad.jsonl"
    source.write_text('{"prompt":"ok","response":"yes"}\n{"prompt":"missing answer"}\n', encoding="utf-8")
    destination = tmp_path / "prepared"
    with pytest.raises(ValueError, match="Row 2"):
        prepare(source, destination, DatasetKind.INSTRUCTION, {}, 0, False, True, False, 42)
    assert not [path for path in destination.iterdir() if path.is_file()]


def test_prepare_api_persists_and_reuses_recipe_after_session_commit(tmp_path, monkeypatch):
    test_engine = make_engine(f"sqlite:///{(tmp_path / 'api.db').as_posix()}")
    upgrade_database(test_engine)
    source = tmp_path / "api-source.jsonl"
    source.write_text("".join(json.dumps({"prompt": f"Q{i}", "response": f"A{i}"}) + "\n"
                              for i in range(10)), encoding="utf-8")
    with Session(test_engine) as db:
        dataset = Dataset(name="API source", kind="instruction", path=str(source), fmt="jsonl")
        db.add(dataset)
        db.commit()
        dataset_id = dataset.id
    monkeypatch.setattr(datasets_api, "engine", test_engine)
    monkeypatch.setattr(datasets_api, "_settings", SimpleNamespace(datasets_dir=tmp_path / "datasets"))
    body = datasets_api.PrepareBody(test_fraction=.2, validation_fraction=.2, seed=7)
    first = asyncio.run(datasets_api.prepare_dataset(dataset_id, body))
    replay = asyncio.run(datasets_api.prepare_dataset(dataset_id, body))
    assert first["reused"] is False and replay["reused"] is True
    assert first["recipe_id"] == replay["recipe_id"]
    assert len(first["datasets"]) == len(replay["datasets"]) == 3
    assert all(Path(row.path).is_file() for row in first["datasets"])
    test_engine.dispose()


def test_version_backfill_references_original_without_copy(tmp_path):
    engine = make_engine(f"sqlite:///{(tmp_path / 'db.sqlite').as_posix()}")
    upgrade_database(engine)
    source = tmp_path / "original.txt"
    source.write_text("unchanged", encoding="utf-8")
    with Session(engine) as db:
        dataset = Dataset(name="legacy", kind="domain_corpus", path=str(source), fmt="txt")
        db.add(dataset)
        db.commit()
        dataset_id = dataset.id
    assert backfill_legacy_versions(engine) == 1
    assert backfill_legacy_versions(engine) == 0
    with Session(engine) as db:
        version = current_version(db, db.get(Dataset, dataset_id))
        assert version.path == str(source)
        assert version.fingerprint == register_version(db, db.get(Dataset, dataset_id)).fingerprint
        assert len(db.exec(select(DatasetVersion)).all()) == 1
    assert source.read_text(encoding="utf-8") == "unchanged"


def test_loss_masks_and_template_requirements():
    row = {"messages": [{"role": "user", "content": "Question"},
                        {"role": "assistant", "content": "Answer"}]}
    tokenizer = FakeTokenizer()
    full = render_and_tokenize(row, tokenizer, max_length=1000)
    completion = render_and_tokenize(row, tokenizer, max_length=1000, loss_policy="completion_only")
    assistant = render_and_tokenize(row, tokenizer, max_length=1000, loss_policy="assistant_only")
    assert all(full["loss_mask"])
    assert not any(completion["loss_mask"][:completion["rendered"].index("Answer")])
    assert sum(assistant["loss_mask"]) == len("Answer")
    assert full["input_ids"][0] == ord("<")  # no duplicate tokenizer BOS
    tokenizer.chat_template = None
    with pytest.raises(TokenizerDataError, match="requires a chat template"):
        render_and_tokenize(row, tokenizer, max_length=1000)
    with pytest.raises(TokenizerDataError, match="removed every target"):
        render_and_tokenize(row, FakeTokenizer(), max_length=2, loss_policy="completion_only")


@pytest.mark.parametrize("revision", [None, "", "main", "v1.2", "refs/pr/7", "a" * 39, "g" * 40])
def test_mutable_tokenizer_revisions_never_take_the_pre_resolution_cache_path(revision):
    assert data_lab._is_immutable_hf_revision(revision) is False


def test_full_hf_commit_revision_can_take_the_pre_resolution_cache_path():
    assert data_lab._is_immutable_hf_revision("0123456789abcdefABCDEF0123456789abcdefAB") is True


def test_quality_profile_reports_examples_without_mutation(tmp_path):
    path = tmp_path / "quality.jsonl"
    rows = [
        {"prompt": "Contact person at name@example.com", "response": "First answer"},
        {"prompt": "Contact person at name@example.com", "response": "Different answer"},
        {"prompt": "Contact   person at NAME@example.com", "response": "First answer"},
    ]
    original = "".join(json.dumps(row) + "\n" for row in rows)
    path.write_text(original, encoding="utf-8")
    report = quality_profile(str(path), "jsonl")
    codes = {warning["code"] for warning in report["warnings"]}
    assert {"possible_pii", "conflicting_answers", "excessive_whitespace"} <= codes
    assert all(warning["examples"] for warning in report["warnings"])
    assert report["destructive_changes"] is False
    assert path.read_text(encoding="utf-8") == original
