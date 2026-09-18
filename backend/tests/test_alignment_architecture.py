from __future__ import annotations

import json
import sys
import types

import pytest

from app.datasets.adapters import KTORecord, PreferenceRecord, canonicalize
from app.domain import HardwareProfile, Method, RunConfig, TaskType
from app.integrations.estimator import ModelSpec, estimate
from app.train_entry.alignment import PreferenceTrainer, normalize_alignment_metrics
from app.train_entry.tokenization import render_kto_row, render_preference_row


class FakeTokenizer:
    chat_template = "test-template"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt=False):
        assert tokenize is False
        rendered = "".join(f"{item['role']}:{item['content']}|" for item in messages)
        return rendered + ("assistant:" if add_generation_prompt else "")


def _alignment_config(**updates) -> RunConfig:
    values = {
        "backend": "transformers",
        "task": TaskType.ALIGNMENT,
        "method": Method.LORA,
        "base_model": "org/model",
        "dataset_id": 1,
        "output_name": "alignment",
    }
    values.update(updates)
    return RunConfig(**values)


def test_preference_and_kto_rows_share_canonical_rendering_contract():
    tokenizer = FakeTokenizer()
    preference = {"prompt": "Question", "chosen": "Good", "rejected": "Bad"}
    canonical_preference = canonicalize(preference)
    assert isinstance(canonical_preference, PreferenceRecord)
    rendered = render_preference_row(preference, tokenizer)
    assert rendered == {
        "prompt": "user:Question|assistant:",
        "chosen": "Good|",
        "rejected": "Bad|",
    }

    kto = {"prompt": "Question", "response": "Answer", "desirable": True}
    canonical_kto = canonicalize(kto)
    assert isinstance(canonical_kto, KTORecord)
    assert render_kto_row(kto, tokenizer)["label"] is True


def test_alignment_config_enforces_reference_semantics():
    with pytest.raises(ValueError, match="reference-free"):
        _alignment_config(alignment={"objective": "orpo"})
    reference_free = _alignment_config(
        alignment={"objective": "orpo", "reference": {"strategy": "none"}}
    )
    assert reference_free.alignment.reference.strategy == "none"
    for objective in ("dpo", "ipo", "kto"):
        with pytest.raises(ValueError, match="explicit reference"):
            _alignment_config(alignment={"objective": objective, "reference": {"strategy": "none"}})
    with pytest.raises(ValueError, match="reference.revision"):
        _alignment_config(alignment={"reference": {"strategy": "base_model", "revision": "ignored-commit"}})
    with pytest.raises(ValueError, match="adapter-based"):
        _alignment_config(
            method=Method.FULL,
            alignment={"reference": {"strategy": "adapter_disabled"}},
        )


def test_reference_model_memory_is_visible_before_launch():
    cfg = _alignment_config(method=Method.FULL)
    result = estimate(cfg, HardwareProfile(), ModelSpec(1_000_000, 128, 4))
    assert result.reference_model_mb == result.weights_mb
    without_reference = _alignment_config(
        method=Method.FULL,
        alignment={"objective": "orpo", "reference": {"strategy": "none"}},
    )
    no_reference_result = estimate(without_reference, HardwareProfile(), ModelSpec(1_000_000, 128, 4))
    assert no_reference_result.reference_model_mb == 0
    assert result.total_mb > no_reference_result.total_mb


def test_metric_normalization_is_stable_across_alignment_engines():
    normalized = normalize_alignment_metrics(
        {"rewards/chosen": 1, "rewards/rejected": -1.0, "rewards/margins": 2, "loss": 0.25}
    )
    assert normalized == {
        "preference_chosen_reward": 1.0,
        "preference_rejected_reward": -1.0,
        "preference_reward_margin": 2.0,
        "loss": 0.25,
    }


def test_dpo_runtime_does_not_import_unrelated_objective_trainers(monkeypatch):
    module = types.ModuleType("trl")
    module.DPOConfig = type("DPOConfig", (), {})
    module.DPOTrainer = type("DPOTrainer", (), {})
    monkeypatch.setitem(sys.modules, "trl", module)
    trainer = PreferenceTrainer(
        _alignment_config(),
        ctx=None,
        emit=lambda _event: None,
        load_trainable_model=lambda *_args: None,
        callback_factory=lambda *_args: None,
    )
    trainer_cls, config_cls, options = trainer._objective_runtime()
    assert trainer_cls is module.DPOTrainer
    assert config_cls is module.DPOConfig
    assert options["loss_type"] == "sigmoid"


def test_dpo_loss_variant_and_objective_specific_validation(monkeypatch):
    module = types.ModuleType("trl")
    module.DPOConfig = type("DPOConfig", (), {})
    module.DPOTrainer = type("DPOTrainer", (), {})
    monkeypatch.setitem(sys.modules, "trl", module)
    trainer = PreferenceTrainer(
        _alignment_config(alignment={"dpo_loss_variant": "robust", "label_smoothing": 0.1}),
        ctx=None,
        emit=lambda _event: None,
        load_trainable_model=lambda *_args: None,
        callback_factory=lambda *_args: None,
    )
    assert trainer._objective_runtime()[2] == {
        "beta": 0.1,
        "loss_type": "robust",
        "label_smoothing": 0.1,
    }
    with pytest.raises(ValueError, match="hinge loss"):
        _alignment_config(alignment={"dpo_loss_variant": "hinge", "label_smoothing": 0.1})
    with pytest.raises(ValueError, match="valid only for the DPO"):
        _alignment_config(
            alignment={"objective": "ipo", "dpo_loss_variant": "robust"}
        )


def test_reward_model_uses_reward_trainer_without_reference(monkeypatch):
    module = types.ModuleType("trl")
    module.RewardConfig = type("RewardConfig", (), {})
    module.RewardTrainer = type("RewardTrainer", (), {})
    monkeypatch.setitem(sys.modules, "trl", module)
    cfg = _alignment_config(
        method=Method.FULL,
        alignment={"objective": "reward_model", "reference": {"strategy": "none"}},
    )
    trainer = PreferenceTrainer(
        cfg,
        ctx=None,
        emit=lambda _event: None,
        load_trainable_model=lambda *_args: None,
        load_reward_model=lambda *_args: None,
        callback_factory=lambda *_args: None,
    )
    trainer_cls, config_cls, options = trainer._objective_runtime()
    assert trainer_cls is module.RewardTrainer
    assert config_cls is module.RewardConfig
    assert options == {}


def test_reward_metric_aliases_are_normalized():
    assert normalize_alignment_metrics({
        "reward/chosen": 2.0,
        "reward/rejected": 0.5,
        "reward/margins": 1.5,
        "reward/accuracies": 0.75,
        "eval_logps/chosen": -2.0,
        "eval_logps/rejected": -3.0,
    }) == {
        "preference_chosen_reward": 2.0,
        "preference_rejected_reward": 0.5,
        "preference_reward_margin": 1.5,
        "preference_accuracy": 0.75,
        "preference_chosen_logprob": -2.0,
        "preference_rejected_logprob": -3.0,
    }


def test_alignment_memory_accounts_for_precision_pairs_and_separate_reference():
    policy = ModelSpec(1_000_000, 128, 4)
    cfg = _alignment_config(method=Method.FULL, runtime={"precision": "fp32"},
                            alignment={"reference": {"strategy": "separate_model", "model": "org/reference"}})
    result = estimate(cfg, HardwareProfile(), policy, ModelSpec(3_000_000))
    assert result.weights_mb == pytest.approx(1_000_000 * 4 / (1024 * 1024), abs=0.1)
    assert result.reference_model_mb == pytest.approx(3 * result.weights_mb, abs=0.2)
    sft = cfg.model_copy(deep=True)
    sft.task = TaskType.FINETUNE
    assert result.activations_mb == pytest.approx(2 * estimate(sft, HardwareProfile(), policy).activations_mb, abs=0.1)
    assert result.gradients_mb == result.weights_mb


@pytest.mark.parametrize("row", [
    {"prompt": "Q", "chosen": "A", "rejected": "A"},
    {"prompt": "Q", "chosen": "", "rejected": "B"},
    {"prompt": "Q", "response": "A", "desirable": "yes"},
])
def test_malformed_alignment_rows_fail_validation(tmp_path, row):
    from app.datasets.validate import validate
    from app.domain import DatasetKind

    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    kind = DatasetKind.KTO if "desirable" in row else DatasetKind.PREFERENCE
    report, stats = validate(path, kind)
    assert not report.ok and stats.invalid_rows == 1


def test_kto_class_balance_is_validated(tmp_path):
    from app.datasets.validate import validate
    from app.domain import DatasetKind

    path = tmp_path / "kto.jsonl"
    rows = [{"prompt": f"Q{i}", "response": "A", "desirable": True} for i in range(6)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report, _ = validate(path, DatasetKind.KTO)
    assert not report.ok and any("both desirable" in issue.message for issue in report.issues)
    rows.append({"prompt": "Other", "response": "B", "desirable": False})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report, _ = validate(path, DatasetKind.KTO)
    assert report.ok and any("imbalance" in issue.message for issue in report.issues)


def test_interrupted_alignment_checkpoints_are_not_resumable(tmp_path):
    from app.train_entry.run import _alignment_checkpoint_complete

    checkpoint = tmp_path / "checkpoint-2"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text('{"global_step": 2}', encoding="utf-8")
    for name in ["optimizer.pt", "scheduler.pt", "adapter_model.safetensors"]:
        (checkpoint / name).write_bytes(b"nonempty fixture")
    assert _alignment_checkpoint_complete(checkpoint)
    (checkpoint / "scheduler.pt").write_bytes(b"")
    assert not _alignment_checkpoint_complete(checkpoint)
    (checkpoint / "trainer_state.json").write_text("{", encoding="utf-8")
    assert not _alignment_checkpoint_complete(checkpoint)
