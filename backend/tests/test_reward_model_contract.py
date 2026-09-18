from __future__ import annotations

import pytest

from app.model_refs import ResolvedModel, normalize_artifact_kind
from app.train_entry import eval_run, model_runtime


def test_registry_categories_preserve_historical_artifact_values():
    assert normalize_artifact_kind("model") == "causal_lm"
    assert normalize_artifact_kind("finetune") == "causal_lm"
    assert normalize_artifact_kind("adapter") == "adapter"
    assert normalize_artifact_kind("merged") == "merged_model"
    assert normalize_artifact_kind("gguf") == "quantized_model"
    assert normalize_artifact_kind("reward_model") == "reward_model"
    assert normalize_artifact_kind("reference_model") == "reference_model"


def test_reward_artifact_advertises_scoring_and_disallows_deployment():
    resolved = ResolvedModel(
        requested_ref="run:7",
        load_ref="output",
        kind="adapter",
        label="reward",
        artifact_kind="reward_model",
    )
    assert resolved.model_category == "reward_model"
    assert resolved.deployable is False
    assert resolved.evaluation_capabilities == [
        "pairwise_accuracy",
        "chosen_score",
        "rejected_score",
        "reward_margin",
    ]


def test_generation_runtime_rejects_reward_artifacts_before_heavy_imports(monkeypatch):
    reward = ResolvedModel(
        requested_ref="run:7",
        load_ref="output",
        kind="transformers",
        label="reward",
        artifact_kind="reward_model",
    )
    monkeypatch.setattr(model_runtime, "resolve_model_ref", lambda _ref: reward)
    with pytest.raises(ValueError, match="cannot be loaded for text generation"):
        model_runtime.load_runtime("run:7")


def test_pairwise_reward_evaluation_reports_scores_margin_and_accuracy(monkeypatch):
    class Tokenizer:
        chat_template = "test"

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt=False):
            assert tokenize is False
            text = "".join(f"{item['role']}:{item['content']}|" for item in messages)
            return text + ("assistant:" if add_generation_prompt else "")

    class Runtime:
        tokenizer = Tokenizer()

        @staticmethod
        def score(text):
            return 2.0 if "Good" in text else 0.5

        @staticmethod
        def unload():
            return None

    monkeypatch.setattr(
        "app.train_entry.model_runtime.load_reward_runtime",
        lambda _model_ref: Runtime(),
    )
    result = eval_run.eval_reward_model(
        "reward",
        [
            {"prompt": "Q1", "chosen": "Good", "rejected": "Bad"},
            {"prompt": "Q2", "chosen": "Bad", "rejected": "Good"},
        ],
    )
    assert result["scores"] == {
        "chosen_score": 1.25,
        "rejected_score": 1.25,
        "reward_margin": 0.0,
        "pairwise_accuracy": 0.5,
    }
    assert result["kind"] == "reward_model"
