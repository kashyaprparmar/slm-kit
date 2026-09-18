"""Offline tiny CPU acceptance jobs; skip cleanly without the optional ML stack."""
from __future__ import annotations

import json
import math

import pytest
from sqlmodel import Session, SQLModel, create_engine

from app.backends.base import RunContext
from app.backends.unsloth_backend import UnslothBackend
from app.db.models import Dataset
from app.domain import RunConfig

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("peft")
pytestmark = pytest.mark.alignment_runtime


@pytest.fixture
def tiny_alignment(tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    from app import model_refs
    from app.db import session

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    torch.set_num_threads(1)
    vocab = {word: index for index, word in enumerate(
        ["<pad>", "<eos>", "<unk>", "user", "assistant", ":", "Question", "Good", "Bad", "Answer"]
    )}
    raw = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>",
        model_input_names=["input_ids", "attention_mask"],
    )
    tokenizer.chat_template = "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    base = tmp_path / "base"
    tokenizer.save_pretrained(base)
    LlamaForCausalLM(LlamaConfig(
        vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=128, pad_token_id=0, eos_token_id=1,
    )).save_pretrained(base)
    database = create_engine(f"sqlite:///{(tmp_path / 'review.db').as_posix()}")
    SQLModel.metadata.create_all(database)
    monkeypatch.setattr(session, "engine", database)
    monkeypatch.setattr(model_refs, "engine", database)

    def make(objective="dpo", method="full", reference="base_model", steps=1):
        data = tmp_path / f"{objective}.jsonl"
        rows = (
            [{"prompt": "Question", "response": "Good" if i % 2 else "Bad", "desirable": bool(i % 2)} for i in range(8)]
            if objective == "kto" else
            [{"prompt": "Question", "chosen": "Good Answer", "rejected": "Bad Answer"} for _ in range(4)]
        )
        data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        with Session(database) as db:
            row = Dataset(name=objective, kind="kto" if objective == "kto" else "preference", path=str(data), fmt="jsonl")
            db.add(row)
            db.commit()
            db.refresh(row)
            dataset_id = row.id
        workdir = tmp_path / f"{objective}-{method}-{reference}"
        checkpoints = workdir / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        cfg = RunConfig(
            backend="transformers", task="alignment", method=method,
            base_model=str(base), dataset_id=dataset_id, output_name="tiny",
            gradient_checkpointing=False,
            runtime={"precision": "fp32", "gradient_checkpointing": "off"},
            lora={"r": 2, "alpha": 4},
            alignment={"objective": objective, "reference": {"strategy": reference}},
            optim={"optimizer": "adamw_torch"},
            train={"max_steps": steps, "per_device_batch_size": 2, "gradient_accumulation": 1,
                   "max_seq_length": 32, "save_steps": 1, "logging_steps": 1},
        )
        return cfg, RunContext(run_id=1, workdir=workdir, checkpoint_dir=checkpoints)

    return make


@pytest.mark.parametrize("objective", ["dpo", "ipo", "orpo", "simpo", "kto", "reward_model"])
def test_tiny_objective_trains_saves_and_reloads(tiny_alignment, objective):
    reference = "none" if objective in {"orpo", "simpo", "reward_model"} else "base_model"
    cfg, ctx = tiny_alignment(objective, reference=reference)
    events = []
    UnslothBackend("transformers")._train(cfg, ctx, events.append)
    artifacts = [event for event in events if event.type == "artifact"]
    assert len(artifacts) == 1
    assert artifacts[0].metadata["reference"]["strategy"] == reference
    metrics = [event.metrics for event in events if event.type == "metric"]
    assert metrics and all(math.isfinite(value) for metric in metrics for value in metric.values())
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

    loader = AutoModelForSequenceClassification if objective == "reward_model" else AutoModelForCausalLM
    reloaded = loader.from_pretrained(ctx.workdir / "output")
    assert sum(parameter.numel() for parameter in reloaded.parameters()) > 0
    if objective == "dpo":
        metric = next(item for item in metrics if "preference_reward_margin" in item)
        assert {"preference_chosen_reward", "preference_rejected_reward", "preference_accuracy",
                "preference_reward_margin", "preference_chosen_logprob", "preference_rejected_logprob"} <= metric.keys()
        assert metric["preference_reward_margin"] == pytest.approx(
            metric["preference_chosen_reward"] - metric["preference_rejected_reward"], abs=1e-5,
        )
    if objective == "reward_model":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(ctx.workdir / "output")
        with torch.inference_mode():
            score = reloaded(**tokenizer("Question Good", return_tensors="pt")).logits
        assert score.shape == (1, 1) and torch.isfinite(score).all()


def test_adapter_disabled_reference_and_checkpoint_resume(tiny_alignment):
    cfg, ctx = tiny_alignment(method="lora", reference="adapter_disabled")
    events = []
    backend = UnslothBackend("transformers")
    backend._train(cfg, ctx, events.append)
    assert (ctx.workdir / "output" / "adapter_config.json").exists()
    assert next(event for event in events if event.type == "artifact").kind == "adapter"
    cfg.train.max_steps = 2
    ctx.resume_from = ctx.checkpoint_dir / "checkpoint-1"
    backend._train(cfg, ctx, events.append)
    assert max(event.step for event in events if event.type == "checkpoint") == 2


def test_graceful_stop_keeps_resume_checkpoint_without_completed_artifact(tiny_alignment):
    cfg, ctx = tiny_alignment(steps=5)
    (ctx.workdir / "STOP").touch()
    events = []
    UnslothBackend("transformers")._train(cfg, ctx, events.append)
    assert (ctx.checkpoint_dir / "checkpoint-1" / "trainer_state.json").exists()
    assert not any(event.type == "artifact" for event in events)


@pytest.mark.parametrize("variant", ["sigmoid", "hinge", "robust", "exo_pair"])
def test_installed_dpo_loss_variants(tiny_alignment, variant):
    cfg, ctx = tiny_alignment()
    cfg.alignment.dpo_loss_variant = variant
    cfg.alignment.label_smoothing = 0.1 if variant in {"robust", "exo_pair"} else 0.0
    events = []
    UnslothBackend("transformers")._train(cfg, ctx, events.append)
    assert any(event.type == "artifact" for event in events)
    assert all(math.isfinite(value) for event in events if event.type == "metric" for value in event.metrics.values())


def test_simpo_loss_matches_reference_free_margin():
    from types import SimpleNamespace

    from trl import CPOTrainer

    from app.train_entry.alignment import PreferenceTrainer

    trainer = object.__new__(CPOTrainer)
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
    trainer.loss_type, trainer.beta, trainer.simpo_gamma, trainer.label_smoothing = "simpo", 0.1, 0.5, 0.0
    chosen, rejected = torch.tensor([-1.0]), torch.tensor([-2.0])
    loss, chosen_reward, rejected_reward = trainer.cpo_loss(chosen, rejected)
    assert loss.item() == pytest.approx(-torch.nn.functional.logsigmoid(torch.tensor(0.1 - 0.5)).item())
    assert chosen_reward.item() - rejected_reward.item() == pytest.approx(0.1)
    cfg = RunConfig(task="alignment", base_model="org/base", dataset_id=1, output_name="test",
                    alignment={"objective": "simpo", "reference": {"strategy": "none"}})
    strategy = PreferenceTrainer(cfg, None, None, load_trainable_model=None, callback_factory=None)
    assert strategy._objective_runtime()[2]["cpo_alpha"] == 0.0


def test_separate_reference_preserves_adapter_and_checks_token_ids(tiny_alignment, tmp_path):
    from transformers import AutoTokenizer

    from app.train_entry.alignment import PreferenceTrainer

    cfg, ctx = tiny_alignment(method="lora", reference="adapter_disabled")
    backend = UnslothBackend("transformers")
    model, tokenizer = backend._load_trainable_model(cfg, lambda event: None)
    adapter = tmp_path / "reference-adapter"
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    cfg.alignment.reference.strategy = "separate_model"
    cfg.alignment.reference.model = str(adapter)
    strategy = PreferenceTrainer(cfg, ctx, None, load_trainable_model=None, callback_factory=None)
    reference, lineage = strategy._reference_model(model, tokenizer)
    assert hasattr(reference, "disable_adapter")
    assert not any(parameter.requires_grad for parameter in reference.parameters())
    assert lineage["adapter"] == str(adapter)
    incompatible = AutoTokenizer.from_pretrained(adapter)
    incompatible.add_tokens(["different-token"])
    with pytest.raises(RuntimeError, match="token IDs differ"):
        strategy._reference_model(model, incompatible)


def test_reward_adapter_can_continue_as_full_model(tiny_alignment, tmp_path):
    cfg, ctx = tiny_alignment("reward_model", method="lora", reference="none")
    backend = UnslothBackend("transformers")
    model, tokenizer = backend._load_reward_model(cfg, lambda event: None)
    adapter = tmp_path / "reward-adapter"
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    cfg.base_model, cfg.method = str(adapter), "full"
    reloaded, _ = backend._load_reward_model(cfg, lambda event: None)
    from peft import PeftModel

    assert not isinstance(reloaded, PeftModel)
    assert all(parameter.requires_grad for parameter in reloaded.parameters())
    assert torch.equal(model.score.modules_to_save["default"].weight, reloaded.score.weight)


def test_alignment_dataset_cache_follows_content_changes(tiny_alignment):
    from app.db.session import engine
    from app.train_entry.alignment import PreferenceTrainer

    cfg, ctx = tiny_alignment()
    backend = UnslothBackend("transformers")
    _, tokenizer = backend._load_trainable_model(cfg, lambda event: None)
    strategy = PreferenceTrainer(cfg, ctx, None, load_trainable_model=None, callback_factory=None)
    first = strategy._dataset(tokenizer)
    with Session(engine) as db:
        dataset = db.get(Dataset, cfg.dataset_id)
    from pathlib import Path

    Path(dataset.path).write_text(json.dumps({"prompt": "Question", "chosen": "Bad Answer", "rejected": "Good Answer"}) + "\n", encoding="utf-8")
    second = strategy._dataset(tokenizer)
    assert second[0]["chosen"] != first[0]["chosen"]
    assert second._fingerprint != first._fingerprint


@pytest.mark.parametrize("method", ["lora", "dora"])
def test_reward_adapter_training_and_registered_scoring(tiny_alignment, method):
    from app.db.models import ModelArtifact
    from app.db.session import engine
    from app.train_entry.model_runtime import load_reward_runtime

    cfg, ctx = tiny_alignment("reward_model", method=method, reference="none")
    events = []
    UnslothBackend("transformers")._train(cfg, ctx, events.append)
    artifact = next(event for event in events if event.type == "artifact")
    assert artifact.kind == "reward_model" and artifact.metadata["format"] == "adapter"
    with Session(engine) as db:
        db.add(ModelArtifact(name="reward", kind="reward_model", local_path=artifact.path, base_model=cfg.base_model))
        db.commit()
    runtime = load_reward_runtime(artifact.path)
    try:
        first, second = runtime.score("Question Good"), runtime.score("Question Good")
        assert math.isfinite(first) and first == pytest.approx(second)
    finally:
        runtime.unload()


def test_app_run_reference_is_resolved_for_policy_and_reference(tiny_alignment, monkeypatch):
    from app import model_refs

    cfg, ctx = tiny_alignment()
    actual = model_refs.resolve_model_ref(cfg.base_model)
    original = model_refs.resolve_model_ref
    monkeypatch.setattr(model_refs, "resolve_model_ref", lambda value: actual if value == "run:7" else original(value))
    monkeypatch.setattr("app.backends.unsloth_backend.resolve_model_ref", model_refs.resolve_model_ref)
    cfg.base_model = "run:7"
    events = []
    UnslothBackend("transformers")._train(cfg, ctx, events.append)
    lineage = next(event for event in events if event.type == "artifact").metadata["reference"]
    assert lineage["requested_model"] == "run:7" and lineage["model"] == actual.load_ref


def test_causal_adapter_cannot_silently_become_reward_adapter(tiny_alignment, tmp_path):
    cfg, _ = tiny_alignment(method="lora", reference="adapter_disabled")
    backend = UnslothBackend("transformers")
    model, tokenizer = backend._load_trainable_model(cfg, lambda event: None)
    adapter = tmp_path / "causal-adapter"
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    cfg, _ = tiny_alignment("reward_model", method="lora", reference="none")
    cfg.base_model = str(adapter)
    with pytest.raises(ValueError, match="SEQ_CLS adapter"):
        backend._load_reward_model(cfg, lambda event: None)
