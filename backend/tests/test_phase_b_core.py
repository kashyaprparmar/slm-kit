from __future__ import annotations

import pytest

from app.backends.scratch_backend import _GPT, ScratchBackend
from app.backends.unsloth_backend import UnslothBackend, _continued_pretraining_text
from app.domain import (
    FreezeParams,
    HardwareProfile,
    LoraParams,
    Method,
    RunConfig,
    ScratchArch,
    TaskType,
)
from app.integrations.estimator import ModelSpec, estimate
from app.models.capabilities import FAMILY_RULES, resolve_capabilities
from app.train_entry.module_selection import (
    ModuleSelectionError,
    apply_freeze_policy,
    discover_lora_targets,
    parameter_summary,
)


class Parameter:
    def __init__(self, size: int):
        self.size = size
        self.requires_grad = True

    def numel(self):
        return self.size

    def requires_grad_(self, enabled):
        self.requires_grad = enabled
        return self


class Linear:
    in_features = 8
    out_features = 8


class LayerNorm:
    pass


class FakeModel:
    def __init__(self):
        names = [
            "transformer.h.0.self_attention.query_key_value.weight",
            "transformer.h.0.mlp.dense_h_to_4h.weight",
            "transformer.h.0.input_layernorm.weight",
            "transformer.h.1.self_attention.query_key_value.weight",
            "transformer.h.1.mlp.dense_h_to_4h.weight",
            "transformer.h.1.input_layernorm.weight",
            "transformer.word_embeddings.weight",
            "lm_head.weight",
        ]
        self._parameters = {name: Parameter(10) for name in names}
        self._modules = {
            name.rsplit(".", 1)[0]: (LayerNorm() if "layernorm" in name else Linear())
            for name in names
        }

    def named_modules(self):
        return list(self._modules.items())

    def named_parameters(self):
        return list(self._parameters.items())

    def parameters(self):
        return list(self._parameters.values())


def _config(**overrides):
    values = {
        "task": TaskType.FINETUNE,
        "method": Method.LORA,
        "backend": "transformers",
        "base_model": "model",
        "dataset_id": 1,
        "output_name": "run",
    }
    values.update(overrides)
    return RunConfig(**values)


def test_run_config_normalizes_stage_defaults_once():
    continued = _config(task=TaskType.CONTINUED_PRETRAIN)
    assert continued.train.packing is True
    assert continued.load_in_4bit is False
    qlora = _config(method=Method.QLORA)
    assert qlora.load_in_4bit is True
    assert qlora.quantization.mode == "nf4"
    assert qlora.training_stage.value == "supervised_fine_tuning"


def test_int8_rejects_four_bit_only_quantization_options():
    with pytest.raises(ValueError, match="Nested/double quantization"):
        _config(quantization={"mode": "int8"})
    valid = _config(quantization={"mode": "int8", "double_quant": False})
    assert valid.quantization.compute_dtype == "auto"


def test_lora_discovery_uses_actual_non_llama_module_names():
    model = FakeModel()
    attention = discover_lora_targets(model, LoraParams(target_strategy="attention"))
    assert attention.matched_modules == (
        "transformer.h.0.self_attention.query_key_value",
        "transformer.h.1.self_attention.query_key_value",
    )
    mlp = discover_lora_targets(model, LoraParams(target_strategy="mlp"))
    assert all("mlp" in name for name in mlp.matched_modules)
    automatic = discover_lora_targets(
        model,
        LoraParams(target_strategy="auto"),
        suggested_modules=["q_proj"],
    )
    assert "transformer.h.0.self_attention.query_key_value" in automatic.matched_modules


def test_custom_lora_discovery_rejects_missing_modules():
    with pytest.raises(ModuleSelectionError, match="not found"):
        discover_lora_targets(
            FakeModel(),
            LoraParams(target_strategy="custom", target_modules=["q_proj"]),
        )


def test_freeze_policy_selects_last_layer_and_explicit_groups():
    model = FakeModel()
    summary, selected = apply_freeze_policy(
        model,
        FreezeParams(last_n_layers=1, train_lm_head=True, train_norms=False),
    )
    assert any("h.1" in name for name in selected)
    assert not any("h.0" in name for name in selected)
    assert "lm_head.weight" in selected
    assert summary.total_parameters == 80
    assert summary.trainable_parameters == 40
    assert summary.frozen_parameters == 40
    assert parameter_summary(model) == summary


def test_freeze_policy_rejects_unknown_custom_module():
    with pytest.raises(ModuleSelectionError, match="not found"):
        apply_freeze_policy(
            FakeModel(),
            FreezeParams(last_n_layers=0, train_lm_head=False, selected_modules=["missing"]),
        )


def test_estimator_reports_parameter_and_optimizer_breakdown_for_freeze():
    cfg = _config(method=Method.FREEZE, freeze={"last_n_layers": 2, "train_lm_head": False})
    result = estimate(cfg, HardwareProfile(), ModelSpec(1_000_000, hidden_size=100, num_layers=10))
    assert result.total_parameters == 1_000_000
    assert 0 < result.trainable_parameters < result.total_parameters
    assert result.frozen_parameters == result.total_parameters - result.trainable_parameters
    assert result.trainable_percentage == pytest.approx(20.0)
    assert result.optimizer_mb > 0


def test_capabilities_include_stages_freeze_runtime_and_guarded_peft_features():
    native = UnslothBackend(name="transformers").capabilities()
    assert native.stages["continued_pretraining"].allowed
    assert native.methods["freeze"].allowed
    assert "flash_attention_2" in native.attention
    assert native.optional_features["oft"].allowed is False
    scratch = ScratchBackend().capabilities()
    assert scratch.stages["from_scratch_pretraining"].allowed
    assert not scratch.stages["alignment"].allowed


def test_model_family_adapter_hooks_and_runtime_capabilities_are_structured():
    adapter = FAMILY_RULES[0]
    config = {"model_type": next(iter(adapter.model_types)), "architectures": []}
    assert adapter.detect("repo", config)
    assert adapter.patch_config(config) == config
    caps = resolve_capabilities("repo", config, {})
    assert {"auto", "bf16", "fp16", "fp32"} <= set(caps.precision)
    assert {"auto", "sdpa", "flash_attention_2", "eager"} <= set(caps.attention)
    assert "backend_optimized" in caps.gradient_checkpointing
    assert "linear" in caps.rope


def test_continued_pretraining_accepts_only_raw_text_without_chat_formatting():
    assert _continued_pretraining_text({"text": "domain text"}, 4, "<eos>") == "domain text<eos>"
    with pytest.raises(ValueError, match="canonical raw text"):
        _continued_pretraining_text(
            {"instruction": "Question", "output": "Answer"},
            5,
            "<eos>",
        )


def test_from_scratch_tiny_model_builds_and_computes_causal_loss():
    torch = pytest.importorskip("torch")
    arch = ScratchArch(n_layers=1, n_heads=2, n_embd=32, block_size=16, vocab_size=256, dropout=0)
    model = _GPT(arch, vocab_size=256)
    tokens = torch.randint(0, 256, (2, 16))
    logits, loss = model(tokens, torch.roll(tokens, shifts=-1, dims=1))
    assert logits.shape == (2, 16, 256)
    assert loss is not None and torch.isfinite(loss)
    before = model.head.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, model.head.weight)
