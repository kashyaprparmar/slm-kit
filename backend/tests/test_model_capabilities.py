import pytest

from app.integrations import hf_hub
from app.models.capabilities import SupportState, resolve_capabilities


@pytest.mark.parametrize(
    ("model_type", "architecture", "family"),
    [
        ("llama", "LlamaForCausalLM", "Llama"),
        ("qwen2", "Qwen2ForCausalLM", "Qwen"),
        ("mistral", "MistralForCausalLM", "Mistral"),
        ("gemma2", "Gemma2ForCausalLM", "Gemma"),
        ("phi3", "Phi3ForCausalLM", "Phi"),
    ],
)
def test_known_decoder_families(model_type, architecture, family):
    caps = resolve_capabilities(
        "org/model",
        {"model_type": model_type, "architectures": [architecture], "max_position_embeddings": 8192},
        {"tokenizer_class": "AutoTokenizer", "chat_template": "{{ messages }}"},
    )
    assert caps.family == family
    assert caps.architecture_kind == "decoder_only"
    assert caps.training["qlora"].state == SupportState.SUPPORTED
    assert caps.inference["transformers"].state == SupportState.SUPPORTED
    assert caps.context_length == 8192
    assert caps.chat_template is True


def test_encoder_decoder_is_truthfully_rejected_by_current_worker():
    caps = resolve_capabilities(
        "google/flan-t5-small",
        {"model_type": "t5", "architectures": ["T5ForConditionalGeneration"], "is_encoder_decoder": True},
    )
    assert caps.architecture_kind == "encoder_decoder"
    assert caps.training["sft"].state == SupportState.UNSUPPORTED
    assert caps.inference["transformers"].state == SupportState.UNSUPPORTED
    assert caps.is_multimodal is False


def test_unknown_causal_architecture_is_experimental_not_claimed_supported():
    caps = resolve_capabilities(
        "vendor/new-model",
        {"model_type": "brand_new", "architectures": ["BrandNewForCausalLM"]},
    )
    assert caps.family == "Brand New"
    assert caps.training["sft"].state == SupportState.EXPERIMENTAL
    assert caps.backends["transformers"].state == SupportState.EXPERIMENTAL
    assert caps.inference["transformers"].state == SupportState.EXPERIMENTAL
    assert caps.backends["unsloth"].state == SupportState.UNSUPPORTED
    assert caps.warnings


def test_encoder_only_model_is_not_misclassified_as_causal():
    caps = resolve_capabilities(
        "google-bert/bert-base-uncased",
        {"model_type": "bert", "architectures": ["BertForMaskedLM"]},
    )
    assert caps.architecture_kind == "encoder"
    assert caps.training["sft"].state == SupportState.UNSUPPORTED


def test_moe_detection_is_metadata_driven():
    caps = resolve_capabilities(
        "org/not-named-like-a-family",
        {"model_type": "qwen3_moe", "architectures": ["Qwen3MoeForCausalLM"], "num_experts": 128},
    )
    assert caps.family == "Qwen MoE"
    assert caps.is_moe is True


def test_capability_cache_is_scoped_to_model_revision(monkeypatch):
    calls = []

    def load_config(model_ref, revision=None):
        calls.append((model_ref, revision))
        return {"model_type": "llama", "architectures": ["LlamaForCausalLM"]}

    hf_hub.get_model_capabilities.cache_clear()
    monkeypatch.setattr(hf_hub, "_load_config", load_config)
    hf_hub.get_model_capabilities("org/model", "revision-a")
    hf_hub.get_model_capabilities("org/model", "revision-b")
    hf_hub.get_model_capabilities("org/model", "revision-a")
    assert calls == [("org/model", "revision-a"), ("org/model", "revision-b")]
    hf_hub.get_model_capabilities.cache_clear()


@pytest.mark.parametrize("name", ["org/llama-7b", "org/qwen-finetune", "org/smollm"])
def test_repository_name_does_not_establish_architecture(name):
    caps = resolve_capabilities(name, None)
    assert caps.architecture_kind == "unknown"
    assert caps.training["sft"].state == SupportState.UNSUPPORTED


def test_adapter_does_not_bypass_base_architecture_constraints():
    caps = resolve_capabilities("org/adapter", {"model_type": "bert", "architectures": ["BertForMaskedLM"]}, kind="adapter")
    assert caps.training["lora"].state == SupportState.UNSUPPORTED
    assert caps.backends["peft"].state == SupportState.UNSUPPORTED


def test_scratch_capabilities_match_native_worker():
    caps = resolve_capabilities("local/scratch", {}, kind="scratch")
    assert caps.backends["scratch"].state == SupportState.SUPPORTED
    assert caps.training["full"].state == SupportState.SUPPORTED
    assert caps.training["sft"].state == SupportState.UNSUPPORTED
    assert caps.training["qlora"].state == SupportState.UNSUPPORTED
    assert caps.backends["transformers"].state == SupportState.UNSUPPORTED
    assert caps.export["safetensors"].state == SupportState.UNSUPPORTED
    assert caps.inference["transformers"].state == SupportState.SUPPORTED


def test_classification_head_is_not_causal_even_for_known_family():
    caps = resolve_capabilities("org/llama", {"model_type": "llama", "architectures": ["LlamaForSequenceClassification"]})
    assert caps.training["sft"].state == SupportState.UNSUPPORTED


def test_encoder_decoder_all_operations_strictly_unsupported():
    caps = resolve_capabilities(
        "google/flan-t5-base",
        {"model_type": "t5", "architectures": ["T5ForConditionalGeneration"], "is_encoder_decoder": True},
    )
    for op in ("sft", "continued_pretraining", "full", "lora", "qlora", "dora"):
        assert caps.training[op].state == SupportState.UNSUPPORTED
        assert "Sequence-to-sequence" in caps.training[op].reason
    assert caps.backends["transformers"].state == SupportState.UNSUPPORTED
    assert caps.backends["unsloth"].state == SupportState.UNSUPPORTED


def test_capabilities_include_dependency_evidence():
    caps = resolve_capabilities("org/model", {"model_type": "llama", "architectures": ["LlamaForCausalLM"]})
    assert isinstance(caps.dependencies, dict)
    assert "transformers" in caps.dependencies
    assert "torch" in caps.dependencies
    assert "installed" in caps.dependencies["transformers"]


def test_local_model_inspection_fingerprints_and_commit(tmp_path):
    import json
    model_dir = tmp_path / "tiny-model"
    model_dir.mkdir()
    cfg = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "vocab_size": 32000}
    (model_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    tok_cfg = {"tokenizer_class": "LlamaTokenizer", "model_max_length": 2048}
    (model_dir / "tokenizer_config.json").write_text(json.dumps(tok_cfg), encoding="utf-8")

    info = hf_hub.inspect_model(str(model_dir))
    assert info["reachable"] is True
    assert info["config_fingerprint"] is not None
    assert len(info["config_fingerprint"]) == 64
    assert info["tokenizer_fingerprint"] is not None
    assert len(info["tokenizer_fingerprint"]) == 64
    assert info["dependencies"] is not None
