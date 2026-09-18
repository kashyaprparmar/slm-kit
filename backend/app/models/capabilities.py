"""Metadata-driven model capability resolution.

This module intentionally contains no Transformers/PyTorch imports.  It is safe
for the API process and makes compatibility decisions explicit and testable.
Runtime loaders remain the final authority and may downgrade a capability after
an environment probe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from app.capabilities import (
    Capability,
    ChatTemplateCapabilities,
    EvidenceLevel,
    PeftMethodCapability,
    QuantizationCapability,
    SupportState,
    dependency_statuses,
)


def check_runtime_dependencies() -> dict[str, dict[str, Any]]:
    """Inspect presence and versions of ML dependencies without importing heavy packages."""
    return {
        name: {"installed": status.installed, "version": status.version}
        for name, status in dependency_statuses().items()
    }


class ModelCapabilities(BaseModel):
    family: str
    architecture_kind: str
    model_type: str | None = None
    architectures: list[str] = Field(default_factory=list)
    auto_map: dict[str, Any] = Field(default_factory=dict)
    is_moe: bool = False
    is_multimodal: bool = False
    trust_remote_code: bool = False
    tokenizer_class: str | None = None
    chat_template: bool = False
    context_length: int | None = None
    vocabulary_size: int | None = None
    dependencies: dict[str, Any] = Field(default_factory=dict)
    training: dict[str, Capability]
    backends: dict[str, Capability]
    inference: dict[str, Capability]
    quantization: dict[str, Capability]
    export: dict[str, Capability]
    precision: dict[str, Capability]
    attention: dict[str, Capability] = Field(default_factory=dict)
    gradient_checkpointing: dict[str, Capability] = Field(default_factory=dict)
    rope: dict[str, Capability] = Field(default_factory=dict)
    templates: dict[str, Capability] = Field(default_factory=dict)
    distributed: dict[str, Capability]
    template_capabilities: ChatTemplateCapabilities
    peft_methods: dict[str, PeftMethodCapability]
    quantization_matrix: dict[str, QuantizationCapability]
    suggested_target_modules: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _cap(state: SupportState, reason: str, *requirements: str) -> Capability:
    return Capability(
        state=state,
        reason=reason,
        requirements=list(requirements),
        evidence=EvidenceLevel.METADATA,
    )


@runtime_checkable
class ModelFamilyAdapter(Protocol):
    family: str
    target_modules: tuple[str, ...]
    template_ids: tuple[str, ...]

    def matches(self, model_ref: str, model_type: str, architectures: list[str]) -> bool: ...

    def suggested_target_modules(self) -> list[str]: ...

    def detect(self, model_ref: str, config: dict[str, Any]) -> bool: ...

    def patch_config(self, config: dict[str, Any]) -> dict[str, Any]: ...

    def patch_tokenizer(self, tokenizer_config: dict[str, Any]) -> dict[str, Any]: ...

    def attention_support(self) -> tuple[str, ...]: ...

    def rope_support(self) -> tuple[str, ...]: ...

    def gradient_checkpointing_support(self) -> tuple[str, ...]: ...

    def tool_template_support(self) -> bool: ...

    def reasoning_template_support(self) -> bool: ...

    def multimodal_support(self) -> bool: ...


@dataclass(frozen=True)
class MetadataModelFamilyAdapter:
    family: str
    model_types: frozenset[str]
    architecture_tokens: tuple[str, ...] = ()
    repo_tokens: tuple[str, ...] = ()
    unsloth: SupportState = SupportState.EXPERIMENTAL
    vllm: SupportState = SupportState.SUPPORTED
    gguf: SupportState = SupportState.SUPPORTED
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    )
    template_ids: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)
    supports_rope_scaling: bool = True
    supports_tools: bool = False
    supports_reasoning: bool = False
    multimodal: bool = False

    def matches(self, model_ref: str, model_type: str, architectures: list[str]) -> bool:
        if model_type in self.model_types:
            return True
        architecture = " ".join(architectures).lower()
        if any(token in architecture for token in self.architecture_tokens):
            return True
        # Repository labels are not evidence of an architecture. In particular,
        # an offline/gated metadata failure must not become supported via a name.
        return False

    def suggested_target_modules(self) -> list[str]:
        return list(self.target_modules)

    def detect(self, model_ref: str, config: dict[str, Any]) -> bool:
        architectures = config.get("architectures") or []
        if isinstance(architectures, str):
            architectures = [architectures]
        return self.matches(model_ref, str(config.get("model_type") or "").lower(), list(architectures))

    def patch_config(self, config: dict[str, Any]) -> dict[str, Any]:
        return dict(config)

    def patch_tokenizer(self, tokenizer_config: dict[str, Any]) -> dict[str, Any]:
        return dict(tokenizer_config)

    def attention_support(self) -> tuple[str, ...]:
        return ("auto", "eager", "sdpa", "flash_attention_2")

    def rope_support(self) -> tuple[str, ...]:
        return ("linear", "dynamic", "yarn") if self.supports_rope_scaling else ()

    def gradient_checkpointing_support(self) -> tuple[str, ...]:
        return ("auto", "off", "standard", "non_reentrant", "backend_optimized")

    def tool_template_support(self) -> bool:
        return self.supports_tools

    def reasoning_template_support(self) -> bool:
        return self.supports_reasoning

    def multimodal_support(self) -> bool:
        return self.multimodal


# Compatibility alias for extensions and tests that imported the old name.
FamilyRule = MetadataModelFamilyAdapter


# Ordered: specific/MoE families precede their dense parents.  Adding a family
# is deliberately isolated to one rule instead of conditionals across routes.
FAMILY_RULES: tuple[ModelFamilyAdapter, ...] = (
    FamilyRule("Mixtral", frozenset({"mixtral"}), ("mixtral",), ("mixtral",), unsloth=SupportState.SUPPORTED),
    FamilyRule("Qwen MoE", frozenset({"qwen2_moe", "qwen3_moe"}), ("qwen2moe", "qwen3moe"), unsloth=SupportState.EXPERIMENTAL),
    FamilyRule("Qwen", frozenset({"qwen", "qwen2", "qwen3"}), ("qwen",), ("qwen",), unsloth=SupportState.SUPPORTED),
    FamilyRule("Llama", frozenset({"llama", "smollm", "smollm3", "yi", "baichuan", "internlm", "internlm2"}), ("llama", "smollm", "internlm", "baichuan"), ("llama", "smollm"), unsloth=SupportState.SUPPORTED),
    FamilyRule("Mistral", frozenset({"mistral"}), ("mistral",), ("mistral",), unsloth=SupportState.SUPPORTED),
    FamilyRule("Gemma", frozenset({"gemma", "gemma2", "gemma3", "gemma3_text"}), ("gemma",), ("gemma",), unsloth=SupportState.SUPPORTED),
    FamilyRule("Phi", frozenset({"phi", "phi3", "phi4mm"}), ("phi",), ("phi-",), unsloth=SupportState.SUPPORTED),
    FamilyRule("DeepSeek", frozenset({"deepseek", "deepseek_v2", "deepseek_v3"}), ("deepseek",), ("deepseek",), unsloth=SupportState.EXPERIMENTAL),
    FamilyRule("Falcon", frozenset({"falcon"}), ("falcon",), ("falcon",), vllm=SupportState.EXPERIMENTAL),
    FamilyRule("GPT-NeoX", frozenset({"gpt_neox", "stablelm"}), ("gptneox", "stablelm"), ("stablelm",)),
    FamilyRule("GPT-Neo/J", frozenset({"gpt_neo", "gptj"}), ("gptneo", "gptj")),
    FamilyRule("OPT", frozenset({"opt"}), ("optforcausallm",)),
    FamilyRule("BLOOM", frozenset({"bloom"}), ("bloom",)),
    FamilyRule("MPT", frozenset({"mpt"}), ("mpt",), ("mpt",), vllm=SupportState.EXPERIMENTAL),
)


ENCODER_DECODER_TYPES = frozenset({"t5", "mt5", "bart", "mbart", "pegasus"})
MULTIMODAL_HINTS = ("vision", "conditionalgeneration", "vl", "image", "audio")


def resolve_capabilities(
    model_ref: str,
    config: dict[str, Any] | None,
    tokenizer_config: dict[str, Any] | None = None,
    *,
    kind: str = "transformers",
    dependencies: dict[str, Any] | None = None,
) -> ModelCapabilities:
    cfg = config or {}
    tok = tokenizer_config or {}
    model_type = str(cfg.get("model_type") or "").lower()
    architectures = cfg.get("architectures") or []
    if isinstance(architectures, str):
        architectures = [architectures]
    architectures = [str(value) for value in architectures]
    architecture_text = " ".join(architectures).lower()
    encoder_decoder = bool(cfg.get("is_encoder_decoder")) or model_type in ENCODER_DECODER_TYPES
    multimodal = not encoder_decoder and (
        bool(cfg.get("vision_config") or cfg.get("audio_config"))
        or any(hint in architecture_text or hint in model_type for hint in MULTIMODAL_HINTS)
    )
    moe = "moe" in model_type or "mixtral" in model_type or bool(cfg.get("num_experts") or cfg.get("num_local_experts"))
    rule = next((candidate for candidate in FAMILY_RULES if candidate.matches(model_ref, model_type, architectures)), None)
    noncausal_head = any(token in architecture_text for token in (
        "maskedlm", "sequenceclassification", "tokenclassification", "questionanswering"
    ))
    causal_hint = not noncausal_head and bool(
        kind == "scratch"
        or cfg.get("is_decoder")
        or "causallm" in architecture_text
        or (rule and not encoder_decoder and not multimodal)
    )
    encoder_only = not encoder_decoder and not causal_hint and bool(
        cfg.get("is_encoder")
        or any(token in architecture_text for token in ("maskedlm", "sequenceclassification", "tokenclassification", "encoder"))
        or model_type in {"bert", "roberta", "distilbert", "deberta", "electra"}
    )
    if kind == "scratch":
        architecture_kind = "scratch"
    elif encoder_decoder:
        architecture_kind = "encoder_decoder"
    elif encoder_only:
        architecture_kind = "encoder"
    elif multimodal:
        architecture_kind = "multimodal"
    elif causal_hint:
        architecture_kind = "decoder_only"
    else:
        architecture_kind = "unknown"

    family = rule.family if rule else (model_type.replace("_", " ").title() if model_type else "Unknown")
    causal = architecture_kind in {"decoder_only", "scratch"}
    known = rule is not None or kind == "scratch"
    supported = SupportState.SUPPORTED
    experimental = SupportState.EXPERIMENTAL
    unsupported = SupportState.UNSUPPORTED

    if architecture_kind == "encoder_decoder":
        train_state = unsupported
        train_reason = (
            "Sequence-to-sequence (encoder-decoder) architectures are not supported by the "
            "causal LM training worker. A separate seq2seq loader, trainer, and evaluation "
            "contract is required."
        )
    elif causal and known:
        train_state = supported
        train_reason = f"{family} causal language models are handled by the tested Transformers training path."
    elif causal:
        train_state = experimental
        train_reason = "Metadata describes a causal LM, but this architecture has no tested SLM Kit adapter. A loader probe is required."
    else:
        train_state = unsupported
        train_reason = f"The installed training worker currently supports causal language modelling, not {architecture_kind.replace('_', ' ')} training."

    # An adapter directory does not establish the architecture of its base.
    # Scratch weights also cannot be attached to PEFT/Transformers.
    # Seq2seq models cannot attach causal LM LoRA adapters.
    lora_state = train_state if (kind != "scratch" and architecture_kind != "encoder_decoder") else unsupported
    training = {
        "sft": _cap(train_state, train_reason, "transformers", "trl"),
        "continued_pretraining": _cap(train_state, train_reason, "transformers"),
        "full": _cap(train_state, train_reason, "transformers"),
        "freeze": _cap(train_state, train_reason, "transformers"),
        "lora": _cap(lora_state, train_reason, "peft"),
        "qlora": _cap(lora_state, train_reason, "peft", "bitsandbytes", "CUDA"),
        "dora": _cap(lora_state, train_reason, "peft"),
        "dpo": _cap(train_state, "Shared TRL preference training supports this causal architecture."),
        "ipo": _cap(train_state, "IPO reuses the shared DPO trainer strategy."),
        "orpo": _cap(train_state, "ORPO uses the shared reference-free preference pipeline."),
        "simpo": _cap(experimental if causal else unsupported, "SimPO requires a compatible TRL CPO runtime."),
        "kto": _cap(train_state, "KTO uses its dedicated label semantics in the shared alignment pipeline."),
        "reward_model": _cap(train_state, "Reward modeling uses a scalar sequence-classification head and pairwise preference data."),
        "ppo": _cap(unsupported, "PPO orchestration is not implemented."),
    }
    if kind == "scratch":
        for operation in (
            "sft",
            "continued_pretraining",
            "freeze",
            "lora",
            "qlora",
            "dora",
            "dpo",
            "ipo",
            "orpo",
            "simpo",
            "kto",
            "reward_model",
        ):
            training[operation] = _cap(unsupported, "Scratch GPT currently supports only from-scratch full training with the scratch backend.")
        training["full"] = _cap(supported, "The scratch backend initializes a new GPT; loading saved weights is not full optimizer-state resume.", "torch", "tokenizers")
    transformers_state = train_state if causal and kind != "scratch" else unsupported
    unsloth_state = (rule.unsloth if rule and causal else unsupported)
    backends = {
        "scratch": _cap(supported if kind == "scratch" else unsupported, "SLM Kit GPT from-scratch training only."),
        "transformers": _cap(transformers_state, "Generic AutoModelForCausalLM + Trainer/TRL path." if causal else train_reason),
        "peft": _cap(lora_state, "PEFT requires a compatible causal Transformers base with trainable target layers.", "peft"),
        "unsloth": _cap(unsloth_state, "Optional optimized loader with an automatic Transformers + PEFT fallback." if causal else "Unsloth does not support encoder-decoder or non-causal models.", "unsloth", "CUDA"),
        "trl": _cap(transformers_state, "SFTTrainer is available for causal-LM supervised training."),
        "axolotl": _cap(SupportState.NOT_INSTALLED, "No Axolotl worker profile is installed."),
        "llamafactory": _cap(SupportState.NOT_INSTALLED, "No LLaMA-Factory worker profile is installed."),
        "deepspeed": _cap(SupportState.NOT_INSTALLED, "No DeepSpeed worker profile is installed."),
        "fsdp": _cap(unsupported, "FSDP orchestration is not exposed by the current single-worker scheduler."),
    }
    vllm_state = rule.vllm if rule and causal else unsupported
    gguf_state = rule.gguf if rule and causal else unsupported
    inference = {
        "transformers": _cap(supported if kind == "scratch" else transformers_state, "The managed subprocess uses the native scratch loader for scratch artifacts and AutoModelForCausalLM for HF models."),
        "vllm": _cap(vllm_state, "Uses a separate OpenAI-compatible vLLM service.", "vllm service"),
        "ollama": _cap(SupportState.REQUIRES_CONVERSION if gguf_state == supported else unsupported, "Requires a compatible Ollama model or GGUF conversion."),
        "llama_cpp": _cap(SupportState.REQUIRES_CONVERSION if gguf_state == supported else unsupported, "Requires a GGUF artifact and llama.cpp runtime."),
        "mlx": _cap(SupportState.REQUIRES_CONVERSION if causal else unsupported, "Requires an MLX conversion and Apple Silicon runtime."),
    }
    quantization = {
        "bitsandbytes_4bit": _cap(lora_state, "Supported for QLoRA on a compatible CUDA worker.", "bitsandbytes", "CUDA"),
        "int8": _cap(lora_state, "Supported by bitsandbytes for compatible linear layers.", "bitsandbytes", "CUDA"),
        "gptq": _cap(experimental if causal else unsupported, "Runtime-specific; validate after model export."),
        "awq": _cap(experimental if causal else unsupported, "Runtime-specific; validate after model export."),
        "gguf": _cap(gguf_state, "Conversion is available for compatible llama.cpp architectures.", "llama.cpp converter"),
    }
    export = {
        "adapter": _cap(lora_state, "PEFT adapter directory."),
        "merged_model": _cap(lora_state, "Base model and PEFT adapter can be merged after validation."),
        "safetensors": _cap(transformers_state, "Transformers-compatible model export."),
        "gguf": quantization["gguf"],
    }
    precision = {
        "auto": _cap(transformers_state, "Select BF16, FP16, or FP32 from verified runtime hardware."),
        "fp32": _cap(transformers_state, "Portable CPU/GPU precision."),
        "fp16": _cap(transformers_state, "Supported on compatible CUDA GPUs."),
        "bf16": _cap(transformers_state, "Requires BF16-capable hardware."),
        "fp8": _cap(experimental if causal else unsupported, "Requires recent GPU hardware and a provider-specific runtime."),
    }
    attention = {
        "auto": _cap(transformers_state, "Use the model and Transformers default attention implementation."),
        "eager": _cap(transformers_state, "Portable eager attention implementation."),
        "sdpa": _cap(transformers_state, "Requires a compatible PyTorch scaled-dot-product attention implementation.", "torch>=2"),
        "flash_attention_2": _cap(
            experimental if causal else unsupported,
            "Requires a compatible CUDA model, dtype, and flash-attn runtime.",
            "flash-attn", "CUDA",
        ),
    }
    gradient_checkpointing = {
        "auto": _cap(train_state, "Select the backend's supported checkpointing mode."),
        "off": _cap(train_state, "Gradient checkpointing disabled explicitly."),
        "standard": _cap(train_state, "Use the model's standard gradient checkpointing hook."),
        "non_reentrant": _cap(train_state, "Use PyTorch non-reentrant checkpointing when accepted by the model."),
        "backend_optimized": _cap(
            rule.unsloth if rule and causal else unsupported,
            "Uses an optimized backend-specific checkpointing implementation.",
            "unsloth",
        ),
    }
    rope = {
        "none": _cap(supported, "Keep the model's configured context and RoPE behavior."),
        "linear": _cap(
            train_state if rule and rule.supports_rope_scaling else unsupported,
            "Linear RoPE scaling must be requested explicitly and verified by the loader.",
        ),
        "dynamic": _cap(
            train_state if rule and rule.supports_rope_scaling else unsupported,
            "Dynamic RoPE scaling must be requested explicitly and verified by the loader.",
        ),
        "yarn": _cap(experimental if rule and rule.supports_rope_scaling else unsupported,
                     "YaRN support depends on the concrete model configuration and Transformers version."),
    }
    templates = {
        "tools": _cap(
            supported if rule and rule.supports_tools and tok.get("chat_template") else unsupported,
            "Tool templates require explicit family and tokenizer-template support.",
        ),
        "reasoning": _cap(
            supported if rule and rule.supports_reasoning and tok.get("chat_template") else unsupported,
            "Reasoning templates require explicit family and tokenizer-template support.",
        ),
        "multimodal": _cap(
            experimental if multimodal else unsupported,
            "Multimodal metadata is detected, but the training processor remains unsupported.",
        ),
    }
    distributed = {
        "multi_gpu": _cap(experimental if causal else unsupported, "Configuration is represented, but the local scheduler currently launches one worker."),
        "tensor_parallel": _cap(vllm_state, "Available through a separately configured vLLM service."),
        "pipeline_parallel": _cap(unsupported, "Not implemented by the local training scheduler."),
    }
    warnings = list(rule.notes if rule else ())
    if not known and causal:
        warnings.append("Unknown causal architecture: compatibility is experimental until a runtime loader probe succeeds.")
    if not causal:
        warnings.append(train_reason)
    if multimodal:
        warnings.append("Multimodal inputs are not implemented in the current dataset and training pipeline.")

    template_capabilities = ChatTemplateCapabilities(
        native=_cap(
            supported if tok.get("chat_template") else unsupported,
            (
                "The tokenizer metadata provides a native chat template."
                if tok.get("chat_template")
                else "The tokenizer metadata does not provide a native chat template."
            ),
        ),
        explicit_override=_cap(
            supported,
            "RunConfig may provide an explicit chat template; the worker never guesses one silently.",
        ),
        fallback=_cap(
            unsupported,
            "Automatic model-family template fallback is not implemented.",
        ),
    )
    peft_methods = {
        method: PeftMethodCapability(
            method=method,
            support=training[method],
            requires_quantized_base=method == "qlora",
        )
        for method in ("lora", "qlora", "dora")
    }
    quantization_matrix = {
        name: QuantizationCapability(
            format=name,
            operations={
                "bitsandbytes_4bit": ["train", "load"],
                "int8": ["load"],
                "gptq": ["load", "serve"],
                "awq": ["load", "serve"],
                "gguf": ["export", "serve"],
            }[name],
            support=capability,
        )
        for name, capability in quantization.items()
    }

    from app.models.quantization_registry import quantization_capabilities
    # Preserve legacy names while using one operation-specific source for new consumers.
    quantization_matrix.update(quantization_capabilities())
    context = next((cfg.get(key) for key in ("max_position_embeddings", "n_positions", "seq_length", "model_max_length") if cfg.get(key)), tok.get("model_max_length"))
    try:
        context = int(context) if context is not None and int(context) < 10**9 else None
    except (TypeError, ValueError):
        context = None
    auto_map = cfg.get("auto_map") if isinstance(cfg.get("auto_map"), dict) else {}
    deps = dependencies if dependencies is not None else check_runtime_dependencies()
    return ModelCapabilities(
        family=family,
        architecture_kind=architecture_kind,
        model_type=model_type or None,
        architectures=architectures,
        auto_map=auto_map,
        is_moe=moe,
        is_multimodal=multimodal,
        trust_remote_code=bool(auto_map),
        tokenizer_class=tok.get("tokenizer_class"),
        chat_template=bool(tok.get("chat_template")),
        context_length=context,
        vocabulary_size=cfg.get("vocab_size"),
        dependencies=deps,
        training=training,
        backends=backends,
        inference=inference,
        quantization=quantization,
        export=export,
        precision=precision,
        attention=attention,
        gradient_checkpointing=gradient_checkpointing,
        rope=rope,
        templates=templates,
        distributed=distributed,
        template_capabilities=template_capabilities,
        peft_methods=peft_methods,
        quantization_matrix=quantization_matrix,
        suggested_target_modules=(
            rule.suggested_target_modules() if rule and causal else ["all-linear"] if causal else []
        ),
        warnings=warnings,
    )


def is_allowed(capability: Capability) -> bool:
    return capability.allowed
