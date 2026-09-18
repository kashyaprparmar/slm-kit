"""Shared domain types: enums, run configuration, and hardware/fit models.

These Pydantic models are the *contract* between the API, the job queue, the
training subprocess, and the frontend. A ``RunConfig`` is serialized to JSON,
stored on the ``Run`` row, and handed verbatim to the training subprocess so any
run is fully reproducible and cloneable.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class TaskType(str, Enum):
    PRETRAIN = "pretrain"                    # Pillar 1: from-scratch
    CONTINUED_PRETRAIN = "continued_pretrain"  # Pillar 2: domain adaptation
    FINETUNE = "finetune"                    # Pillar 3: instruction fine-tuning
    ALIGNMENT = "alignment"                  # Preference/alignment objectives


class TrainingStage(str, Enum):
    """Stable execution stages shared by backends and dataset adapters.

    ``TaskType`` remains the persisted/API vocabulary.  This separate stage
    model lets later alignment work extend execution without changing old run
    records or overloading the existing task enum.
    """

    FROM_SCRATCH_PRETRAINING = "from_scratch_pretraining"
    CONTINUED_PRETRAINING = "continued_pretraining"
    SUPERVISED_FINE_TUNING = "supervised_fine_tuning"
    ALIGNMENT = "alignment"


class Method(str, Enum):
    FREEZE = "freeze"
    LORA = "lora"
    QLORA = "qlora"
    DORA = "dora"
    FULL = "full"
    PROMPT_TUNING = "prompt_tuning"


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DatasetKind(str, Enum):
    PRETRAIN_CORPUS = "pretrain_corpus"      # raw text for from-scratch
    DOMAIN_CORPUS = "domain_corpus"          # raw text for continued pretraining
    INSTRUCTION = "instruction"              # chat/instruction pairs
    EVAL = "eval"                            # held-out evaluation set
    PREFERENCE = "preference"                # chosen/rejected preference pairs
    KTO = "kto"                              # desirable/undesirable responses


class Quantization(str, Enum):
    FP16 = "fp16"       # / bf16 full precision base
    INT8 = "int8"
    NF4 = "nf4"         # 4-bit (QLoRA / GGUF Q4-ish planning)


class ArtifactKind(str, Enum):
    CAUSAL_LM = "causal_lm"
    ADAPTER = "adapter"
    MERGED_MODEL = "merged_model"
    REWARD_MODEL = "reward_model"
    REFERENCE_MODEL = "reference_model"
    QUANTIZED_MODEL = "quantized_model"
    # Historical values remain valid for stored rows and API clients.
    PRETRAIN = "pretrain"
    DOMAIN = "domain"
    FINETUNE = "finetune"
    GGUF = "gguf"


# --------------------------------------------------------------------------- #
# Hardware & memory-fit
# --------------------------------------------------------------------------- #
class GPUDeviceProfile(BaseModel):
    id: int
    uuid: str | None = None
    name: str
    vram_total_mb: int
    vram_free_mb: int
    utilization_pct: float | None = None
    compute_capability: str | None = None
    temperature_c: float | None = None
    power_watts: float | None = None
    bf16_supported: bool = False
    fp16_supported: bool = True
    fp8_supported: bool = False
    flash_attention_feasible: bool = False


class HardwareProfile(BaseModel):
    gpu_name: str | None = None
    vram_total_mb: int | None = None
    vram_free_mb: int | None = None
    gpu_util_pct: float | None = None
    ram_total_mb: int | None = None
    ram_free_mb: int | None = None
    cpu_count: int | None = None
    cpu_util_pct: float | None = None
    disk_free_mb: int | None = None
    disk_total_mb: int | None = None
    gpus: list[GPUDeviceProfile] = Field(default_factory=list)
    gpu_count: int = 0
    cuda_available: bool = False
    cuda_runtime_version: str | None = None
    nvidia_driver_version: str | None = None
    mps_available: bool = False
    platform: str | None = None
    source: str = "unknown"  # "pynvml" | "llmfit" | "fallback"


class FitLevel(str, Enum):
    FITS = "fits"
    TIGHT = "tight"
    WONT_FIT = "wont_fit"


class MemoryEstimate(BaseModel):
    """Breakdown of predicted peak VRAM for a run, in MB."""

    weights_mb: float = 0.0
    optimizer_mb: float = 0.0
    reference_model_mb: float = 0.0
    gradients_mb: float = 0.0
    adapters_mb: float = 0.0
    activations_mb: float = 0.0
    kv_cache_mb: float = 0.0
    overhead_mb: float = 0.0
    total_mb: float = 0.0
    budget_mb: float = 0.0
    available_mb: float | None = None
    safe_budget_mb: float = 0.0
    headroom_mb: float = 0.0
    verdict: str = "Should Fit"
    suggestions: list[str] = Field(default_factory=list)
    fit: FitLevel = FitLevel.FITS
    source: str = "fallback"        # "llmfit" | "fallback"
    notes: list[str] = Field(default_factory=list)
    total_parameters: int | None = None
    trainable_parameters: int | None = None
    frozen_parameters: int | None = None
    trainable_percentage: float | None = None


class ValidationIssue(BaseModel):
    level: Literal["error", "warning", "info"]
    message: str
    line: int | None = None


class ValidationReport(BaseModel):
    ok: bool = True
    issues: list[ValidationIssue] = Field(default_factory=list)

    def error(self, msg: str, line: int | None = None) -> None:
        self.ok = False
        self.issues.append(ValidationIssue(level="error", message=msg, line=line))

    def warn(self, msg: str, line: int | None = None) -> None:
        self.issues.append(ValidationIssue(level="warning", message=msg, line=line))

    def info(self, msg: str, line: int | None = None) -> None:
        self.issues.append(ValidationIssue(level="info", message=msg, line=line))


# --------------------------------------------------------------------------- #
# Run configuration (config-as-data)
# --------------------------------------------------------------------------- #
class LoraParams(BaseModel):
    r: int = Field(default=16, ge=1, le=512)
    alpha: int = Field(default=16, ge=1, le=2048)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    target_modules: list[str] = Field(default_factory=list)  # automatic all-linear targeting
    use_rslora: bool = False
    use_dora: bool = False  # flipped on automatically when method == DORA
    target_strategy: Literal["auto", "all_linear", "attention", "mlp", "custom"] = "auto"
    init_method: Literal["standard", "pissa", "loftq", "eva"] = "standard"
    lora_plus_lr_ratio: float | None = Field(default=None, gt=0)


class FreezeParams(BaseModel):
    """Explicit parameter groups for partial/freeze tuning."""

    last_n_layers: int = Field(default=1, ge=0, le=1024)
    train_embeddings: bool = False
    train_lm_head: bool = True
    train_norms: bool = False
    selected_modules: list[str] = Field(default_factory=list)


class QuantizationConfig(BaseModel):
    mode: Literal["none", "nf4", "fp4", "int8"] = "none"
    compute_dtype: Literal["auto", "bf16", "fp16", "fp32"] = "auto"
    double_quant: bool = True
    storage_dtype: Literal["auto", "uint8", "bf16", "fp16", "fp32"] = "auto"


class RopeConfig(BaseModel):
    enabled: bool = False
    factor: float = Field(default=1.0, ge=1.0, le=64.0)
    type: Literal["linear", "dynamic", "yarn"] = "linear"

    @model_validator(mode="after")
    def require_extension_factor_when_enabled(self) -> RopeConfig:
        if self.enabled and self.factor <= 1:
            raise ValueError("Enabled RoPE extension requires factor greater than 1.")
        return self


class RuntimeConfig(BaseModel):
    precision: Literal["auto", "bf16", "fp16", "fp32"] = "auto"
    attention: Literal["auto", "sdpa", "flash_attention_2", "eager"] = "auto"
    gradient_checkpointing: Literal[
        "auto", "off", "standard", "non_reentrant", "backend_optimized"
    ] = "auto"
    use_liger: bool = False
    neftune_noise_alpha: float | None = Field(default=None, gt=0, le=100)
    rope: RopeConfig = Field(default_factory=RopeConfig)


class ReferenceModelConfig(BaseModel):
    strategy: Literal["base_model", "separate_model", "adapter_disabled", "none"] = "base_model"
    model: str | None = None
    revision: str | None = None

    @model_validator(mode="after")
    def validate_strategy(self) -> ReferenceModelConfig:
        if self.strategy == "separate_model" and not self.model:
            raise ValueError("A separate reference strategy requires reference.model.")
        if self.strategy != "separate_model" and self.model:
            raise ValueError("reference.model is valid only with the separate_model strategy.")
        if self.strategy != "separate_model" and self.revision:
            raise ValueError("reference.revision is valid only with the separate_model strategy; base references use the run revision.")
        return self


class AlignmentConfig(BaseModel):
    objective: Literal["dpo", "ipo", "orpo", "simpo", "kto", "reward_model"] = "dpo"
    beta: float = Field(default=0.1, gt=0, le=100)
    dpo_loss_variant: Literal["sigmoid", "hinge", "robust", "exo_pair"] = "sigmoid"
    label_smoothing: float = Field(default=0.0, ge=0, lt=0.5)
    simpo_gamma: float = Field(default=0.5, ge=0)
    desirable_weight: float = Field(default=1.0, gt=0)
    undesirable_weight: float = Field(default=1.0, gt=0)
    reference: ReferenceModelConfig = Field(default_factory=ReferenceModelConfig)

    @model_validator(mode="after")
    def validate_objective(self) -> AlignmentConfig:
        if self.objective in {"orpo", "simpo", "reward_model"} and self.reference.strategy != "none":
            raise ValueError(f"{self.objective.upper()} is reference-free; set reference.strategy to 'none'.")
        if self.objective in {"dpo", "ipo", "kto"} and self.reference.strategy == "none":
            raise ValueError(f"{self.objective.upper()} requires an explicit reference strategy.")
        if self.objective == "ipo" and self.label_smoothing:
            raise ValueError("IPO does not support label smoothing in the shared DPO trainer strategy.")
        if self.objective != "simpo" and self.simpo_gamma != 0.5:
            raise ValueError("simpo_gamma is valid only for the SimPO objective.")
        if self.objective != "dpo" and self.dpo_loss_variant != "sigmoid":
            raise ValueError("dpo_loss_variant is valid only for the DPO objective.")
        if self.objective != "dpo" and self.label_smoothing:
            raise ValueError("label_smoothing is valid only for the DPO objective.")
        if self.objective == "dpo" and self.label_smoothing and self.dpo_loss_variant == "hinge":
            raise ValueError("DPO hinge loss does not support label smoothing.")
        return self


class OptimConfig(BaseModel):
    learning_rate: float = Field(default=2e-4, gt=0, le=1.0)
    lr_scheduler: str = "cosine"          # cosine | linear | constant
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=1.0)
    weight_decay: float = Field(default=0.01, ge=0.0, le=1.0)
    optimizer: str = "adamw_8bit"         # 8-bit Adam is the 8GB-safe default
    max_grad_norm: float = Field(default=1.0, gt=0)
    strategy: Literal["default", "galore", "apollo", "badam", "adam_mini", "muon"] = "default"
    target_modules: list[str] = Field(default_factory=list)
    low_rank_rank: int = Field(default=128, ge=1, le=65_536)
    update_interval: int = Field(default=200, ge=1, le=100_000)


class TrainConfig(BaseModel):
    epochs: float = Field(default=1.0, gt=0, le=1_000)
    max_steps: int | None = Field(default=None, ge=1)  # if set, overrides epochs
    per_device_batch_size: int = Field(default=2, ge=1, le=1_024)
    gradient_accumulation: int = Field(default=4, ge=1, le=65_536)
    max_seq_length: int = Field(default=1024, ge=16, le=1_048_576)
    packing: bool | None = None
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    save_steps: int = Field(default=100, ge=1)
    logging_steps: int = Field(default=5, ge=1)
    eval_on_completion: bool = True


class TokenizerConfig(BaseModel):
    """Versioned tokenizer/rendering policy shared by preparation and workers."""

    mode: Literal["reuse", "extend", "train", "import"] = "reuse"
    source: str | None = None
    chat_template: str | None = None
    loss_policy: Literal["full_sequence", "completion_only", "assistant_only"] = "full_sequence"
    added_tokens: list[str] = Field(default_factory=list)
    added_special_tokens: list[str] = Field(default_factory=list)


class ScratchArch(BaseModel):
    """Architecture config for the from-scratch pretraining pillar."""

    vocab_size: int = Field(default=8192, ge=256, le=1_000_000)
    n_layers: int = Field(default=6, ge=1, le=256)
    n_heads: int = Field(default=6, ge=1, le=256)
    n_embd: int = Field(default=384, ge=32, le=65_536)
    block_size: int = Field(default=256, ge=16, le=1_048_576)  # context length
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    tokenizer_name: str = "byte_bpe"      # custom tokenizer trained on the corpus


class TrainingOperation(BaseModel):
    """Normalized operation identity used for backend selection and validation."""

    backend: str
    task: TaskType
    stage: TrainingStage
    method: Method


class RunConfig(BaseModel):
    """The complete, reproducible description of a training run."""

    schema_version: Literal[1] = 1
    backend: str = "unsloth"              # registry key of the TrainingBackend
    task: TaskType
    method: Method = Method.QLORA
    base_model: str = ""                  # HF repo id or local checkpoint dir
    revision: str | None = None
    dataset_id: int | None = Field(default=None, ge=1)
    output_name: str = Field(min_length=1, max_length=160)

    load_in_4bit: bool = True             # QLoRA base quantization
    gradient_checkpointing: bool = True
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    lora: LoraParams = Field(default_factory=LoraParams)
    freeze: FreezeParams = Field(default_factory=FreezeParams)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    alignment: AlignmentConfig = Field(default_factory=AlignmentConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)

    # Only used when task == PRETRAIN.
    arch: ScratchArch | None = None

    # Escape hatch for backend-specific knobs without schema churn.
    extra: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def resolve_compatible_defaults(self) -> RunConfig:
        """Normalize defaults once for API, workers, exports, and future CLI callers."""
        if self.train.packing is None:
            self.train.packing = self.task in {TaskType.PRETRAIN, TaskType.CONTINUED_PRETRAIN}
        if self.method == Method.QLORA:
            if self.quantization.mode == "none":
                self.quantization.mode = "nf4"
        self.load_in_4bit = self.quantization.mode in {"nf4", "fp4"}
        if self.method == Method.DORA:
            self.lora.use_dora = True
        if self.task == TaskType.ALIGNMENT:
            if self.method not in {Method.FULL, Method.LORA, Method.QLORA, Method.DORA}:
                raise ValueError("Alignment supports full, LoRA, QLoRA, or DoRA tuning.")
            if self.alignment.reference.strategy == "adapter_disabled" and self.method not in {
                Method.LORA, Method.QLORA, Method.DORA
            }:
                raise ValueError("adapter_disabled reference handling requires an adapter-based method.")
        if self.quantization.mode == "int8":
            if self.quantization.compute_dtype != "auto":
                raise ValueError("8-bit loading does not accept a 4-bit compute dtype override.")
            if self.quantization.storage_dtype != "auto":
                raise ValueError("8-bit loading does not accept a 4-bit storage dtype override.")
            if self.quantization.double_quant:
                raise ValueError("Nested/double quantization is available only for NF4 or FP4.")
        if self.runtime.gradient_checkpointing == "off":
            self.gradient_checkpointing = False
        elif self.runtime.gradient_checkpointing != "auto":
            self.gradient_checkpointing = True
        return self

    @property
    def training_stage(self) -> TrainingStage:
        return training_stage_for_task(self.task)

    @property
    def operation(self) -> TrainingOperation:
        return TrainingOperation(
            backend=self.backend,
            task=self.task,
            stage=self.training_stage,
            method=self.method,
        )


def training_stage_for_task(task: TaskType) -> TrainingStage:
    """Translate the persisted task contract to the execution-stage contract."""
    return {
        TaskType.PRETRAIN: TrainingStage.FROM_SCRATCH_PRETRAINING,
        TaskType.CONTINUED_PRETRAIN: TrainingStage.CONTINUED_PRETRAINING,
        TaskType.FINETUNE: TrainingStage.SUPERVISED_FINE_TUNING,
        TaskType.ALIGNMENT: TrainingStage.ALIGNMENT,
    }[task]


class ExportedConfig(BaseModel):
    """A run config re-expressed in another engine's native format."""

    format: str            # "axolotl_yaml" | "llamafactory_yaml" | "json"
    filename: str
    content: str
