"""Shared domain types: enums, run configuration, and hardware/fit models.

These Pydantic models are the *contract* between the API, the job queue, the
training subprocess, and the frontend. A ``RunConfig`` is serialized to JSON,
stored on the ``Run`` row, and handed verbatim to the training subprocess so any
run is fully reproducible and cloneable.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class TaskType(str, Enum):
    PRETRAIN = "pretrain"                    # Pillar 1: from-scratch
    CONTINUED_PRETRAIN = "continued_pretrain"  # Pillar 2: domain adaptation
    FINETUNE = "finetune"                    # Pillar 3: instruction fine-tuning


class Method(str, Enum):
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


class Quantization(str, Enum):
    FP16 = "fp16"       # / bf16 full precision base
    INT8 = "int8"
    NF4 = "nf4"         # 4-bit (QLoRA / GGUF Q4-ish planning)


class ArtifactKind(str, Enum):
    PRETRAIN = "pretrain"
    DOMAIN = "domain"
    FINETUNE = "finetune"
    GGUF = "gguf"


# --------------------------------------------------------------------------- #
# Hardware & memory-fit
# --------------------------------------------------------------------------- #
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
    source: str = "unknown"  # "pynvml" | "llmfit" | "fallback"


class FitLevel(str, Enum):
    FITS = "fits"
    TIGHT = "tight"
    WONT_FIT = "wont_fit"


class MemoryEstimate(BaseModel):
    """Breakdown of predicted peak VRAM for a run, in MB."""

    weights_mb: float = 0.0
    optimizer_mb: float = 0.0
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


class OptimConfig(BaseModel):
    learning_rate: float = Field(default=2e-4, gt=0, le=1.0)
    lr_scheduler: str = "cosine"          # cosine | linear | constant
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=1.0)
    weight_decay: float = Field(default=0.01, ge=0.0, le=1.0)
    optimizer: str = "adamw_8bit"         # 8-bit Adam is the 8GB-safe default
    max_grad_norm: float = Field(default=1.0, gt=0)


class TrainConfig(BaseModel):
    epochs: float = Field(default=1.0, gt=0, le=1_000)
    max_steps: int | None = Field(default=None, ge=1)  # if set, overrides epochs
    per_device_batch_size: int = Field(default=2, ge=1, le=1_024)
    gradient_accumulation: int = Field(default=4, ge=1, le=65_536)
    max_seq_length: int = Field(default=1024, ge=16, le=1_048_576)
    packing: bool = False
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    save_steps: int = Field(default=100, ge=1)
    logging_steps: int = Field(default=5, ge=1)
    eval_on_completion: bool = True


class ScratchArch(BaseModel):
    """Architecture config for the from-scratch pretraining pillar."""

    vocab_size: int = Field(default=8192, ge=256, le=1_000_000)
    n_layers: int = Field(default=6, ge=1, le=256)
    n_heads: int = Field(default=6, ge=1, le=256)
    n_embd: int = Field(default=384, ge=32, le=65_536)
    block_size: int = Field(default=256, ge=16, le=1_048_576)  # context length
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    tokenizer_name: str = "byte_bpe"      # custom tokenizer trained on the corpus


class RunConfig(BaseModel):
    """The complete, reproducible description of a training run."""

    backend: str = "unsloth"              # registry key of the TrainingBackend
    task: TaskType
    method: Method = Method.QLORA
    base_model: str = ""                  # HF repo id or local checkpoint dir
    revision: str | None = None
    dataset_id: int | None = Field(default=None, ge=1)
    output_name: str = Field(min_length=1, max_length=160)

    load_in_4bit: bool = True             # QLoRA base quantization
    gradient_checkpointing: bool = True
    lora: LoraParams = Field(default_factory=LoraParams)
    optim: OptimConfig = Field(default_factory=OptimConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)

    # Only used when task == PRETRAIN.
    arch: ScratchArch | None = None

    # Escape hatch for backend-specific knobs without schema churn.
    extra: dict = Field(default_factory=dict)


class ExportedConfig(BaseModel):
    """A run config re-expressed in another engine's native format."""

    format: str            # "axolotl_yaml" | "llamafactory_yaml" | "json"
    filename: str
    content: str
