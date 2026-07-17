"""Shared domain types: enums, run configuration, and hardware/fit models.

These Pydantic models are the *contract* between the API, the job queue, the
training subprocess, and the frontend. A ``RunConfig`` is serialized to JSON,
stored on the ``Run`` row, and handed verbatim to the training subprocess so any
run is fully reproducible and cloneable.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

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
    gpu_name: Optional[str] = None
    vram_total_mb: Optional[int] = None
    vram_free_mb: Optional[int] = None
    gpu_util_pct: Optional[float] = None
    ram_total_mb: Optional[int] = None
    ram_free_mb: Optional[int] = None
    cpu_count: Optional[int] = None
    cpu_util_pct: Optional[float] = None
    disk_free_mb: Optional[int] = None
    source: str = "unknown"  # "pynvml" | "llmfit" | "fallback"


class FitLevel(str, Enum):
    FITS = "fits"
    TIGHT = "tight"
    WONT_FIT = "wont_fit"


class MemoryEstimate(BaseModel):
    """Breakdown of predicted peak VRAM for a run, in MB."""

    weights_mb: float = 0.0
    optimizer_mb: float = 0.0
    activations_mb: float = 0.0
    kv_cache_mb: float = 0.0
    overhead_mb: float = 0.0
    total_mb: float = 0.0
    budget_mb: float = 0.0
    fit: FitLevel = FitLevel.FITS
    source: str = "fallback"        # "llmfit" | "fallback"
    notes: list[str] = Field(default_factory=list)


class ValidationIssue(BaseModel):
    level: Literal["error", "warning", "info"]
    message: str
    line: Optional[int] = None


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
    r: int = 16
    alpha: int = 16
    dropout: float = 0.0
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    use_rslora: bool = False
    use_dora: bool = False  # flipped on automatically when method == DORA


class OptimConfig(BaseModel):
    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"          # cosine | linear | constant
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    optimizer: str = "adamw_8bit"         # 8-bit Adam is the 8GB-safe default
    max_grad_norm: float = 1.0


class TrainConfig(BaseModel):
    epochs: float = 1.0
    max_steps: Optional[int] = None       # if set, overrides epochs
    per_device_batch_size: int = 2
    gradient_accumulation: int = 4        # effective batch 8 by default
    max_seq_length: int = 1024
    packing: bool = False
    seed: int = 42
    save_steps: int = 100
    logging_steps: int = 5
    eval_on_completion: bool = True


class ScratchArch(BaseModel):
    """Architecture config for the from-scratch pretraining pillar."""

    vocab_size: int = 8192
    n_layers: int = 6
    n_heads: int = 6
    n_embd: int = 384
    block_size: int = 256                 # context length
    dropout: float = 0.1
    tokenizer_name: str = "byte_bpe"      # custom tokenizer trained on the corpus


class RunConfig(BaseModel):
    """The complete, reproducible description of a training run."""

    backend: str = "unsloth"              # registry key of the TrainingBackend
    task: TaskType
    method: Method = Method.QLORA
    base_model: str = ""                  # HF repo id or local checkpoint dir
    dataset_id: Optional[int] = None
    output_name: str

    load_in_4bit: bool = True             # QLoRA base quantization
    lora: LoraParams = Field(default_factory=LoraParams)
    optim: OptimConfig = Field(default_factory=OptimConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)

    # Only used when task == PRETRAIN.
    arch: Optional[ScratchArch] = None

    # Escape hatch for backend-specific knobs without schema churn.
    extra: dict = Field(default_factory=dict)


class ExportedConfig(BaseModel):
    """A run config re-expressed in another engine's native format."""

    format: str            # "axolotl_yaml" | "llamafactory_yaml" | "json"
    filename: str
    content: str
