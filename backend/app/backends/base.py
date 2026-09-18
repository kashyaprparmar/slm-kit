"""The pluggable ``TrainingBackend`` interface.

Every engine (Unsloth, from-scratch, and future Axolotl/LlamaFactory) implements
this. Backends run *inside the training subprocess* — their ``run`` method is a
generator that yields ``TrainingEvent`` objects, which the subprocess entrypoint
serializes to stdout for the parent to consume.

The non-``run`` methods (``validate_config``, ``estimate_footprint``,
``export_config``) are cheap and are also called from the API process to power
the pre-launch fit/OOM guard and config export — so they must NOT import heavy
GPU libraries at module load time.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from app.capabilities import (
    Capability,
    OptimizationCapability,
    PeftMethodCapability,
    QuantizationCapability,
    TokenizerCapabilities,
)
from app.core.events import TrainingEvent
from app.domain import (
    ExportedConfig,
    HardwareProfile,
    MemoryEstimate,
    Method,
    RunConfig,
    TaskType,
    ValidationReport,
)

if TYPE_CHECKING:
    from app.models.capabilities import ModelCapabilities


@dataclass
class RunContext:
    """Runtime paths + cancellation, handed to a backend's ``run``."""

    run_id: int
    workdir: Path
    checkpoint_dir: Path
    resume_from: Path | None = None

    def should_stop(self) -> bool:
        """True once a graceful-stop sentinel is written by the runner.

        Backends should poll this at a natural boundary (e.g. each logging step),
        save a checkpoint, and return so VRAM is released cleanly. A hard cancel
        still works by terminating the subprocess.
        """
        return (self.workdir / "STOP").exists()


class TrainingBackendCapabilities(BaseModel):
    """Authoritative backend contract returned by the API and used to validate runs."""

    schema_version: Literal[1] = 1
    name: str
    display_name: str
    description: str
    availability: Capability
    tasks: dict[str, Capability]
    stages: dict[str, Capability]
    methods: dict[str, Capability]
    tokenizer: TokenizerCapabilities
    peft: dict[str, PeftMethodCapability] = Field(default_factory=dict)
    quantization: dict[str, QuantizationCapability] = Field(default_factory=dict)
    precision: dict[str, Capability] = Field(default_factory=dict)
    attention: dict[str, Capability] = Field(default_factory=dict)
    gradient_checkpointing: dict[str, Capability] = Field(default_factory=dict)
    rope: dict[str, Capability] = Field(default_factory=dict)
    optimizations: dict[str, OptimizationCapability] = Field(default_factory=dict)
    optional_features: dict[str, Capability] = Field(default_factory=dict)
    platforms: list[str] = Field(default_factory=list)
    architectures: list[str] = Field(default_factory=list)
    required_dependencies: list[str] = Field(default_factory=list)
    optional_dependencies: list[str] = Field(default_factory=list)

    def supports_task(self, task: TaskType) -> bool:
        capability = self.tasks.get(task.value)
        return bool(capability and capability.allowed)

    def supports_method(self, method: Method) -> bool:
        capability = self.methods.get(method.value)
        return bool(capability and capability.allowed)

    def validate_operation(self, cfg: RunConfig, report: ValidationReport) -> None:
        if not self.supports_task(cfg.task):
            report.error(f"{self.name} backend does not support task '{cfg.task.value}'.")
        if not self.supports_method(cfg.method):
            capability = self.methods.get(cfg.method.value)
            reason = f" {capability.reason}" if capability else ""
            report.error(f"{self.name} backend does not support method '{cfg.method.value}'.{reason}")

    def validate_availability(self, report: ValidationReport, *, required: bool) -> None:
        if self.availability.allowed:
            return
        message = f"Backend '{self.name}' is unavailable: {self.availability.reason}"
        report.error(message) if required else report.warn(message)

    @property
    def supported_tasks(self) -> set[TaskType]:
        return {task for task in TaskType if self.supports_task(task)}

    @property
    def supported_methods(self) -> set[Method]:
        return {method for method in Method if self.supports_method(method)}


@runtime_checkable
class TrainingBackend(Protocol):
    name: str

    @property
    def supported_tasks(self) -> set[TaskType]: ...

    @property
    def supported_methods(self) -> set[Method]: ...

    def capabilities(self) -> TrainingBackendCapabilities:
        ...

    def validate_config(self, cfg: RunConfig, hw: HardwareProfile) -> ValidationReport:
        ...

    def estimate_footprint(self, cfg: RunConfig, hw: HardwareProfile) -> MemoryEstimate:
        ...

    def export_config(self, cfg: RunConfig) -> ExportedConfig:
        ...

    def run(self, cfg: RunConfig, ctx: RunContext) -> Iterator[TrainingEvent]:
        ...


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_REGISTRY: dict[str, TrainingBackend] = {}


def register_backend(backend: TrainingBackend) -> None:
    _REGISTRY[backend.name] = backend


def get_backend(name: str) -> TrainingBackend:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown training backend: {name!r}. Registered: {list(_REGISTRY)}")
    return _REGISTRY[name]


def list_backends() -> list[TrainingBackend]:
    return list(_REGISTRY.values())


@dataclass(frozen=True)
class BackendSelection:
    backend: TrainingBackend
    capabilities: TrainingBackendCapabilities
    requested_backend: str
    reason: str
    alternatives: tuple[dict[str, str], ...] = ()

    def as_dict(self) -> dict:
        return {
            "requested_backend": self.requested_backend,
            "selected_backend": self.backend.name,
            "reason": self.reason,
            "alternatives": list(self.alternatives),
        }


class BackendSelector(Protocol):
    def select(
        self,
        cfg: RunConfig,
        hw: HardwareProfile | None = None,
        model: ModelCapabilities | None = None,
    ) -> BackendSelection: ...


class RegisteredBackendSelector:
    """Select an explicitly requested registered backend.

    Automatic selection is intentionally deferred until the scheduler has
    enough runtime evidence to make a reliable decision.
    """

    def select(
        self,
        cfg: RunConfig,
        hw: HardwareProfile | None = None,
        model: ModelCapabilities | None = None,
    ) -> BackendSelection:
        backend = get_backend(cfg.backend)
        capabilities = backend.capabilities()
        alternatives = tuple(
            {"backend": candidate.name, "reason": "An explicit backend was requested."}
            for candidate in list_backends()
            if candidate.name != backend.name
        )
        return BackendSelection(
            backend=backend,
            capabilities=capabilities,
            requested_backend=cfg.backend,
            reason=f"Backend '{cfg.backend}' was requested explicitly.",
            alternatives=alternatives,
        )


class AutoBackendSelector(RegisteredBackendSelector):
    """Resolve Auto without changing the requested operation semantics."""

    def select(
        self,
        cfg: RunConfig,
        hw: HardwareProfile | None = None,
        model: ModelCapabilities | None = None,
    ) -> BackendSelection:
        if cfg.backend != "auto":
            return super().select(cfg, hw, model)
        hw = hw or HardwareProfile()
        candidates = list_backends()
        compatible: list[tuple[int, TrainingBackend, TrainingBackendCapabilities, str]] = []
        rejected: list[dict[str, str]] = []
        for backend in candidates:
            descriptor = backend.capabilities()
            reason = self._incompatibility(cfg, hw, model, descriptor)
            if reason:
                rejected.append({"backend": backend.name, "reason": reason})
                continue
            priority = self._priority(cfg, backend.name)
            compatible.append((priority, backend, descriptor, self._selection_reason(cfg, backend.name)))
        if not compatible:
            detail = "; ".join(f"{item['backend']}: {item['reason']}" for item in rejected)
            raise KeyError(f"No compatible training backend for this operation. {detail}")
        compatible.sort(key=lambda item: (item[0], item[1].name))
        _, backend, descriptor, reason = compatible[0]
        for _, alternative, _, _ in compatible[1:]:
            rejected.append({
                "backend": alternative.name,
                "reason": f"Compatible, but ranked below '{backend.name}' for this exact operation.",
            })
        return BackendSelection(
            backend=backend,
            capabilities=descriptor,
            requested_backend="auto",
            reason=reason,
            alternatives=tuple(rejected),
        )

    @staticmethod
    def _incompatibility(cfg, hw, model, descriptor: TrainingBackendCapabilities) -> str | None:
        if not descriptor.availability.allowed:
            return descriptor.availability.reason
        task_capability = descriptor.tasks.get(cfg.task.value)
        if not task_capability or not task_capability.allowed:
            return task_capability.reason if task_capability else f"Task '{cfg.task.value}' is unsupported."
        method_capability = descriptor.methods.get(cfg.method.value)
        if not method_capability or not method_capability.allowed:
            return method_capability.reason if method_capability else f"Method '{cfg.method.value}' is unsupported."
        if descriptor.name == "unsloth" and not hw.cuda_available:
            return "Unsloth requires a detected CUDA GPU."
        if cfg.method == Method.QLORA and not hw.cuda_available:
            return "QLoRA requires a detected CUDA GPU."
        if cfg.runtime.attention == "flash_attention_2" and not hw.cuda_available:
            return "FlashAttention 2 requires CUDA."
        for group, choice in (
            (descriptor.precision, cfg.runtime.precision),
            (descriptor.attention, cfg.runtime.attention),
            (descriptor.gradient_checkpointing, cfg.runtime.gradient_checkpointing),
        ):
            capability = group.get(choice)
            if capability is not None and not capability.allowed:
                return capability.reason
        if cfg.runtime.rope.enabled:
            capability = descriptor.rope.get(cfg.runtime.rope.type)
            if capability is None or not capability.allowed:
                return capability.reason if capability else f"RoPE policy '{cfg.runtime.rope.type}' is unsupported."
        if cfg.runtime.use_liger:
            capability = descriptor.optimizations.get("liger")
            capability = capability.support if capability is not None else descriptor.optional_features.get("liger")
            if capability is None or not capability.allowed:
                return capability.reason if capability else "Liger is unsupported."
        requested_optimizations = []
        if cfg.lora.use_rslora:
            requested_optimizations.append("rslora")
        if cfg.lora.init_method != "standard":
            requested_optimizations.append(cfg.lora.init_method)
        if cfg.lora.lora_plus_lr_ratio is not None:
            requested_optimizations.append("lora_plus")
        if cfg.optim.strategy != "default":
            requested_optimizations.append(cfg.optim.strategy)
        if cfg.runtime.neftune_noise_alpha is not None:
            requested_optimizations.append("neftune")
        for option_id in requested_optimizations:
            capability = descriptor.optimizations.get(option_id)
            if capability is None or not capability.support.allowed:
                return (
                    capability.support.reason
                    if capability is not None
                    else f"Optimization '{option_id}' is not provided by {descriptor.display_name}."
                )
        tokenizer_mode = descriptor.tokenizer.modes.get(cfg.tokenizer.mode)
        if tokenizer_mode is None or not tokenizer_mode.allowed:
            return tokenizer_mode.reason if tokenizer_mode else f"Tokenizer mode '{cfg.tokenizer.mode}' is unsupported."
        if cfg.task == TaskType.FINETUNE:
            loss_policy = descriptor.tokenizer.loss_policies.get(cfg.tokenizer.loss_policy)
            if loss_policy is None or not loss_policy.allowed:
                return loss_policy.reason if loss_policy else f"Loss policy '{cfg.tokenizer.loss_policy}' is unsupported."
        if cfg.tokenizer.chat_template:
            custom_template = descriptor.tokenizer.templates.explicit_override
            if not custom_template.allowed:
                return custom_template.reason
        if cfg.task == TaskType.ALIGNMENT:
            objective = descriptor.optional_features.get(f"objective:{cfg.alignment.objective}")
            if objective is None:
                return f"Objective '{cfg.alignment.objective}' is not provided by {descriptor.display_name}."
            if not objective.allowed:
                return objective.reason
            if cfg.alignment.objective == "dpo":
                loss = descriptor.optional_features.get(f"dpo_loss:{cfg.alignment.dpo_loss_variant}")
                if loss is None or not loss.allowed:
                    return loss.reason if loss else "The requested DPO loss is unsupported."
            reference = descriptor.optional_features.get(f"reference:{cfg.alignment.reference.strategy}")
            if reference is None or not reference.allowed:
                return reference.reason if reference else "The requested alignment reference strategy is unsupported."
        if model is not None and descriptor.name in {"transformers", "unsloth"}:
            capability = model.backends.get(descriptor.name)
            if capability is not None and not capability.allowed:
                return capability.reason
        return None

    @staticmethod
    def _priority(cfg: RunConfig, backend_name: str) -> int:
        if cfg.task == TaskType.PRETRAIN:
            order = {"scratch": 0}
        elif cfg.task == TaskType.ALIGNMENT:
            order = {"transformers": 0, "llamafactory": 1, "unsloth": 2}
        elif cfg.runtime.gradient_checkpointing == "backend_optimized":
            order = {"unsloth": 0}
        elif cfg.method in {Method.LORA, Method.QLORA, Method.DORA}:
            order = {"unsloth": 0, "transformers": 1, "llamafactory": 2}
        else:
            order = {"transformers": 0, "llamafactory": 1, "unsloth": 2}
        return order.get(backend_name, 100)

    @staticmethod
    def _selection_reason(cfg: RunConfig, backend_name: str) -> str:
        operation = f"{cfg.task.value}/{cfg.method.value}"
        if backend_name == "scratch":
            return f"Selected the built-in scratch engine for {operation}."
        if backend_name == "unsloth":
            return f"Selected installed Unsloth acceleration for compatible {operation} on CUDA."
        if backend_name == "llamafactory":
            return f"Selected the optional installed LLaMA-Factory engine for compatible {operation}."
        return f"Selected the native Transformers/TRL/PEFT engine for broad, exact {operation} semantics."


backend_selector: BackendSelector = AutoBackendSelector()


def load_builtin_backends() -> None:
    """Import + register the built-in backends.

    Imported lazily (called at API startup and in the subprocess) so that a
    machine without the GPU stack can still register the metadata-only surface
    of each backend. Heavy imports live inside each backend's ``run``.
    """
    from app.backends import llamafactory_backend, scratch_backend, unsloth_backend

    register_backend(unsloth_backend.UnslothBackend())
    register_backend(unsloth_backend.UnslothBackend(name="transformers"))
    register_backend(scratch_backend.ScratchBackend())
    register_backend(llamafactory_backend.LlamaFactoryBackend())
