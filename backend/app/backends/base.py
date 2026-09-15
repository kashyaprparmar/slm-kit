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
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from app.capabilities import (
    Capability,
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


class BackendSelector(Protocol):
    def select(self, cfg: RunConfig) -> BackendSelection: ...


class RegisteredBackendSelector:
    """Select an explicitly requested registered backend.

    Automatic selection is intentionally deferred until the scheduler has
    enough runtime evidence to make a reliable decision.
    """

    def select(self, cfg: RunConfig) -> BackendSelection:
        backend = get_backend(cfg.backend)
        capabilities = backend.capabilities()
        return BackendSelection(backend=backend, capabilities=capabilities)


backend_selector: BackendSelector = RegisteredBackendSelector()


def load_builtin_backends() -> None:
    """Import + register the built-in backends.

    Imported lazily (called at API startup and in the subprocess) so that a
    machine without the GPU stack can still register the metadata-only surface
    of each backend. Heavy imports live inside each backend's ``run``.
    """
    from app.backends import scratch_backend, unsloth_backend

    register_backend(unsloth_backend.UnslothBackend())
    register_backend(unsloth_backend.UnslothBackend(name="transformers"))
    register_backend(scratch_backend.ScratchBackend())
