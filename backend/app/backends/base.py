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
from typing import Protocol, runtime_checkable

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


@runtime_checkable
class TrainingBackend(Protocol):
    name: str
    supported_tasks: set[TaskType]
    supported_methods: set[Method]

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


def load_builtin_backends() -> None:
    """Import + register the built-in backends.

    Imported lazily (called at API startup and in the subprocess) so that a
    machine without the GPU stack can still register the metadata-only surface
    of each backend. Heavy imports live inside each backend's ``run``.
    """
    from app.backends import scratch_backend, unsloth_backend

    register_backend(unsloth_backend.UnslothBackend())
    generic = unsloth_backend.UnslothBackend()
    generic.name = "transformers"
    register_backend(generic)
    register_backend(scratch_backend.ScratchBackend())
