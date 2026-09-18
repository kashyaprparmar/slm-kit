"""Registry for optional PEFT, optimizer, and acceleration integrations.

This module is the single import-light source for advanced optimization
metadata, compatibility checks, and coarse estimator adjustments.  Runtime
objects stay in ``train_entry.optimization_runtime`` so API startup never loads
Torch, PEFT, or optional CUDA extensions.
"""

from __future__ import annotations

from collections.abc import Iterable

from app.capabilities import (
    Capability,
    DependencyStatus,
    EvidenceLevel,
    OptimizationCapability,
    SupportState,
    dependency_statuses,
)
from app.domain import Method, RunConfig, ValidationReport


def _version_at_least(value: str | None, minimum: tuple[int, ...]) -> bool:
    if not value or value == "unknown":
        return False
    parts: list[int] = []
    for token in value.split("."):
        digits = "".join(character for character in token if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts[: len(minimum)]) >= minimum if parts else False


def _capability_for_dependency(
    dependency: str,
    dependencies: dict[str, DependencyStatus],
    *,
    reason: str,
    minimum: tuple[int, ...] | None = None,
    extra_requirements: Iterable[str] = (),
) -> Capability:
    status = dependencies[dependency]
    requirements = [dependency, *extra_requirements]
    if not status.installed:
        return Capability(
            state=SupportState.MISSING_DEPENDENCY,
            reason=f"Install {dependency} to use this optimization.",
            requirements=requirements,
        )
    if minimum and not _version_at_least(status.version, minimum):
        display = ".".join(str(part) for part in minimum)
        return Capability(
            state=SupportState.INCOMPATIBLE,
            reason=f"{dependency} {display} or newer is required; installed version is {status.version or 'unknown'}.",
            requirements=[f"{dependency}>={display}", *extra_requirements],
            evidence=EvidenceLevel.INSTALLED,
        )
    return Capability(
        state=SupportState.SUPPORTED,
        reason=reason,
        requirements=requirements,
        evidence=EvidenceLevel.INSTALLED,
    )


def _unimplemented(id: str, label: str, category: str, disclosure: str, reason: str, *, requirements: list[str] | None = None, schema: dict | None = None, impact: dict | None = None) -> OptimizationCapability:
    return OptimizationCapability(
        id=id,
        label=label,
        category=category,  # type: ignore[arg-type]
        disclosure=disclosure,  # type: ignore[arg-type]
        support=Capability(state=SupportState.UNSUPPORTED, reason=reason, requirements=requirements or []),
        configuration_schema=schema or {},
        impact=impact or {},
    )


def optimization_capabilities(
    backend_name: str,
    dependencies: dict[str, DependencyStatus] | None = None,
) -> dict[str, OptimizationCapability]:
    """Return the authoritative advanced-option contract for a backend."""
    dependencies = dependencies or dependency_statuses()
    native = backend_name == "transformers"
    peft = dependencies["peft"]
    peft_base = _capability_for_dependency(
        "peft", dependencies, reason="PEFT exposes this adapter option in the installed runtime.", minimum=(0, 11)
    )
    native_only = Capability(
        state=SupportState.UNSUPPORTED,
        reason="This option uses the native Transformers/PEFT worker to preserve its training semantics.",
        requirements=["backend=transformers"],
    )
    if native and not peft_base.allowed:
        native_only = peft_base
    elif native:
        native_only = peft_base
    registry: dict[str, OptimizationCapability] = {
        "rslora": OptimizationCapability(
            id="rslora", label="rsLoRA", category="peft", disclosure="advanced",
            support=peft_base if native else Capability(state=SupportState.SUPPORTED if peft_base.allowed else peft_base.state, reason="Unsloth exposes rsLoRA when its PEFT runtime is compatible." if peft_base.allowed else peft_base.reason, requirements=peft_base.requirements, evidence=peft_base.evidence),
            installed_version=peft.version,
            compatibility={"methods": ["lora", "qlora", "dora"]},
            configuration_schema={"fields": [{"path": "lora.use_rslora", "type": "boolean"}]},
            impact={"memory_multiplier": 1.0, "performance": "Changes LoRA scaling; memory impact is negligible."},
        ),
        "lora_plus": OptimizationCapability(
            id="lora_plus", label="LoRA+", category="peft", disclosure="advanced",
            support=native_only,
            installed_version=peft.version,
            compatibility={"backends": ["transformers"], "methods": ["lora", "dora"], "optimizer": ["adamw_torch"]},
            configuration_schema={"fields": [{"path": "lora.lora_plus_lr_ratio", "type": "number", "minimum": 1.0}]},
            impact={"memory_multiplier": 1.0, "performance": "Uses separate adapter parameter groups; no material VRAM increase is expected."},
        ),
        "pissa": OptimizationCapability(
            id="pissa", label="PiSSA", category="peft", disclosure="advanced",
            support=native_only,
            installed_version=peft.version,
            compatibility={"backends": ["transformers"], "methods": ["lora", "dora"], "quantization": ["none"]},
            configuration_schema={"fields": [{"path": "lora.init_method", "type": "enum", "values": ["pissa"]}]},
            impact={"memory_multiplier": 1.0, "performance": "Performs an SVD during adapter initialization; startup can be slower."},
        ),
        "loftq": _unimplemented("loftq", "LoftQ", "peft", "expert", "LoftQ requires a verified joint quantization and initialization path; it is not exposed until that runtime path is tested.", requirements=["peft", "bitsandbytes", "CUDA"], schema={"fields": [{"path": "lora.init_method", "type": "enum", "values": ["loftq"]}]}, impact={"memory_multiplier": 1.0, "performance": "Initialization may be slow and requires a compatible weight format."}),
        "eva": _unimplemented("eva", "EVA", "peft", "expert", "EVA needs a calibration pass over the training dataset, which is not yet integrated with the shared lifecycle.", requirements=["peft"], schema={"fields": [{"path": "lora.init_method", "type": "enum", "values": ["eva"]}]}, impact={"memory_multiplier": 1.0, "performance": "Adds a data-driven initialization pass before training."}),
        "oft": _unimplemented("oft", "OFT", "peft", "expert", "OFT is not implemented by the current model loader."),
        "qoft": _unimplemented("qoft", "QOFT", "peft", "expert", "QOFT is not implemented by the current model loader."),
    }
    optimizer_specs = {
        "galore": ("GaLore", "galore-torch", "galore_adamw", "Low-rank gradient projection can reduce optimizer-state memory."),
        "apollo": ("APOLLO", "apollo-torch", "apollo_adamw", "Low-rank optimizer states can reduce optimizer memory."),
        "badam": ("BAdam", "badam", None, "Blockwise updates can reduce active optimizer memory."),
        "adam_mini": ("Adam-mini", "adam-mini", None, "Reduced optimizer-state memory depends on model parameter grouping."),
        "muon": ("Muon", "muon", None, "Performance and memory depend on model architecture and parameter grouping."),
    }
    for option_id, (label, dependency, trainer_name, performance) in optimizer_specs.items():
        status = dependencies[dependency]
        if not status.installed:
            support = _capability_for_dependency(dependency, dependencies, reason="Optional optimizer is installed.")
        else:
            support = Capability(
                state=SupportState.UNSUPPORTED,
                reason=f"{label} is installed but no verified adapter has been registered for this runtime yet.",
                requirements=[dependency], evidence=EvidenceLevel.INSTALLED,
            )
        registry[option_id] = OptimizationCapability(
            id=option_id, label=label, category="optimizer", disclosure="expert", support=support,
            installed_version=status.version,
            compatibility={"backends": ["transformers"], "methods": ["full", "freeze", "lora", "dora"], "trainer_optimizer": trainer_name},
            configuration_schema={"fields": [
                {"path": "optim.strategy", "type": "enum", "values": [option_id]},
                {"path": "optim.target_modules", "type": "string[]", "optional": True},
                {"path": "optim.low_rank_rank", "type": "integer", "minimum": 1},
                {"path": "optim.update_interval", "type": "integer", "minimum": 1},
            ]},
            impact={"memory_multiplier": None, "performance": performance},
        )
    flash = _capability_for_dependency("flash-attn", dependencies, reason="FlashAttention 2 is installed.", extra_requirements=["CUDA"])
    liger = _capability_for_dependency("liger-kernel", dependencies, reason="Liger kernels are installed.")
    transformers = dependencies["transformers"]
    neftune = _capability_for_dependency("transformers", dependencies, reason="The installed Transformers runtime supports NEFTune.", minimum=(4, 36))
    registry.update({
        "flash_attention_2": OptimizationCapability(id="flash_attention_2", label="FlashAttention 2", category="acceleration", disclosure="expert", support=flash, installed_version=dependencies["flash-attn"].version, compatibility={"precision": ["auto", "bf16", "fp16"], "hardware": ["CUDA"]}, configuration_schema={"fields": [{"path": "runtime.attention", "type": "enum", "values": ["flash_attention_2"]}]}, impact={"memory_multiplier": 0.8, "performance": "Can reduce attention memory and improve throughput on supported CUDA hardware."}),
        "liger": OptimizationCapability(id="liger", label="Liger kernels", category="acceleration", disclosure="expert", support=liger, installed_version=dependencies["liger-kernel"].version, compatibility={"backends": ["transformers"], "runtime": ["TRL"]}, configuration_schema={"fields": [{"path": "runtime.use_liger", "type": "boolean"}]}, impact={"memory_multiplier": None, "performance": "Kernel-specific memory and throughput effects vary by model and runtime."}),
        "neftune": OptimizationCapability(id="neftune", label="NEFTune", category="acceleration", disclosure="advanced", support=neftune if native else Capability(state=SupportState.UNSUPPORTED, reason="NEFTune uses the native Transformers trainer configuration.", requirements=["backend=transformers"]), installed_version=transformers.version, compatibility={"backends": ["transformers"], "tasks": ["finetune", "alignment"]}, configuration_schema={"fields": [{"path": "runtime.neftune_noise_alpha", "type": "number", "minimum": 0.000001}]}, impact={"memory_multiplier": 1.0, "performance": "Adds embedding noise during training; memory impact is negligible."}),
    })
    return registry


def validate_optimization_config(cfg: RunConfig, optimizations: dict[str, OptimizationCapability], report: ValidationReport) -> None:
    """Validate enabled options against the same descriptors returned by API."""
    requested: list[str] = []
    if cfg.lora.use_rslora:
        requested.append("rslora")
    if cfg.lora.init_method != "standard":
        requested.append(cfg.lora.init_method)
    if cfg.lora.lora_plus_lr_ratio is not None:
        requested.append("lora_plus")
    if cfg.optim.strategy != "default":
        requested.append(cfg.optim.strategy)
    if cfg.runtime.use_liger:
        requested.append("liger")
    if cfg.runtime.neftune_noise_alpha is not None:
        requested.append("neftune")
    for option_id in requested:
        descriptor = optimizations.get(option_id)
        if descriptor is None:
            report.error(f"Unknown optimization '{option_id}'.")
        elif not descriptor.support.allowed:
            report.error(f"{descriptor.label}: {descriptor.support.reason}")
    if cfg.lora.lora_plus_lr_ratio is not None:
        if cfg.method not in {Method.LORA, Method.DORA}:
            report.error("LoRA+ supports LoRA or DoRA, not the requested training method.")
        if cfg.optim.optimizer != "adamw_torch":
            report.error("LoRA+ currently requires optim.optimizer='adamw_torch'.")
    if cfg.lora.init_method == "pissa":
        if cfg.method not in {Method.LORA, Method.DORA} or cfg.quantization.mode != "none":
            report.error("PiSSA currently supports unquantized LoRA or DoRA only.")
    if cfg.runtime.neftune_noise_alpha is not None and cfg.task.value not in {"finetune", "alignment"}:
        report.error("NEFTune is available only for fine-tuning and alignment runs.")


def optimizer_memory_multiplier(cfg: RunConfig) -> float:
    """Return a conservative estimator adjustment for executable strategies."""
    # No optional external optimizer is enabled until it has a registered
    # runtime adapter.  Keep the default estimate exact rather than guessing.
    return 1.0
