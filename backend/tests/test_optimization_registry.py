from __future__ import annotations

from app.backends.base import AutoBackendSelector
from app.backends.unsloth_backend import UnslothBackend
from app.capabilities import DEPENDENCIES, Capability, DependencyStatus, SupportState
from app.domain import HardwareProfile, Method, RunConfig, TaskType, ValidationReport
from app.optimizations import optimization_capabilities, validate_optimization_config


def _dependencies(**installed_versions: str) -> dict[str, DependencyStatus]:
    return {
        spec.name: DependencyStatus(
            name=spec.name,
            installed=spec.name in installed_versions,
            version=installed_versions.get(spec.name),
            optional=spec.optional,
        )
        for spec in DEPENDENCIES
    }


def _config(**overrides) -> RunConfig:
    values = {
        "backend": "transformers",
        "task": TaskType.FINETUNE,
        "method": Method.LORA,
        "base_model": "model",
        "dataset_id": 1,
        "output_name": "optimization-test",
        "quantization": {"mode": "none"},
        "optim": {"optimizer": "adamw_torch"},
    }
    values.update(overrides)
    return RunConfig(**values)


def test_registry_has_one_descriptor_for_every_requested_optimization():
    registry = optimization_capabilities(
        "transformers", _dependencies(peft="0.15.2", transformers="4.52.4", torch="2.7.1")
    )
    assert {
        "rslora", "lora_plus", "pissa", "loftq", "eva", "oft", "qoft",
        "galore", "apollo", "badam", "adam_mini", "muon",
        "flash_attention_2", "liger", "neftune",
    } <= set(registry)
    assert registry["pissa"].support.allowed
    assert registry["lora_plus"].support.allowed
    assert registry["neftune"].support.allowed
    assert registry["galore"].support.state == SupportState.MISSING_DEPENDENCY
    assert registry["loftq"].support.state == SupportState.UNSUPPORTED
    assert registry["pissa"].configuration_schema["fields"]
    assert "performance" in registry["pissa"].impact


def test_registry_keeps_native_only_peft_paths_out_of_unsloth_profile():
    registry = optimization_capabilities(
        "unsloth", _dependencies(peft="0.15.2", transformers="4.52.4", torch="2.7.1")
    )
    assert registry["rslora"].support.allowed
    assert not registry["pissa"].support.allowed
    assert not registry["lora_plus"].support.allowed


def test_validation_accepts_versioned_pissa_loraplus_and_neftune_contracts():
    registry = optimization_capabilities(
        "transformers", _dependencies(peft="0.15.2", transformers="4.52.4", torch="2.7.1")
    )
    cfg = _config(
        lora={"init_method": "pissa", "lora_plus_lr_ratio": 4},
        runtime={"neftune_noise_alpha": 5},
    )
    report = ValidationReport()
    validate_optimization_config(cfg, registry, report)
    assert report.ok


def test_validation_rejects_unverified_or_incompatible_options_before_worker_launch():
    registry = optimization_capabilities(
        "transformers", _dependencies(peft="0.15.2", transformers="4.52.4", torch="2.7.1")
    )
    cfg = _config(method=Method.QLORA, lora={"init_method": "pissa"})
    report = ValidationReport()
    validate_optimization_config(cfg, registry, report)
    assert not report.ok
    assert any("PiSSA" in issue.message for issue in report.issues)

    external = _config(optim={"optimizer": "adamw_torch", "strategy": "galore"})
    report = ValidationReport()
    validate_optimization_config(external, registry, report)
    assert not report.ok
    assert any("GaLore" in issue.message for issue in report.issues)


def test_auto_selector_rejects_backend_that_cannot_preserve_requested_optimization():
    cfg = _config(backend="auto", lora={"init_method": "pissa"})
    base = UnslothBackend("transformers").capabilities()
    supported = Capability(state=SupportState.SUPPORTED, reason="test backend")
    descriptor = base.model_copy(
        update={
            "name": "other",
            "availability": supported,
            "tasks": {**base.tasks, "finetune": supported},
            "methods": {**base.methods, "lora": supported},
            "optimizations": {
                **base.optimizations,
                "pissa": base.optimizations["pissa"].model_copy(
                    update={"support": Capability(state=SupportState.UNSUPPORTED, reason="native-only")}
                ),
            },
        },
    )
    assert AutoBackendSelector._incompatibility(cfg, HardwareProfile(), None, descriptor) == "native-only"
