"""Shared, import-light capability contracts.

These models are the wire-format used by API validation and the frontend.  The
module must stay safe to import in the API process: dependency inspection uses
package metadata and never imports GPU libraries.
"""

from __future__ import annotations

import importlib.util
import shutil
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

from pydantic import BaseModel, Field


class SupportState(str, Enum):
    SUPPORTED = "supported"
    EXPERIMENTAL = "experimental"
    UNSUPPORTED = "unsupported"
    NOT_INSTALLED = "not_installed"
    MISSING_DEPENDENCY = "missing_dependency"
    INCOMPATIBLE = "incompatible"
    REQUIRES_CONVERSION = "requires_conversion"


class EvidenceLevel(str, Enum):
    """How strongly a capability has been established."""

    DECLARED = "declared"
    INSTALLED = "installed"
    METADATA = "metadata"
    RUNTIME = "runtime"


class Capability(BaseModel):
    state: SupportState
    reason: str
    requirements: list[str] = Field(default_factory=list)
    evidence: EvidenceLevel = EvidenceLevel.DECLARED

    @property
    def allowed(self) -> bool:
        return self.state in {SupportState.SUPPORTED, SupportState.EXPERIMENTAL}


class DependencySpec(BaseModel):
    name: str
    import_name: str | None = None
    optional: bool = False
    executable: bool = False


class DependencyStatus(BaseModel):
    name: str
    installed: bool
    version: str | None = None
    optional: bool = False


DEPENDENCIES: tuple[DependencySpec, ...] = (
    DependencySpec(name="torch"),
    DependencySpec(name="transformers"),
    DependencySpec(name="peft"),
    DependencySpec(name="trl"),
    DependencySpec(name="unsloth", optional=True),
    DependencySpec(name="bitsandbytes", optional=True),
    DependencySpec(name="accelerate", optional=True),
    DependencySpec(name="tokenizers"),
    DependencySpec(name="datasets"),
    DependencySpec(name="pyarrow", optional=True),
    DependencySpec(name="sentencepiece", optional=True),
    DependencySpec(name="tiktoken", optional=True),
    DependencySpec(name="torchao", optional=True),
    DependencySpec(name="xformers", optional=True),
    DependencySpec(name="flash-attn", import_name="flash_attn", optional=True),
    DependencySpec(name="liger-kernel", import_name="liger_kernel", optional=True),
    # Advanced optimization integrations are deliberately optional.  Keep this
    # metadata-only so inspecting capabilities never imports a CUDA extension.
    DependencySpec(name="galore-torch", import_name="galore_torch", optional=True),
    DependencySpec(name="apollo-torch", import_name="apollo_torch", optional=True),
    DependencySpec(name="badam", optional=True),
    DependencySpec(name="adam-mini", import_name="adam_mini", optional=True),
    DependencySpec(name="muon", optional=True),
    DependencySpec(name="llamafactory", optional=True),
    DependencySpec(name="llamafactory-cli", optional=True, executable=True),
    DependencySpec(name="vllm", optional=True),
    DependencySpec(name="sglang", optional=True),
    DependencySpec(name="ollama", optional=True, executable=True),
)


def detect_dependency(spec: DependencySpec) -> DependencyStatus:
    """Inspect an optional dependency without importing it."""
    if spec.executable:
        installed = shutil.which(spec.name) is not None
        return DependencyStatus(name=spec.name, installed=installed, optional=spec.optional)
    import_name = spec.import_name or spec.name.replace("-", "_")
    try:
        installed = importlib.util.find_spec(import_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        installed = False
    package_version: str | None = None
    if installed:
        try:
            package_version = version(spec.name)
        except PackageNotFoundError:
            package_version = "unknown"
        except Exception:
            package_version = None
    return DependencyStatus(
        name=spec.name,
        installed=installed,
        version=package_version,
        optional=spec.optional,
    )


def dependency_statuses() -> dict[str, DependencyStatus]:
    return {spec.name: detect_dependency(spec) for spec in DEPENDENCIES}


class ChatTemplateCapabilities(BaseModel):
    native: Capability
    explicit_override: Capability
    fallback: Capability


class TokenizerCapabilities(BaseModel):
    modes: dict[str, Capability]
    loss_policies: dict[str, Capability]
    templates: ChatTemplateCapabilities


class PeftMethodCapability(BaseModel):
    method: str
    support: Capability
    adapter_based: bool = True
    requires_quantized_base: bool = False
    features: dict[str, Capability] = Field(default_factory=dict)


class QuantizationCapability(BaseModel):
    format: str
    operations: list[str] = Field(default_factory=list)
    support: Capability
    compute_dtypes: list[str] = Field(default_factory=list)
    storage_dtypes: list[str] = Field(default_factory=list)
    double_quantization: bool = False
    installed_version: str | None = None
    operation_capabilities: dict[str, Capability] = Field(default_factory=dict)
    calibration_required: bool = False


class OptimizationCapability(BaseModel):
    """One advanced optimization option exposed consistently to API and UI.

    ``configuration_schema`` is intentionally descriptive JSON rather than a
    second configuration model.  The authoritative values remain RunConfig;
    this shape lets clients render only controls which are safe for the active
    backend/runtime.
    """

    id: str
    label: str
    category: Literal["peft", "optimizer", "acceleration"]
    disclosure: Literal["recommended", "advanced", "expert"]
    support: Capability
    installed_version: str | None = None
    compatibility: dict[str, object] = Field(default_factory=dict)
    configuration_schema: dict[str, object] = Field(default_factory=dict)
    impact: dict[str, object] = Field(default_factory=dict)


class ServingProviderCapabilities(BaseModel):
    schema_version: Literal[1] = 1
    name: str
    display_name: str
    availability: Capability
    operations: dict[str, Capability]
    model_formats: list[str] = Field(default_factory=list)
    required_dependencies: list[str] = Field(default_factory=list)
    installed_version: str | None = None
    features: dict[str, Capability] = Field(default_factory=dict)
    configuration_schema: dict = Field(default_factory=dict)

    @property
    def enabled_operations(self) -> list[str]:
        return [name for name, capability in self.operations.items() if capability.allowed]
