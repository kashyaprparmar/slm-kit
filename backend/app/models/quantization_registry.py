"""Operation-specific quantization support; importing this never loads ML packages."""
from importlib import metadata, util

from app.capabilities import Capability, QuantizationCapability, SupportState


def quantization_capabilities() -> dict[str, QuantizationCapability]:
    result = {}
    for name, package, module, calibration in (
        ("bitsandbytes", "bitsandbytes", "bitsandbytes", False),
        ("gptq", "gptqmodel", "gptqmodel", True),
        ("awq", "autoawq", "awq", True),
        ("hqq", "hqq", "hqq", False),
        ("eetq", "eetq", "eetq", False),
        ("aqlm", "aqlm", "aqlm", True),
    ):
        try:
            installed = util.find_spec(module) is not None
            version = metadata.version(package) if installed else None
        except (ImportError, ValueError, metadata.PackageNotFoundError):
            installed, version = False, None
        operations = {}
        for operation in ("training", "loading", "inference", "export", "merging"):
            state = SupportState.NOT_INSTALLED if not installed else SupportState.UNSUPPORTED
            reason = f"Install optional {package}." if not installed else "No verified SLM Kit execution path for this operation."
            if operation == "merging":
                state, reason = SupportState.UNSUPPORTED, "Merge adapters into an unquantized base before quantization."
            elif installed and (operation in ("loading", "inference") or (name == "bitsandbytes" and operation == "training")):
                state, reason = SupportState.EXPERIMENTAL, "Requires compatible model, Transformers integration and worker hardware validation."
            operations[operation] = Capability(state=state, reason=reason, requirements=[package])
        result[name] = QuantizationCapability(
            format=name, support=operations["loading"], installed_version=version,
            operations=[key for key, cap in operations.items() if cap.allowed],
            operation_capabilities=operations, calibration_required=calibration,
        )
    return result
