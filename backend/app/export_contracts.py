"""Validated export requests shared by the API and isolated worker."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.capabilities import Capability, SupportState


def export_capabilities(artifact) -> dict[str, Capability]:
    """The same artifact gates drive validation and Registry controls."""
    path = Path(artifact.local_path) if artifact.local_path else None
    ready = artifact.status == "ready" and path is not None and path.exists()
    adapter = ready and path.is_dir() and (path / "adapter_config.json").exists()
    full = ready and path.is_dir() and (path / "config.json").exists() and not adapter
    gguf = ready and path.is_file() and path.suffix.lower() == ".gguf"
    from app.integrations.gguf import _convert_script
    from app.integrations.hf_hub import _token

    choices = {
        "adapter": (adapter, "Requires a ready local adapter."),
        "huggingface": (full, "Requires a ready full Hugging Face model."),
        "merged": (
            adapter and artifact.kind not in ("reward_model", "reference_model"),
            "Requires a causal-LM adapter and a compatible unquantized base.",
        ),
        "gguf": (
            full and bool(_convert_script()),
            "Requires a full model and optional llama.cpp converter; quantizer also required for low-bit output.",
        ),
        "quantized": (
            full and bool(_convert_script()),
            "Only verified GGUF conversion is enabled.",
        ),
        "ollama": (gguf, "Requires a ready GGUF artifact."),
        "hub": (
            ready and bool(_token()),
            "Requires a ready local artifact and configured HF token.",
        ),
    }
    return {
        name: Capability(
            state=SupportState.SUPPORTED if allowed else SupportState.UNSUPPORTED,
            reason="Available; request-specific checks still apply." if allowed else reason,
        )
        for name, (allowed, reason) in choices.items()
    }


class CalibrationMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_version_id: int = Field(ge=1)
    fingerprint: str = Field(min_length=1)
    samples: int = Field(ge=1)
    sequence_length: int = Field(ge=1)
    seed: int = 42


class OllamaExportOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chat_template: str | None = Field(default=None, max_length=100_000)
    stop_tokens: list[str] | None = Field(default=None, max_length=32)
    generation_defaults: dict[str, float | int] | None = None

    @model_validator(mode="after")
    def validate_values(self):
        if self.chat_template is not None and not self.chat_template.strip():
            self.chat_template = None
        for token in self.stop_tokens or []:
            if not token or "\n" in token or '"' in token:
                raise ValueError(
                    "Ollama stop tokens must be non-empty single-line strings without quotes."
                )
        allowed = {"temperature", "top_p", "top_k", "repeat_penalty", "num_predict"}
        if self.generation_defaults and set(self.generation_defaults) - allowed:
            raise ValueError("Unsupported Ollama generation default.")
        return self


class ExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artifact_id: int = Field(ge=1)
    target: Literal["adapter", "huggingface", "merged", "quantized", "gguf", "ollama", "hub"]
    quant_type: Literal["q4_k_m", "q5_k_m", "q8_0", "f16"] = "q4_k_m"
    quantization: str | None = None
    base_revision: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{40}$")
    calibration: CalibrationMetadata | None = None
    repo_id: str | None = Field(default=None, pattern=r"^[\w.-]+/[\w.-]+$")
    private: bool = True
    ollama: OllamaExportOptions | None = None

    @model_validator(mode="after")
    def validate_destination(self):
        if self.target == "hub" and not self.repo_id:
            raise ValueError("Hub exports require an explicit repository ID.")
        if self.target == "quantized" and self.quantization not in (None, "gguf"):
            raise ValueError(
                "Only the verified GGUF quantization exporter is available; other methods are loading/training capabilities."
            )
        return self
