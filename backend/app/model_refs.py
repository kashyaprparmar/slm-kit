"""Stable, lightweight references for models used across the product.

The UI should never need to know whether a model is an HF repo, a local full
checkpoint, a PEFT adapter, or a model produced by the scratch backend.  This
module resolves that detail without importing torch/transformers, so it is safe
to use from request handlers, fit validation, and subprocess entrypoints.

``run:<id>`` is the canonical reference for a completed SLM Kit run.  Plain HF
repository ids and local paths remain valid for backwards compatibility.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from sqlmodel import Session, select

from app.db.models import ModelArtifact, Run
from app.db.session import engine


class ModelReferenceError(ValueError):
    """A user-visible model reference could not be resolved safely."""


@dataclass(frozen=True)
class ResolvedModel:
    requested_ref: str
    load_ref: str
    kind: str  # transformers | adapter | scratch
    label: str
    base_model: str | None = None
    adapter_path: str | None = None
    run_id: int | None = None
    local_path: str | None = None

    @property
    def deployable(self) -> bool:
        # The bundled transformers server supports every format we can resolve.
        return True

    def public(self) -> dict:
        data = asdict(self)
        data["deployable"] = self.deployable
        return data


def run_ref(run_id: int) -> str:
    return f"run:{run_id}"


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelReferenceError(f"Could not read model metadata at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ModelReferenceError(f"Model metadata at {path} is not a JSON object.")
    return value


def _classify_path(
    path: Path,
    requested_ref: str,
    *,
    label: str | None = None,
    base_model: str | None = None,
    run_id: int | None = None,
) -> ResolvedModel:
    if not path.is_dir():
        raise ModelReferenceError(f"Local model directory does not exist: {path}")

    if (path / "model.pt").is_file() and (path / "arch.json").is_file():
        if not ((path / "vocab.json").is_file() and (path / "merges.txt").is_file()):
            raise ModelReferenceError(
                f"Scratch checkpoint at {path} is incomplete (expected vocab.json and merges.txt)."
            )
        return ResolvedModel(
            requested_ref=requested_ref,
            load_ref=str(path),
            kind="scratch",
            label=label or path.name,
            base_model=None,
            run_id=run_id,
            local_path=str(path),
        )

    adapter_config = path / "adapter_config.json"
    if adapter_config.is_file():
        cfg = _read_json(adapter_config)
        adapter_base = cfg.get("base_model_name_or_path") or base_model
        if not isinstance(adapter_base, str) or not adapter_base.strip():
            raise ModelReferenceError(
                f"Adapter at {path} has no base_model_name_or_path in adapter_config.json."
            )
        return ResolvedModel(
            requested_ref=requested_ref,
            load_ref=str(path),
            kind="adapter",
            label=label or path.name,
            base_model=adapter_base,
            adapter_path=str(path),
            run_id=run_id,
            local_path=str(path),
        )

    if not (path / "config.json").is_file():
        raise ModelReferenceError(f"No config.json, adapter_config.json or scratch checkpoint found in {path}.")
    if not any(path.glob("*.safetensors")) and not any(path.glob("pytorch_model*.bin")):
        raise ModelReferenceError(f"No model weights found in {path}.")
    return ResolvedModel(
        requested_ref=requested_ref,
        load_ref=str(path),
        kind="transformers",
        label=label or path.name,
        base_model=base_model,
        run_id=run_id,
        local_path=str(path),
    )


def _resolve_run(run_id: int, requested_ref: str) -> ResolvedModel:
    with Session(engine) as db:
        run = db.get(Run, run_id)
        if run is None:
            raise ModelReferenceError(f"Run {run_id} was not found.")
        if run.status != "done":
            raise ModelReferenceError(f"Run {run_id} is {run.status}; only completed runs can be loaded.")
        if not run.output_dir:
            raise ModelReferenceError(f"Run {run_id} has no output directory.")
        output = Path(run.output_dir) / "output"
        return _classify_path(
            output,
            requested_ref,
            label=run.name,
            base_model=run.base_model or None,
            run_id=run.id,
        )


def resolve_model_ref(model_ref: str) -> ResolvedModel:
    """Resolve a run reference, local checkpoint, or HF model/adapter reference.

    Remote HF references intentionally remain lightweight: their adapter metadata
    is loaded later by the runtime loader.  This keeps API validation offline
    safe and avoids an unexpected network request on every UI keystroke.
    """
    ref = model_ref.strip()
    if not ref:
        raise ModelReferenceError("A model reference is required.")
    if ref.startswith("run:"):
        raw_id = ref.removeprefix("run:")
        if not raw_id.isdigit() or int(raw_id) < 1:
            raise ModelReferenceError("Run references must look like 'run:123'.")
        return _resolve_run(int(raw_id), ref)

    path = Path(ref).expanduser()
    if path.exists():
        return _classify_path(path.resolve(), ref)
    if path.is_absolute() or ref.startswith((".", "~", "\\")) or ":" in ref:
        raise ModelReferenceError(f"Local model path does not exist: {ref}. Docker paths must exist inside the backend container.")
    if not re.fullmatch(r"[\w.-]+(?:/[\w.-]+)?", ref):
        raise ModelReferenceError("Use a Hugging Face repository ID, a local model folder, or run:123.")

    # A remote repo could itself be a PEFT adapter; the runtime probes that
    # after it is allowed to import huggingface_hub/transformers.
    return ResolvedModel(
        requested_ref=ref,
        load_ref=ref,
        kind="transformers",
        label=ref,
        base_model=None,
    )


def model_options() -> list[dict]:
    """Return every locally-known, loadable model once, newest first."""
    options: list[dict] = []
    seen: set[str] = set()
    with Session(engine) as db:
        runs = db.exec(select(Run).where(Run.status == "done").order_by(Run.created_at.desc())).all()
        artifacts = db.exec(select(ModelArtifact).order_by(ModelArtifact.created_at.desc())).all()

    for run in runs:
        ref = run_ref(run.id)
        try:
            item = _resolve_run(run.id, ref).public()
        except ModelReferenceError:
            continue
        item.update({"ref": ref, "source": "run", "task": run.task, "method": run.method})
        options.append(item)
        seen.add(ref)
        seen.add(item["load_ref"])

    for artifact in artifacts:
        if artifact.status != "ready" or not artifact.local_path:
            continue
        # GGUF is deployable via llama.cpp/Ollama, but not through the bundled
        # transformers endpoint. It remains visible in the artifact list.
        if artifact.kind == "gguf":
            continue
        ref = artifact.local_path
        if ref in seen:
            continue
        try:
            item = resolve_model_ref(ref).public()
        except ModelReferenceError:
            continue
        item.update({"ref": ref, "source": "artifact", "artifact_id": artifact.id})
        options.append(item)
        seen.add(ref)
    return options
