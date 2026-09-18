"""Model Registry: local artifacts + HF-synced view, publish/import."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from app.config import get_settings
from app.core.deployment import manager as deployment_manager
from app.core.eval_manager import manager as eval_manager
from app.core.export_jobs import manager as export_jobs
from app.core.log_capture import read_log_tail
from app.core.logging_config import get_logger
from app.core.queue import queue
from app.core.resources import ResourceBusy
from app.db.models import ModelArtifact, Run
from app.db.session import engine
from app.export_contracts import ExportRequest, export_capabilities
from app.integrations import gguf, hf_hub, vllm_serve
from app.integrations.hf_hub import generate_model_card  # noqa: F401 -- retained public helper
from app.model_refs import (
    ModelReferenceError,
    model_options,
    normalize_artifact_kind,
    resolve_model_ref,
    run_ref,
)
from app.models.quantization_registry import quantization_capabilities

router = APIRouter(prefix="/api/registry", tags=["registry"])
_settings = get_settings()
log = get_logger(__name__)


def _gguf_log_path(art_id: int) -> Path:
    return _settings.runs_dir / "gguf" / str(art_id) / "output.log"


@router.get("/capabilities")
def capabilities():
    return {
        "gguf_available": gguf.llamacpp_available(),
        "hf_token_set": bool(_settings.hf_token),
        "quant_types": gguf.QUANT_TYPES,
        "deployment": deployment_manager.status(),
        "quantization": {name: cap.model_dump(mode="json") for name, cap in quantization_capabilities().items()},
        "export_targets": ["adapter", "huggingface", "merged", "quantized", "gguf", "ollama", "hub"],
    }


@router.get("/model-options")
def available_model_options():
    """Stable model choices for studios, Eval Lab, and automation clients."""
    return {"models": model_options()}


@router.post("/exports", status_code=202)
async def start_export(body: ExportRequest):
    try:
        return await export_jobs.start(body)
    except (ValueError, ResourceBusy) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/exports/{job_id}")
def export_status(job_id: int):
    with Session(engine) as db:
        artifact = db.get(ModelArtifact, job_id)
        if not artifact or not (artifact.meta or {}).get("export_job"):
            raise HTTPException(404, "Export job not found.")
        return {"artifact": artifact.model_dump(), "logs": export_jobs.logs(job_id)}


@router.post("/exports/{job_id}/cancel")
async def cancel_export(job_id: int):
    return {"cancelled": await export_jobs.cancel(job_id)}


@router.get("/local")
def local_models():
    with Session(engine) as db:
        artifacts = db.exec(select(ModelArtifact).order_by(ModelArtifact.created_at.desc())).all()
        # Finished runs whose output isn't yet registered as an artifact.
        done_runs = db.exec(
            select(Run).where(Run.status == "done").order_by(Run.created_at.desc())
        ).all()
    # Export/merge derivatives must not hide the source run's Publish action.
    registered_run_ids = {a.run_id for a in artifacts if a.run_id and a.published}
    unregistered = [
        {"run_id": r.id, "name": r.name, "output_dir": r.output_dir,
         "hf_repo": r.hf_repo, "method": r.method, "base_model": r.base_model,
         "model_ref": run_ref(r.id)}
        for r in done_runs if r.id not in registered_run_ids and r.output_dir
    ]
    lineage = [
        {
            "run_id": r.id,
            "name": r.name,
            "model_ref": run_ref(r.id),
            "parent_ref": r.base_model or None,
            "task": r.task,
            "method": r.method,
            "status": r.status,
            "dataset_id": r.dataset_id,
            "created_at": r.created_at,
        }
        for r in done_runs
    ]
    artifact_rows = []
    for artifact in artifacts:
        category = normalize_artifact_kind(artifact.kind)
        capabilities = (
            ["pairwise_accuracy", "chosen_score", "rejected_score", "reward_margin"]
            if category == "reward_model"
            else ["generation", "perplexity"] if category != "reference_model" else ["preference_log_probability"]
        )
        artifact_rows.append({
            **artifact.model_dump(),
            "model_category": category,
            "evaluation_capabilities": capabilities,
            "export_capabilities": {name: cap.model_dump(mode="json") for name, cap in export_capabilities(artifact).items()},
        })
    return {"artifacts": artifact_rows, "unpublished_runs": unregistered, "lineage": lineage}


@router.get("/hf")
def hf_models():
    return {"models": hf_hub.list_my_models(), "token_set": bool(_settings.hf_token)}


@router.delete("/artifacts/{artifact_id}")
def delete_artifact(artifact_id: int):
    with Session(engine) as db:
        artifact = db.get(ModelArtifact, artifact_id)
        if not artifact:
            raise HTTPException(404, "Model artifact not found.")
        if artifact.status in {"quantizing", "merging", "exporting"}:
            raise HTTPException(409, "Finish or cancel the export before deleting this artifact.")
        jobs = db.exec(select(ModelArtifact).where(ModelArtifact.status == "exporting")).all()
        if any((job.meta or {}).get("lineage", {}).get("source_artifact_id") == artifact_id for job in jobs):
            raise HTTPException(409, "An active export references this source artifact.")
        if deployment_manager.active and deployment_manager.status().get("model_ref") == artifact.local_path:
            raise HTTPException(409, "Stop the model server before deleting this artifact.")
        local_path = Path(artifact.local_path).resolve() if artifact.local_path else None
        models_root = _settings.models_dir.resolve()
        db.delete(artifact)
        db.commit()
    # Only remove files owned by the registry models directory. Run outputs
    # remain attached to their experiment even if their registry row is removed.
    if local_path and models_root in local_path.parents:
        if local_path.is_dir():
            shutil.rmtree(local_path)
        elif local_path.is_file():
            local_path.unlink()
    return {"deleted": artifact_id}


class PublishBody(BaseModel):
    run_id: int
    repo_id: str
    private: bool = True


@router.post("/publish")
async def publish(body: PublishBody):
    try:
        result = await export_jobs.start(ExportRequest(artifact_id=_export_source(run_ref(body.run_id)),
            target="hub", repo_id=body.repo_id, private=body.private))
        task = export_jobs.tasks.get(result["artifact_id"])
        if task:
            await asyncio.shield(task)
        with Session(engine) as db:
            artifact = db.get(ModelArtifact, result["artifact_id"])
            if artifact.status != "ready":
                raise HTTPException(502, artifact.error or "Hub export failed; inspect export job logs.")
            run = db.get(Run, body.run_id)
            run.hf_repo = artifact.hf_repo
            db.add(run)
            db.commit()
            return {"hf_repo": artifact.hf_repo, "artifact_id": artifact.id}
    except (ValueError, ModelReferenceError) as exc:
        raise HTTPException(422, str(exc)) from exc


class ImportBody(BaseModel):
    repo_id: str


@router.post("/import")
def import_model(body: ImportBody):
    import re

    if not re.fullmatch(r"[\w.\-]+/[\w.\-]+", body.repo_id):
        raise HTTPException(400, "repo_id must look like 'username/model-name'.")
    dest = _settings.models_dir / body.repo_id.replace("/", "__")
    try:
        path = hf_hub.import_model(body.repo_id, str(dest))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"Import failed: {e}")
    with Session(engine) as db:
        art = ModelArtifact(name=body.repo_id, kind="finetune", base_model=body.repo_id,
                            local_path=path, hf_repo=f"https://huggingface.co/{body.repo_id}",
                            published=True)
        db.add(art)
        db.commit()
        db.refresh(art)
    return {"artifact": art, "path": path}


class InspectBody(BaseModel):
    model_ref: str
    revision: str | None = None


@router.post("/inspect")
async def inspect_model(body: InspectBody):
    """Inspect architecture and compatibility without loading weights."""
    try:
        resolved = resolve_model_ref(body.model_ref)
    except ModelReferenceError as exc:
        raise HTTPException(400, str(exc))
    metadata = await asyncio.to_thread(hf_hub.inspect_model, resolved.load_ref, body.revision)
    adapter_base = resolved.base_model if resolved.kind == "adapter" else metadata.get("adapter_base_model")
    if metadata.get("kind") == "adapter" and adapter_base:
        base_metadata = await asyncio.to_thread(hf_hub.inspect_model, adapter_base, None)
        if base_metadata.get("reachable"):
            metadata = {
                **base_metadata,
                "model_ref": body.model_ref,
                "kind": "adapter",
                "adapter_base_model": adapter_base,
                "revision": body.revision,
                "warnings": metadata.get("warnings", []),
            }
    metadata["resolved"] = resolved.public()
    if not metadata.get("reachable"):
        raise HTTPException(422, detail={**metadata, "message": "Model metadata could not be loaded."})
    return metadata


class PreflightBody(BaseModel):
    model_ref: str
    revision: str | None = None
    backend: str = "unsloth"
    load_in_4bit: bool = False


@router.post("/preflight")
async def model_preflight(body: PreflightBody):
    """Run isolated subprocess runtime preflight holding GPU lease."""
    from app.core.preflight import preflight_runner
    from app.core.resources import ResourceBusy

    try:
        resolved = resolve_model_ref(body.model_ref)
    except ModelReferenceError as exc:
        raise HTTPException(400, str(exc))

    try:
        result = await preflight_runner.execute(
            model_ref=resolved.load_ref,
            revision=body.revision,
            backend=body.backend,
            load_in_4bit=body.load_in_4bit,
        )
    except ResourceBusy as exc:
        raise HTTPException(409, str(exc))

    result["model_ref"] = body.model_ref
    result["resolved"] = resolved.public()
    if not result.get("ok"):
        raise HTTPException(422, detail=result)
    return result


class DeployBody(BaseModel):
    model_ref: str


@router.get("/deployment")
def deployment_status():
    return deployment_manager.status()


@router.post("/deploy")
async def deploy(body: DeployBody):
    """Serve one model locally through /v1/chat/completions and /v1/completions.

    The server owns the GPU until stopped. Training/evaluation therefore refuse
    to start while it is active, and a queued training run stops it first.
    """
    try:
        resolved = resolve_model_ref(body.model_ref)
    except ModelReferenceError as exc:
        raise HTTPException(400, str(exc))
    if not resolved.deployable:
        raise HTTPException(422, f"{resolved.model_category.replace('_', ' ').title()} artifacts cannot be deployed for text generation.")
    if queue.current_id is not None:
        raise HTTPException(409, "GPU is busy with a training run.")
    current = eval_manager.current()
    if current and current.get("kind") in {"eval", "gen"}:
        raise HTTPException(409, "GPU is busy with an evaluation or playground request.")
    from app.serving.providers import external_gpu_owner

    owner = await external_gpu_owner()
    if owner:
        raise HTTPException(409, f"GPU is already in use by external provider '{owner}'. Stop it before deploying this model.")
    # A warm playground vLLM server is idle, so replace it rather than making
    # the user hunt for a separate stop control.
    if vllm_serve.manager.model is not None:
        await vllm_serve.manager.stop()
    try:
        status = await deployment_manager.start(resolved.requested_ref)
    except ResourceBusy as exc:
        raise HTTPException(409, str(exc))
    except (RuntimeError, TimeoutError) as exc:
        raise HTTPException(502, f"Deployment failed to start: {exc}")
    return status


@router.delete("/deployment")
async def stop_deployment():
    return {"stopped": await deployment_manager.stop()}


class MergeBody(BaseModel):
    model_ref: str
    name: str | None = None


def _export_source(model_ref: str) -> int:
    resolved = resolve_model_ref(model_ref)
    if not resolved.local_path:
        raise HTTPException(422, "Export requires a local artifact. Import the model first.")
    with Session(engine) as db:
        existing = db.exec(select(ModelArtifact).where(ModelArtifact.local_path == resolved.local_path, ModelArtifact.status == "ready")).first()
        if existing:
            return existing.id
        artifact = ModelArtifact(name=resolved.label, kind=resolved.artifact_kind or resolved.kind,
            local_path=resolved.local_path, run_id=resolved.run_id, base_model=resolved.base_model)
        db.add(artifact)
        db.commit()
        db.refresh(artifact)
        return artifact.id


@router.post("/merge", status_code=202)
async def merge_adapter(body: MergeBody):
    try:
        resolved = resolve_model_ref(body.model_ref)
        if resolved.kind != "adapter" or resolved.model_category == "reward_model":
            raise ValueError("Merge requires a causal-LM PEFT adapter.")
        result = await export_jobs.start(ExportRequest(artifact_id=_export_source(body.model_ref), target="merged"))
        if body.name:
            with Session(engine) as db:
                artifact = db.get(ModelArtifact, result["artifact_id"])
                artifact.name = body.name
                db.add(artifact)
                db.commit()
        return result
    except (ValueError, ModelReferenceError, ResourceBusy) as exc:
        raise HTTPException(422, str(exc)) from exc


class QuantizeBody(BaseModel):
    run_id: int | None = None
    artifact_id: int | None = None
    quant_type: str = "q4_k_m"


@router.post("/quantize", status_code=202)
async def quantize(body: QuantizeBody):
    try:
        source_id = body.artifact_id or (_export_source(run_ref(body.run_id)) if body.run_id else None)
        if not source_id:
            raise ValueError("Choose a completed run or artifact.")
        return await export_jobs.start(ExportRequest(artifact_id=source_id, target="gguf", quant_type=body.quant_type))
    except (ValueError, ModelReferenceError, ResourceBusy) as exc:
        raise HTTPException(422, str(exc)) from exc


async def stop_background_tasks() -> None:
    await export_jobs.shutdown()


@router.get("/quantize/{artifact_id}/logs")
def quantize_logs(artifact_id: int, lines: int = 2000):
    new_path = export_jobs.workdir(artifact_id) / "output.log"
    return {"lines": read_log_tail(new_path if new_path.exists() else _gguf_log_path(artifact_id), lines)}
