"""Model Registry: local artifacts + HF-synced view, publish/import."""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from app.config import get_settings
from app.core.deployment import manager as deployment_manager
from app.core.eval_manager import manager as eval_manager
from app.core.log_capture import append_log_line, read_log_tail
from app.core.logging_config import get_logger
from app.core.queue import queue
from app.core.resources import ResourceBusy, gpu
from app.core.ws import hub
from app.db.models import ModelArtifact, Run
from app.db.session import engine
from app.integrations import gguf, hf_hub, vllm_serve
from app.model_refs import ModelReferenceError, model_options, resolve_model_ref, run_ref

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
    }


@router.get("/model-options")
def available_model_options():
    """Stable model choices for studios, Eval Lab, and automation clients."""
    return {"models": model_options()}


def generate_model_card(run: Run) -> str:
    cfg = run.config or {}
    metrics = run.metrics or {}
    hw = run.hardware or {}
    lines = [
        f"# {run.name}",
        "",
        "Model produced with **SLM Kit**.",
        "",
        "## Training summary",
        f"- **Task:** {run.task}",
        f"- **Method:** {run.method}",
        f"- **Base model:** `{run.base_model}`",
        f"- **Backend:** {run.backend}",
        f"- **Trained:** {run.finished_at or run.started_at}",
    ]
    if hw.get("gpu_name"):
        lines.append(f"- **Hardware:** {hw.get('gpu_name')} ({hw.get('vram_total_mb')} MB VRAM)")
    lines += ["", "## Hyperparameters", "```json", _pretty(cfg), "```"]
    if metrics:
        lines += ["", "## Final metrics", "```json", _pretty(metrics), "```"]
    lines += ["", "_Generated automatically from the run's stored metadata._"]
    return "\n".join(lines)


def _pretty(obj) -> str:
    import json

    return json.dumps(obj, indent=2, default=str)


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
    return {"artifacts": artifacts, "unpublished_runs": unregistered, "lineage": lineage}


@router.get("/hf")
def hf_models():
    return {"models": hf_hub.list_my_models(), "token_set": bool(_settings.hf_token)}


@router.delete("/artifacts/{artifact_id}")
def delete_artifact(artifact_id: int):
    with Session(engine) as db:
        artifact = db.get(ModelArtifact, artifact_id)
        if not artifact:
            raise HTTPException(404, "Model artifact not found.")
        if artifact.status == "quantizing":
            raise HTTPException(409, "Wait for GGUF export to finish before deleting this artifact.")
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
def publish(body: PublishBody):
    with Session(engine) as db:
        run = db.get(Run, body.run_id)
        if not run:
            raise HTTPException(404, "Run not found")
        if not run.output_dir:
            raise HTTPException(400, "Run has no output directory to publish.")
        output = f"{run.output_dir}/output"
        card = generate_model_card(run)
    try:
        url = hf_hub.publish_model(output, body.repo_id, body.private, card)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"HF upload failed: {e}")

    with Session(engine) as db:
        run = db.get(Run, body.run_id)
        run.hf_repo = url
        db.add(run)
        db.add(ModelArtifact(
            name=run.name, kind=run.task, run_id=run.id, base_model=run.base_model,
            local_path=output, hf_repo=url, published=True,
            meta={"published_at": datetime.utcnow().isoformat()},
        ))
        db.commit()
    return {"hf_repo": url}


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
    if queue.current_id is not None:
        raise HTTPException(409, "GPU is busy with a training run.")
    current = eval_manager.current()
    if current and current.get("kind") in {"eval", "gen"}:
        raise HTTPException(409, "GPU is busy with an evaluation or playground request.")
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


@router.post("/merge", status_code=202)
async def merge_adapter(body: MergeBody):
    try:
        resolved = resolve_model_ref(body.model_ref)
    except ModelReferenceError as exc:
        raise HTTPException(400, str(exc))
    if resolved.kind != "adapter":
        raise HTTPException(422, "Only a PEFT/LoRA adapter can be merged. Full models are already standalone.")
    try:
        lease = gpu.acquire("merging", body.model_ref)
    except ResourceBusy as exc:
        raise HTTPException(409, str(exc))
    try:
        with Session(engine) as db:
            artifact = ModelArtifact(
                name=body.name or f"{resolved.label}-merged",
                kind="merged",
                run_id=resolved.run_id,
                base_model=resolved.base_model,
                status="merging",
                meta={"source_ref": body.model_ref},
            )
            db.add(artifact)
            db.commit()
            db.refresh(artifact)
            artifact_id = artifact.id
    except BaseException:
        gpu.release(lease)
        raise
    task = asyncio.create_task(_run_merge(artifact_id, body.model_ref, lease))
    _BACKGROUND_TASKS.add(task)
    _MERGE_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    task.add_done_callback(_MERGE_TASKS.discard)
    return {"artifact_id": artifact_id, "status": "merging"}


class QuantizeBody(BaseModel):
    run_id: int | None = None
    artifact_id: int | None = None
    quant_type: str = "q4_k_m"


@router.post("/quantize", status_code=202)
async def quantize(body: QuantizeBody):
    if body.quant_type not in gguf.QUANT_TYPES:
        raise HTTPException(400, f"Unknown quant type. Choose one of {gguf.QUANT_TYPES}.")
    with Session(engine) as db:
        run = db.get(Run, body.run_id) if body.run_id else None
        source_artifact = db.get(ModelArtifact, body.artifact_id) if body.artifact_id else None
        if not run and not source_artifact:
            raise HTTPException(404, "Choose a completed run or merged artifact to quantize.")
        if source_artifact:
            if source_artifact.kind not in {"merged", "pretrain"} or source_artifact.status != "ready" or not source_artifact.local_path:
                raise HTTPException(422, "GGUF export requires a ready full or merged model artifact.")
            src = source_artifact.local_path
            source_name = source_artifact.name
            source_run_id = source_artifact.run_id
            source_base = source_artifact.base_model
        else:
            assert run is not None
            if run.method != "full" and run.task != "pretrain":
                raise HTTPException(422, "Merge this adapter first, then export the merged artifact to GGUF.")
            if not run.output_dir:
                raise HTTPException(400, "Run has no output directory to quantize.")
            src = f"{run.output_dir}/output"
            source_name = run.name
            source_run_id = run.id
            source_base = run.base_model
        art = ModelArtifact(
            name=f"{source_name}-{body.quant_type}.gguf", kind="gguf", run_id=source_run_id,
            base_model=source_base, status="quantizing",
            meta={"quant_type": body.quant_type, "source": src},
        )
        db.add(art)
        db.commit()
        db.refresh(art)
        art_id = art.id

    # Non-GPU, potentially long: run off the request path. Not on the GPU queue.
    # Held in _BACKGROUND_TASKS (not a bare asyncio.create_task) — an
    # unreferenced task can be garbage collected mid-run.
    task = asyncio.create_task(_run_quantize(art_id, src, body.quant_type))
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    return {"artifact_id": art_id, "status": "quantizing"}


_BACKGROUND_TASKS: set[asyncio.Task] = set()
_MERGE_TASKS: set[asyncio.Task] = set()


async def stop_background_tasks() -> None:
    tasks = list(_MERGE_TASKS)
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def _run_merge(artifact_id: int, model_ref: str, lease: str) -> None:
    output_dir = _settings.models_dir / f"merged-{artifact_id}"
    workdir = _settings.runs_dir / "merge" / str(artifact_id)
    workdir.mkdir(parents=True, exist_ok=True)
    config_path = workdir / "config.json"
    log_file = workdir / "output.log"
    config_path.write_text(json.dumps({"model_ref": model_ref, "output_dir": str(output_dir)}), encoding="utf-8")
    error = None
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-u", "-m", "app.train_entry.merge", str(config_path),
            cwd=str(Path(__file__).resolve().parents[2]),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").strip()
            if line:
                append_log_line(log_file, "error" if line.startswith("ERROR:") else "info", line)
                await hub.publish(f"merge:{artifact_id}", {"type": "log", "message": line})
        await proc.wait()
        if proc.returncode:
            error = f"Merge subprocess exited with code {proc.returncode}."
    except asyncio.CancelledError:
        error = "Merge interrupted by backend shutdown."
        if proc is not None and proc.returncode is None:
            proc.kill()
            await proc.wait()
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
    finally:
        gpu.release(lease)
    if error:
        _update_artifact(artifact_id, status="failed", error=error)
        await hub.publish(f"merge:{artifact_id}", {"type": "status", "status": "failed", "detail": error})
    else:
        _update_artifact(artifact_id, status="ready", local_path=str(output_dir))
        await hub.publish(f"merge:{artifact_id}", {"type": "status", "status": "ready"})


@router.get("/quantize/{artifact_id}/logs")
def quantize_logs(artifact_id: int, lines: int = 2000):
    """Full persisted log history for a GGUF export job — works after it
    finishes too, not just while /ws/gguf/{id} is streaming it live."""
    return {"lines": read_log_tail(_gguf_log_path(artifact_id), lines)}


async def _run_quantize(art_id: int, src: str, quant_type: str) -> None:
    out_dir = _settings.models_dir / f"gguf-{art_id}"
    topic = f"gguf:{art_id}"
    lpath = _gguf_log_path(art_id)
    loop = asyncio.get_running_loop()

    def on_line(line: str) -> None:
        # Called from the worker thread `asyncio.to_thread` runs `gguf.quantize`
        # on — never touch the event loop directly from here. Persisting to
        # disk is thread-safe (plain file append); publishing to the WS hub
        # must be scheduled back onto the loop via run_coroutine_threadsafe.
        append_log_line(lpath, "info", line)
        asyncio.run_coroutine_threadsafe(hub.publish(topic, {"type": "log", "message": line}), loop)

    append_log_line(lpath, "info", f"Starting GGUF export ({quant_type}) for artifact {art_id} from {src}")
    await hub.publish(topic, {"type": "log", "message": f"Starting GGUF export ({quant_type})…"})
    try:
        path = await asyncio.to_thread(gguf.quantize, src, str(out_dir), quant_type, on_line)
        append_log_line(lpath, "info", f"Quantize complete → {path}")
        _update_artifact(art_id, status="ready", local_path=path)
        await hub.publish(topic, {"type": "status", "status": "ready"})
        log.info("gguf quantize %s complete: %s", art_id, path)
    except Exception as e:  # noqa: BLE001 — surfaced on the artifact row
        append_log_line(lpath, "error", str(e))
        _update_artifact(art_id, status="failed", error=str(e))
        await hub.publish(topic, {"type": "status", "status": "failed", "detail": str(e)})
        log.warning("gguf quantize %s failed: %s", art_id, e)
    finally:
        await hub.publish(topic, {"type": "done"})


def _update_artifact(art_id: int, **fields) -> None:
    with Session(engine) as db:
        art = db.get(ModelArtifact, art_id)
        if art:
            for k, v in fields.items():
                setattr(art, k, v)
            db.add(art)
            db.commit()
