"""Model Registry: local artifacts + HF-synced view, publish/import."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from app.config import get_settings
from app.core.log_capture import append_log_line, read_log_tail
from app.core.logging_config import get_logger
from app.core.ws import hub
from app.db.models import ModelArtifact, Run
from app.db.session import engine
from app.integrations import gguf, hf_hub

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
    }


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
    registered_run_ids = {a.run_id for a in artifacts if a.run_id}
    unregistered = [
        {"run_id": r.id, "name": r.name, "output_dir": r.output_dir,
         "hf_repo": r.hf_repo, "method": r.method, "base_model": r.base_model}
        for r in done_runs if r.id not in registered_run_ids and r.output_dir
    ]
    return {"artifacts": artifacts, "unpublished_runs": unregistered}


@router.get("/hf")
def hf_models():
    return {"models": hf_hub.list_my_models(), "token_set": bool(_settings.hf_token)}


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


class QuantizeBody(BaseModel):
    run_id: int
    quant_type: str = "q4_k_m"


@router.post("/quantize", status_code=202)
async def quantize(body: QuantizeBody):
    if body.quant_type not in gguf.QUANT_TYPES:
        raise HTTPException(400, f"Unknown quant type. Choose one of {gguf.QUANT_TYPES}.")
    with Session(engine) as db:
        run = db.get(Run, body.run_id)
        if not run:
            raise HTTPException(404, "Run not found")
        if not run.output_dir:
            raise HTTPException(400, "Run has no output directory to quantize.")
        src = f"{run.output_dir}/output"
        art = ModelArtifact(
            name=f"{run.name}-{body.quant_type}.gguf", kind="gguf", run_id=run.id,
            base_model=run.base_model, status="quantizing",
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
