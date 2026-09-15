"""System + hardware endpoints."""

from __future__ import annotations

from collections import deque

from fastapi import APIRouter

from app.config import get_settings
from app.core.cpu_jobs import cpu_jobs
from app.core.diagnostics import checks
from app.core.hardware import read_hardware
from app.core.observability import activity
from app.core.queue import queue
from app.db.migrate import DatabaseDiagnostics
from app.integrations import llmfit
from app.serving.providers import provider_statuses

router = APIRouter(prefix="/api/system", tags=["system"])
_settings = get_settings()


@router.get("/hardware")
def hardware():
    return read_hardware()


@router.get("/status")
def status():
    return {
        "llmfit_available": llmfit.llmfit_available(),
        "hf_token_set": bool(_settings.hf_token),
        "judge_configured": bool(_settings.judge_api_key),
        "current_run": queue.current_id,
        **{k: v for k, v in queue.snapshot().items() if k != "type"},
        "home": str(_settings.home),
        "vram_budget_mb": _settings.vram_budget_mb,
    }


@router.get("/diagnostics")
def diagnostics():
    return checks()


@router.get("/database", response_model=DatabaseDiagnostics)
def database():
    from app.db.migrate import database_diagnostics
    from app.db.session import engine

    return database_diagnostics(engine)


@router.get("/services")
async def services():
    from sqlalchemy import text

    from app.db.session import engine

    provider_states = await provider_statuses()
    resource = queue.snapshot().get("resource") or {"kind": "idle"}
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        database = {"status": "ready", "engine": "sqlite"}
    except Exception as exc:
        database = {"status": "error", "engine": "sqlite", "error": str(exc)}
    return {
        "api": {"status": "ready"},
        "database": database,
        "training_worker": {"status": "busy" if queue.current_id else "ready" if queue.healthy else "error", "run_id": queue.current_id},
        "gpu_resource": resource,
        "cpu_jobs": cpu_jobs.snapshot(),
        "providers": provider_states,
    }


@router.get("/activity")
def activity_events(after: int = 0, limit: int = 500):
    return {"events": activity.since(after, limit)}


@router.get("/logs")
def logs(lines: int = 200):
    """Tail of the backend log file, for the in-app log viewer."""
    from app.core.logging_config import log_file_path

    path = log_file_path()
    if not path.exists():
        return {"lines": [], "path": str(path)}
    # Read the last N lines without loading a huge file fully.
    count = max(1, min(lines, 2000))
    with path.open(encoding="utf-8", errors="replace") as stream:
        text = list(deque(stream, maxlen=count))
    return {"lines": [line.rstrip("\n") for line in text], "path": str(path)}
