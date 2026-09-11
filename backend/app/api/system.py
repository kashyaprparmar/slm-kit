"""System + hardware endpoints."""

from __future__ import annotations

from collections import deque

from fastapi import APIRouter

from app.config import get_settings
from app.core.diagnostics import checks
from app.core.hardware import read_hardware
from app.core.observability import activity
from app.core.queue import queue
from app.integrations import llmfit

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
