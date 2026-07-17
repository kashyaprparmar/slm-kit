"""System + hardware endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app.config import get_settings
from app.core.hardware import read_hardware
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
        "home": str(_settings.home),
        "vram_budget_mb": _settings.vram_budget_mb,
    }


@router.get("/logs")
def logs(lines: int = 200):
    """Tail of the backend log file, for the in-app log viewer."""
    from app.core.logging_config import log_file_path

    path = log_file_path()
    if not path.exists():
        return {"lines": [], "path": str(path)}
    # Read the last N lines without loading a huge file fully.
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return {"lines": text[-max(1, min(lines, 2000)):], "path": str(path)}
