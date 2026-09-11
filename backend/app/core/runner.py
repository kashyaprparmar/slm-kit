"""Supervises a single training run as an isolated subprocess.

Why a subprocess: killing the child returns VRAM to the OS immediately and
completely — the only reliable way to honor "clean cancel that frees VRAM". The
child speaks newline-delimited JSON (see ``events.py``) on stdout; we parse each
line, persist metrics/checkpoints, and rebroadcast over the run's WebSocket topic.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from sqlmodel import Session

from app.config import get_settings
from app.core.events import parse_event
from app.core.log_capture import append_log_line
from app.core.logging_config import get_logger
from app.core.ws import hub
from app.db.models import Checkpoint, Run
from app.db.session import engine
from app.domain import RunConfig, RunStatus

_settings = get_settings()
log = get_logger(__name__)


def _now() -> datetime:
    return datetime.now(UTC)


def _topic(run_id: int) -> str:
    return f"run:{run_id}"


def log_path(run_id: int) -> Path:
    return _settings.runs_dir / str(run_id) / "output.log"


class RunHandle:
    def __init__(self, run_id: int, proc: asyncio.subprocess.Process, workdir: Path):
        self.run_id = run_id
        self.proc = proc
        self.workdir = workdir

    async def stop(self, graceful: bool = True) -> None:
        # Ask the backend to checkpoint-and-exit first — but only give it a short
        # window. A training loop honors the STOP sentinel on the next step
        # (~1-2s for QLoRA); if it's stuck loading a model, escalate fast.
        if graceful:
            (self.workdir / "STOP").touch()
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=5)
                return
            except TimeoutError:
                pass
        # ...then escalate to termination to guarantee VRAM release.
        # terminate() is cross-platform (SIGTERM on POSIX, TerminateProcess on
        # Windows; send_signal(SIGTERM) is unreliable on Windows).
        try:
            self.proc.terminate()
            await asyncio.wait_for(self.proc.wait(), timeout=3)
        except (TimeoutError, ProcessLookupError):
            try:
                self.proc.kill()
                await self.proc.wait()
            except ProcessLookupError:
                pass


async def execute(run_id: int, cancel_event: asyncio.Event) -> RunStatus:
    """Run one job to completion. Returns the terminal status."""
    workdir = _settings.runs_dir / str(run_id)
    checkpoint_dir = workdir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with Session(engine) as db:
        run = db.get(Run, run_id)
        assert run is not None
        run.status = RunStatus.RUNNING.value
        run.started_at = _now()
        run.output_dir = str(workdir)
        cfg = RunConfig(**run.config)
        db.add(run)
        db.commit()

    # Persist config for the subprocess entrypoint.
    (workdir / "config.json").write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    # Clear any stale stop sentinel from a previous run of this id.
    stop_flag = workdir / "STOP"
    if stop_flag.exists():
        stop_flag.unlink()

    await hub.publish(_topic(run_id), {"type": "status", "status": "running"})

    # -u = unbuffered: stream every line (incl. verbose torch/unsloth library
    # logs) live to the UI, instead of in delayed chunks.
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-u", "-m", "app.train_entry.run", str(run_id),
        cwd=str(Path(__file__).resolve().parents[2]),  # backend/ dir
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    handle = RunHandle(run_id, proc, workdir)

    err_capture: dict = {}
    cancel_task = asyncio.create_task(cancel_event.wait())
    read_task = asyncio.create_task(_pump_stdout(run_id, proc, err_capture))

    try:
        done, _ = await asyncio.wait({cancel_task, read_task}, return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError:
        await handle.stop(graceful=False)
        cancel_task.cancel()
        await asyncio.gather(read_task, cancel_task, return_exceptions=True)
        _finalize(run_id, proc.returncode, True)
        raise

    cancelled = cancel_task in done and cancel_event.is_set()
    if cancelled:
        log.info("run %s: cancellation requested — stopping subprocess", run_id)
        msg = "Cancellation requested — stopping and freeing VRAM…"
        await hub.publish(_topic(run_id), {"type": "log", "level": "warning", "message": msg})
        append_log_line(log_path(run_id), "warning", msg)
        await handle.stop(graceful=True)
    cancel_task.cancel()

    await proc.wait()
    await read_task  # drain remaining stdout

    status = _finalize(run_id, proc.returncode, cancelled, err_capture.get("msg"))
    log.info("run %s finished: %s (exit=%s)", run_id, status.value, proc.returncode)
    append_log_line(log_path(run_id), "info", f"Run finished: {status.value} (exit code {proc.returncode})")
    return status


async def _pump_stdout(run_id: int, proc: asyncio.subprocess.Process, err_capture: dict) -> None:
    assert proc.stdout is not None
    lpath = log_path(run_id)
    async for raw in proc.stdout:
        line = raw.decode(errors="replace").rstrip("\n")
        if not line:
            continue
        event = parse_event(line)
        if event is None:
            # Stray library output (torch/unsloth/transformers logging etc.) →
            # surface as a plain log line. This is most of what makes the Logs
            # panel comprehensive during training.
            await hub.publish(_topic(run_id), {"type": "log", "level": "info", "message": line})
            append_log_line(lpath, "info", line)
            continue
        await hub.publish(_topic(run_id), event.model_dump())
        if event.type == "checkpoint":
            _record_checkpoint(run_id, event.step, event.path, event.is_final)
            append_log_line(lpath, "info",
                             f"Checkpoint saved at step {event.step}: {event.path}" + (" (final)" if event.is_final else ""))
        elif event.type == "metric":
            _record_last_metric(run_id, event.metrics)
            _append_metric_history(run_id, event.step, event.metrics)
            m = event.metrics
            bits = ", ".join(f"{k}={v}" for k, v in m.items() if k in ("loss", "tokens_per_sec", "learning_rate", "grad_norm"))
            step_str = f"step {event.step}" + (f"/{event.total_steps}" if event.total_steps else "")
            append_log_line(lpath, "info", f"{step_str}: {bits}" if bits else f"{step_str}: metric update")
        elif event.type == "sample":
            append_log_line(lpath, "info", f"Sample generated at step {event.step} ({len(event.text)} chars)")
        elif event.type == "status":
            level = "error" if event.status == "failed" else "warning" if event.status == "cancelled" else "info"
            append_log_line(lpath, level, f"Status: {event.status}" + (f" — {event.detail}" if event.detail else ""))
        elif event.type == "log":
            append_log_line(lpath, event.level, event.message)

        if event.type == "status" and event.status == "failed" and event.detail:
            err_capture["msg"] = event.detail
        elif event.type == "log" and event.level == "error":
            # Keep the most specific error line seen (e.g. the exception message).
            err_capture["msg"] = event.message.strip().splitlines()[-1] if event.message.strip() else err_capture.get("msg")


def _record_checkpoint(run_id: int, step: int, path: str, is_final: bool) -> None:
    with Session(engine) as db:
        db.add(Checkpoint(run_id=run_id, step=step, path=path, is_final=is_final))
        db.commit()


def _record_last_metric(run_id: int, metrics: dict) -> None:
    with Session(engine) as db:
        run = db.get(Run, run_id)
        if run:
            run.metrics = {**(run.metrics or {}), **metrics}
            db.add(run)
            db.commit()


def _append_metric_history(run_id: int, step: int, metrics: dict) -> None:
    """Append one flattened metric row to the run's metrics.jsonl for later replot."""
    path = _settings.runs_dir / str(run_id) / "metrics.jsonl"
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")
    except OSError:
        pass


def _finalize(run_id: int, returncode: int | None, cancelled: bool, error: str | None = None) -> RunStatus:
    if cancelled:
        status = RunStatus.CANCELLED
    elif returncode == 0:
        status = RunStatus.DONE
    else:
        status = RunStatus.FAILED
    with Session(engine) as db:
        run = db.get(Run, run_id)
        if run:
            run.status = status.value
            run.finished_at = _now()
            if status == RunStatus.FAILED and not run.error:
                run.error = error or f"Training subprocess exited with code {returncode}"
            db.add(run)
            db.commit()
    return status
