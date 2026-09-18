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

from sqlmodel import Session, select

from app.config import get_settings
from app.core.errors import normalize_failure
from app.core.events import parse_event
from app.core.log_capture import append_log_line
from app.core.logging_config import get_logger
from app.core.ws import hub
from app.db.models import Checkpoint, ModelArtifact, Run
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
        await asyncio.to_thread(_terminate_process_tree, self.proc.pid)
        try:
            await asyncio.wait_for(self.proc.wait(), timeout=5)
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
    _write_manifest(run_id, phase="preparing", status=RunStatus.RUNNING.value)
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
    _write_manifest(run_id, phase="completed" if status == RunStatus.DONE else status.value, status=status.value)
    if status == RunStatus.FAILED:
        _write_failure_report(run_id)
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
        elif event.type == "warning":
            append_log_line(lpath, "warning", f"{event.code}: {event.message}")
        elif event.type == "error":
            append_log_line(lpath, "error", f"{event.code}: {event.message}")
        elif event.type == "artifact":
            _record_artifact(run_id, event.kind, event.path, event.metadata)
            append_log_line(lpath, "info", f"Artifact {event.kind}: {event.path}")
        elif event.type == "progress" and event.message:
            append_log_line(lpath, "info", event.message)
        elif event.type == "profile":
            append_log_line(
                lpath,
                "info",
                f"Profile {event.name}: {json.dumps(event.values, ensure_ascii=False, sort_keys=True)}",
            )

        if event.type == "status" and event.status == "failed" and event.detail:
            err_capture["msg"] = event.detail
        elif event.type == "error":
            err_capture["msg"] = event.detail or event.message
        elif event.type == "log" and event.level == "error":
            # Keep the most specific error line seen (e.g. the exception message).
            err_capture["msg"] = event.message.strip().splitlines()[-1] if event.message.strip() else err_capture.get("msg")


def _record_checkpoint(run_id: int, step: int, path: str, is_final: bool) -> None:
    with Session(engine) as db:
        existing = db.exec(
            select(Checkpoint).where(Checkpoint.run_id == run_id, Checkpoint.path == path)
        ).first()
        if existing:
            existing.step = step
            existing.is_final = existing.is_final or is_final
            db.add(existing)
        else:
            db.add(Checkpoint(run_id=run_id, step=step, path=path, is_final=is_final))
        db.commit()
    if is_final:
        _record_artifact(run_id, "model", path, {"source": "final_checkpoint"})


def _record_artifact(run_id: int, kind: str, path: str, metadata: dict | None = None) -> None:
    """Idempotently register a backend artifact using the existing lineage model."""
    with Session(engine) as db:
        run = db.get(Run, run_id)
        if not run:
            return
        existing = db.exec(
            select(ModelArtifact).where(
                ModelArtifact.run_id == run_id,
                ModelArtifact.local_path == path,
            )
        ).first()
        lineage = {
            "backend": run.backend,
            "method": run.method,
            "task": run.task,
            "base_model": run.base_model,
            "revision": run.config.get("revision"),
            "dataset_id": run.dataset_id,
            "reference": (run.config.get("alignment") or {}).get("reference"),
            **(metadata or {}),
        }
        effective_kind = (
            "adapter" if kind == "model" and run.method in {"lora", "qlora", "dora"} else kind
        )
        if existing:
            existing.kind = effective_kind
            existing.meta = {**(existing.meta or {}), **lineage}
            existing.status = "ready"
            db.add(existing)
        else:
            db.add(ModelArtifact(
                name=run.name,
                kind=effective_kind,
                run_id=run_id,
                base_model=run.base_model,
                local_path=path,
                status="ready",
                meta=lineage,
            ))
        db.commit()


def _terminate_process_tree(pid: int) -> None:
    """Terminate descendants before their worker parent so external engines cannot orphan."""
    try:
        import psutil

        parent = psutil.Process(pid)
        processes = [*reversed(parent.children(recursive=True)), parent]
        for process in processes:
            try:
                process.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(processes, timeout=3)
        for process in alive:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
    except Exception:
        return


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


def _redact(value):
    if isinstance(value, dict):
        credential_keys = {"token", "hf_token", "access_token", "auth_token", "api_token", "bearer_token", "hugging_face_hub_token"}
        return {key: ("***" if key.lower() in credential_keys or any(secret in key.lower() for secret in ("secret", "password", "api_key")) else _redact(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _write_manifest(run_id: int, *, phase: str, status: str) -> None:
    """Write a portable, atomically replaceable snapshot alongside each run."""
    try:
        with Session(engine) as db:
            run = db.get(Run, run_id)
            if not run:
                return
            checkpoints = db.exec(select(Checkpoint).where(Checkpoint.run_id == run_id)).all()
            payload = {
                "schema_version": 1,
                "run_id": run.id,
                "project_id": (run.config.get("extra") or {}).get("project_id"),
                "name": run.name,
                "task": run.task,
                "phase": phase,
                "status": status,
                "model": {"reference": run.base_model, "revision": run.config.get("revision")},
                "dataset_id": run.dataset_id,
                "backend": run.backend,
                "method": run.method,
                "configuration": _redact(run.config),
                "estimate": run.estimate,
                "hardware": run.hardware,
                "metrics": run.metrics,
                "checkpoints": [{"step": item.step, "path": item.path, "final": item.is_final} for item in checkpoints],
                "output_dir": run.output_dir,
                "error": run.error,
                "timestamps": {"created": run.created_at.isoformat(), "started": run.started_at.isoformat() if run.started_at else None,
                               "finished": run.finished_at.isoformat() if run.finished_at else None},
            }
        path = _settings.runs_dir / str(run_id) / "run.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        temporary.replace(path)
    except Exception as exc:  # noqa: BLE001 — manifest is best-effort observability
        # Observability output must never change the result of the ML workload.
        log.warning("run %s: could not write portable manifest: %s", run_id, exc)


def _write_failure_report(run_id: int) -> None:
    try:
        with Session(engine) as db:
            run = db.get(Run, run_id)
            if not run:
                return
            normalized = normalize_failure(run.error or "Training worker failed without a structured error.")
            report = {
                **normalized,
                "run_id": run_id,
                "stage": "training",
                "backend": run.backend,
                "configuration": _redact(run.config),
                "hardware": run.hardware,
                "recent_logs": [],
            }
        try:
            report["recent_logs"] = log_path(run_id).read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
        except OSError:
            pass
        workdir = _settings.runs_dir / str(run_id)
        (workdir / "failure.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        suggestions = "\n".join(f"- {item}" for item in report["suggestions"])
        (workdir / "failure.md").write_text(
            f"# Run {run_id} failure\n\n**Code:** `{report['code']}`  \n**Category:** {report['category']}  \n"
            f"**Stage:** training  \n**Backend:** {report['backend']}\n\n## Message\n\n{report['message']}\n\n"
            f"## Recommended fixes\n\n{suggestions}\n",
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001 — failure report must not hide the original failure
        log.warning("run %s: could not write failure report: %s", run_id, exc)
