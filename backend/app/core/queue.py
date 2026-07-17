"""Single-GPU job queue: exactly one training job runs at a time.

States: queued → running → (done | failed | cancelled). Queued jobs can be
cancelled instantly; the running job is cancelled by signaling the runner, which
frees VRAM before the next job starts.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from sqlmodel import Session, select

from app.core import runner
from app.core.hardware import poller
from app.core.logging_config import get_logger
from app.core.ws import hub
from app.db.models import Run
from app.db.session import engine
from app.domain import RunStatus
from app.integrations import vllm_serve

log = get_logger(__name__)


class JobQueue:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[int] = asyncio.Queue()
        self._worker: asyncio.Task | None = None
        self._current_id: int | None = None
        self._cancel_event: asyncio.Event | None = None
        # run_ids cancelled while still queued
        self._cancelled_queued: set[int] = set()

    # ---- lifecycle -----------------------------------------------------
    def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run_forever())

    async def stop(self) -> None:
        if self._worker:
            self._worker.cancel()
            self._worker = None

    async def recover_orphans(self) -> None:
        """Mark runs left RUNNING by a crash as failed (nothing is on the GPU)."""
        with Session(engine) as db:
            rows = db.exec(select(Run).where(Run.status == RunStatus.RUNNING.value)).all()
            for run in rows:
                run.status = RunStatus.FAILED.value
                run.error = "Interrupted by a backend restart (orphaned run)."
                run.finished_at = datetime.now(timezone.utc)
                db.add(run)
            # Re-enqueue anything still marked queued (collect ids before the
            # session closes — instances expire after commit).
            queued = db.exec(select(Run).where(Run.status == RunStatus.QUEUED.value)).all()
            queued_ids = [run.id for run in queued]
            db.commit()
        for run_id in queued_ids:
            await self._queue.put(run_id)

    # ---- public API ----------------------------------------------------
    async def enqueue(self, run_id: int) -> int:
        await self._queue.put(run_id)
        log.info("run %s enqueued (queue size=%s)", run_id, self._queue.qsize())
        await self._broadcast_queue()
        return self._queue.qsize()

    @property
    def current_id(self) -> int | None:
        return self._current_id

    def position(self, run_id: int) -> int | None:
        if run_id == self._current_id:
            return 0
        try:
            return list(self._queue._queue).index(run_id) + 1  # type: ignore[attr-defined]
        except ValueError:
            return None

    async def cancel(self, run_id: int) -> bool:
        if run_id == self._current_id and self._cancel_event is not None:
            self._cancel_event.set()
            return True
        # Queued but not started: tombstone it; the worker will skip it.
        if self.position(run_id) is not None:
            self._cancelled_queued.add(run_id)
            self._set_status(run_id, RunStatus.CANCELLED)
            await self._broadcast_queue()
            return True
        return False

    # ---- worker loop ---------------------------------------------------
    async def _run_forever(self) -> None:
        while True:
            run_id = await self._queue.get()
            if run_id in self._cancelled_queued:
                self._cancelled_queued.discard(run_id)
                self._queue.task_done()
                continue
            self._current_id = run_id
            self._cancel_event = asyncio.Event()
            poller.set_active(True)
            log.info("run %s dequeued — starting", run_id)
            # A warm vLLM server holds real VRAM even when idle — free it before
            # training claims the (single) GPU. Best-effort: never block a run
            # over a serving-side hiccup.
            if vllm_serve.manager.model is not None:
                log.info("run %s: stopping warm vLLM server to free VRAM for training", run_id)
                try:
                    await vllm_serve.manager.stop()
                except Exception:  # noqa: BLE001
                    log.exception("run %s: failed to stop vLLM server (continuing anyway)", run_id)
            await self._broadcast_queue()
            try:
                status = await runner.execute(run_id, self._cancel_event)
            except Exception as exc:  # noqa: BLE001 — never let the worker die
                self._set_status(run_id, RunStatus.FAILED, error=str(exc))
                await hub.publish(f"run:{run_id}",
                                  {"type": "status", "status": "failed", "detail": str(exc)})
            else:
                await hub.publish(f"run:{run_id}",
                                  {"type": "status", "status": status.value})
            finally:
                self._current_id = None
                self._cancel_event = None
                poller.set_active(False)
                self._queue.task_done()
                await self._broadcast_queue()

    # ---- helpers -------------------------------------------------------
    def _set_status(self, run_id: int, status: RunStatus, error: str | None = None) -> None:
        with Session(engine) as db:
            run = db.get(Run, run_id)
            if run:
                run.status = status.value
                if error:
                    run.error = error
                if status in (RunStatus.CANCELLED, RunStatus.FAILED, RunStatus.DONE):
                    run.finished_at = datetime.now(timezone.utc)
                db.add(run)
                db.commit()

    async def _broadcast_queue(self) -> None:
        await hub.publish("system", {
            "type": "queue",
            "current": self._current_id,
            "queued": list(self._queue._queue),  # type: ignore[attr-defined]
        })


queue = JobQueue()
