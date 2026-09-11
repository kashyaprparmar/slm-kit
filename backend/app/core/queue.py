"""FIFO training queue sharing atomic GPU leases with evaluation and serving."""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from sqlmodel import Session, select

from app.core import runner
from app.core.hardware import poller
from app.core.logging_config import get_logger
from app.core.resources import ResourceBusy, gpu
from app.core.ws import hub
from app.db.models import EvalResult, ModelArtifact, Run
from app.db.session import engine
from app.domain import RunStatus

log = get_logger(__name__)

class JobQueue:
    def __init__(self):
        self._queue: asyncio.Queue[int] = asyncio.Queue()
        self._pending: list[int] = []
        self._worker: asyncio.Task | None = None
        self._current_id: int | None = None
        self._cancel_event: asyncio.Event | None = None

    def start(self):
        if self._worker is None:
            self._worker = asyncio.create_task(self._run_forever())

    async def stop(self):
        if self._worker:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
            self._worker = None

    async def recover_orphans(self):
        with Session(engine) as db:
            for run in db.exec(select(Run).where(Run.status == "running")).all():
                run.status = "failed"
                run.error = "Interrupted by a backend restart."
                run.finished_at = datetime.now(UTC)
                db.add(run)
            for result in db.exec(select(EvalResult)).all():
                if (result.detail or {}).get("status") == "running":
                    result.detail = {**result.detail, "status": "failed", "error": "Interrupted by a backend restart."}
                    db.add(result)
            for artifact in db.exec(select(ModelArtifact)).all():
                if artifact.status in {"quantizing", "merging"}:
                    artifact.status = "failed"
                    artifact.error = "Interrupted by a backend restart. Start the operation again."
                    db.add(artifact)
            ids = [r.id for r in db.exec(select(Run).where(Run.status == "queued").order_by(Run.created_at)).all()]
            db.commit()
        for run_id in ids:
            await self.enqueue(run_id)

    async def enqueue(self, run_id):
        if run_id not in self._pending and run_id != self._current_id:
            self._pending.append(run_id)
            self._queue.put_nowait(run_id)
            log.info("Training run %s queued", run_id)
        await self._broadcast_queue()
        return len(self._pending)

    @property
    def current_id(self):
        return self._current_id

    def snapshot(self):
        return {"type": "queue", "current": self._current_id, "queued": list(self._pending),
                "resource": gpu.snapshot()}

    def position(self, run_id):
        if run_id == self._current_id:
            return 0
        return self._pending.index(run_id) + 1 if run_id in self._pending else None

    async def cancel(self, run_id):
        if run_id == self._current_id and self._cancel_event:
            self._cancel_event.set()
            return True
        if run_id in self._pending:
            self._pending.remove(run_id)
            self._set_status(run_id, RunStatus.CANCELLED)
            await hub.publish(f"run:{run_id}", {"type": "status", "status": "cancelled"})
            await self._broadcast_queue()
            return True
        return False

    async def _run_forever(self):
        while True:
            run_id = await self._queue.get()
            lease = None
            try:
                while run_id in self._pending:
                    try:
                        lease = gpu.acquire("training", run_id)
                        break
                    except ResourceBusy:
                        await asyncio.sleep(0.25)
                if lease is None:
                    continue
                self._pending.remove(run_id)
                self._current_id = run_id
                self._cancel_event = asyncio.Event()
                poller.set_active(True)
                await self._broadcast_queue()
                log.info("Training run %s started", run_id)
                status = await runner.execute(run_id, self._cancel_event)
                await hub.publish(f"run:{run_id}", {"type": "status", "status": status.value})
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.exception("Training run %s failed", run_id)
                self._set_status(run_id, RunStatus.FAILED, str(exc))
                await hub.publish(f"run:{run_id}", {"type": "status", "status": "failed", "detail": str(exc)})
            finally:
                gpu.release(lease)
                self._current_id = None
                self._cancel_event = None
                poller.set_active(False)
                self._queue.task_done()
                await self._broadcast_queue()

    def _set_status(self, run_id, status, error=None):
        with Session(engine) as db:
            run = db.get(Run, run_id)
            if run:
                run.status = status.value
                run.error = error
                run.finished_at = datetime.now(UTC)
                db.add(run)
                db.commit()

    async def _broadcast_queue(self):
        await hub.publish("system", self.snapshot())

queue = JobQueue()
