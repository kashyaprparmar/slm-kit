"""Lifecycle manager for the one managed local model deployment.

Serving is a GPU workload too.  It runs in a child process, so Stop always
releases VRAM just like a cancelled training run.  Other GPU owners consult
``active`` before they start; the API keeps the single-GPU promise intact.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path

import httpx

from app.config import get_settings
from app.core.logging_config import get_logger
from app.core.resources import gpu
from app.model_refs import ModelReferenceError, resolve_model_ref

_settings = get_settings()
log = get_logger(__name__)


class DeploymentManager:
    def __init__(self) -> None:
        self._proc: asyncio.subprocess.Process | None = None
        self._model_ref: str | None = None
        self._kind: str | None = None
        self._started_at: float | None = None
        self._logs: list[str] = []
        self._drain_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._lease: str | None = None
        self._ready = False

    @property
    def active(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    def status(self) -> dict:
        return {
            "available": self.available(),
            "active": self.active,
            "state": "ready" if self.active and self._ready else "loading" if self.active else "stopped",
            "model_ref": self._model_ref if self.active else None,
            "kind": self._kind if self.active else None,
            "endpoint": f"http://localhost:{_settings.deploy_port}/v1" if self.active else None,
            "health_url": f"http://localhost:{_settings.deploy_port}/health" if self.active else None,
            "started_at": self._started_at if self.active else None,
            "pid": self._proc.pid if self.active and self._proc else None,
            "logs": self._logs[-50:],
        }

    @staticmethod
    def available() -> bool:
        return (
            importlib.util.find_spec("torch") is not None
            and importlib.util.find_spec("transformers") is not None
            and importlib.util.find_spec("tokenizers") is not None
        )

    async def start(self, model_ref: str) -> dict:
        """Start/reuse the deployment and wait until its health endpoint is ready."""
        if not self.available():
            raise RuntimeError("The model runtime is not installed. Use the GPU image or install the [gpu] extra.")
        try:
            spec = resolve_model_ref(model_ref)
        except ModelReferenceError:
            raise
        async with self._lock:
            if self.active and self._model_ref == spec.requested_ref:
                return self.status()
            await self._stop_locked()
            self._lease = gpu.acquire("serving", spec.requested_ref)
            workdir = _settings.deployments_dir / "active"
            workdir.mkdir(parents=True, exist_ok=True)
            config_path = workdir / "config.json"
            config_path.write_text(
                json.dumps({"model_ref": spec.requested_ref, "model_id": spec.label}, indent=2),
                encoding="utf-8",
            )
            self._logs = [f"Starting {spec.kind} deployment for {spec.requested_ref}"]
            try:
                self._proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-u",
                "-m",
                "app.train_entry.serve",
                str(config_path),
                cwd=str(Path(__file__).resolve().parents[2]),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            except BaseException:
                gpu.release(self._lease)
                self._lease = None
                raise
            self._model_ref = spec.requested_ref
            self._kind = spec.kind
            self._started_at = time.time()
            self._drain_task = asyncio.create_task(self._drain_output(self._proc))

        try:
            await self._wait_healthy()
        except Exception:
            async with self._lock:
                await self._stop_locked()
            raise
        log.info("deployment ready: model=%s", spec.requested_ref)
        self._ready = True
        return self.status()

    async def stop(self) -> bool:
        async with self._lock:
            existed = self.active or self._proc is not None
            await self._stop_locked()
            return existed

    async def _wait_healthy(self) -> None:
        deadline = time.time() + _settings.deploy_startup_timeout_seconds
        health_url = f"http://127.0.0.1:{_settings.deploy_port}/health"
        async with httpx.AsyncClient() as client:
            while time.time() < deadline:
                if self._proc is None or self._proc.returncode is not None:
                    tail = "\n".join(self._logs[-15:])
                    raise RuntimeError(f"Deployment exited before becoming healthy. Recent logs:\n{tail}")
                try:
                    response = await client.get(health_url, timeout=2)
                    if response.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(1)
        raise TimeoutError(
            f"Deployment did not become healthy within {_settings.deploy_startup_timeout_seconds:.0f}s."
        )

    async def _drain_output(self, proc: asyncio.subprocess.Process) -> None:
        if proc.stdout is None:
            return
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").strip()
            if line:
                self._logs.append(line)
                if len(self._logs) > 500:
                    del self._logs[:-500]
                log.info("deployment: %s", line)
        await proc.wait()
        if proc is self._proc:
            gpu.release(self._lease)
            self._lease = None

    async def _stop_locked(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.returncode is None:
            log.info("stopping deployment pid=%s", proc.pid)
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=10)
            except (TimeoutError, ProcessLookupError):
                try:
                    proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass
        if self._drain_task is not None:
            if not self._drain_task.done():
                self._drain_task.cancel()
                await asyncio.gather(self._drain_task, return_exceptions=True)
            self._drain_task = None
        self._model_ref = None
        self._kind = None
        self._started_at = None
        self._ready = False
        gpu.release(self._lease)
        self._lease = None


manager = DeploymentManager()
