"""Runtime model preflight coordinator.

Executes preflight probes in an isolated subprocess while holding an atomic GPU lease.
Guarantees complete VRAM release upon process termination or cancellation.
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from typing import Any

from app.config import get_settings
from app.core.logging_config import get_logger
from app.core.resources import ResourceBusy, gpu

_settings = get_settings()
log = get_logger(__name__)


class PreflightCancelledError(Exception):
    pass


class PreflightRunner:
    def __init__(self):
        self._active_procs: dict[str, asyncio.subprocess.Process] = {}

    async def execute(
        self,
        model_ref: str,
        revision: str | None = None,
        backend: str = "unsloth",
        load_in_4bit: bool = False,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """Run preflight in an isolated subprocess holding the GPU lease."""
        lease = None
        task_id = uuid.uuid4().hex
        try:
            lease = gpu.acquire("preflight", f"{model_ref}@{revision or 'default'}")
        except ResourceBusy as exc:
            raise ResourceBusy(f"Cannot run model preflight: {exc}")

        tmp_dir = _settings.home / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        config_path = tmp_dir / f"preflight-{task_id}.json"

        worker_cfg = {
            "model_ref": model_ref,
            "revision": revision,
            "backend": backend,
            "load_in_4bit": load_in_4bit,
            "cache_dir": str(_settings.hf_cache_dir),
            "trust_remote_code": _settings.trust_remote_code,
        }
        config_path.write_text(json.dumps(worker_cfg, ensure_ascii=False), encoding="utf-8")

        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "app.train_entry.preflight", str(config_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self._active_procs[task_id] = proc

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except TimeoutError:
                log.warning("Preflight for %s timed out after %ss — terminating", model_ref, timeout)
                proc.kill()
                await proc.wait()
                return {
                    "ok": False,
                    "model_ref": model_ref,
                    "error": f"Model preflight timed out after {timeout:.0f} seconds.",
                    "warnings": ["Preflight process was killed to release system resources."],
                }

            lines = stdout.decode("utf-8", errors="replace").strip().splitlines()
            payload: dict[str, Any] = {}
            if lines:
                try:
                    payload = json.loads(lines[-1])
                except json.JSONDecodeError:
                    pass

            if proc.returncode != 0 or not payload.get("ok"):
                err_msg = (
                    payload.get("error")
                    or stderr.decode("utf-8", errors="replace").strip()[-1000:]
                    or f"Preflight process exited with code {proc.returncode}"
                )
                return {
                    "ok": False,
                    "model_ref": model_ref,
                    "error": err_msg,
                    "warnings": payload.get("warnings", []),
                }

            return payload
        finally:
            self._active_procs.pop(task_id, None)
            config_path.unlink(missing_ok=True)
            if lease is not None:
                gpu.release(lease)

    def cancel(self, task_id: str) -> bool:
        proc = self._active_procs.get(task_id)
        if proc and proc.returncode is None:
            proc.kill()
            return True
        return False


preflight_runner = PreflightRunner()
