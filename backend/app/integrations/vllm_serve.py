"""Optional vLLM serving backend for the Playground.

vLLM gives warm, persistent model serving via an OpenAI-compatible HTTP API —
the model loads once and stays resident, so the second and third prompt are
fast instead of re-loading the model from a cold subprocess every time (the
transformers path in ``train_entry/generate.py``).

vLLM is Linux-first; on native Windows it's normally not installable, so this
whole module degrades gracefully: if ``vllm`` isn't importable,
``vllm_available()`` returns False and callers fall back to the transformers
subprocess path with no behavior change. See docs/installation.md for the
WSL2/Linux install step.

Scope: only the Playground (single-turn streamed chat) is served by vLLM. The
eval harness needs raw-logit access for perplexity and its own per-sample
control flow, so it always uses the transformers path.

Single-GPU safety: a warm vLLM server holds significant VRAM even between
requests, so ``manager.model is not None`` counts as "GPU busy" (see
``eval_manager.busy()``), and the job queue stops any warm server before a
training run starts (see ``core/queue.py``). An idle timeout also auto-stops
the server so it doesn't permanently block training if left running.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import time
from typing import AsyncIterator, Callable, Optional

import httpx

from app.config import get_settings
from app.core.logging_config import get_logger

_settings = get_settings()
log = get_logger(__name__)

LogCallback = Optional[Callable[[str], None]]


def vllm_available() -> bool:
    """Cheap check — does not import vllm (which is slow and torch-heavy)."""
    return importlib.util.find_spec("vllm") is not None


class VLLMServerManager:
    """Owns at most one running vLLM server subprocess at a time."""

    def __init__(self) -> None:
        self._proc: asyncio.subprocess.Process | None = None
        self._model: Optional[str] = None
        self._lock = asyncio.Lock()
        self._last_used = 0.0
        self._idle_task: asyncio.Task | None = None

    @property
    def model(self) -> Optional[str]:
        """The currently-loaded model, or None if no server is warm."""
        if self._proc is not None and self._proc.returncode is None:
            return self._model
        return None

    def base_url(self) -> str:
        return f"http://127.0.0.1:{_settings.vllm_port}"

    async def ensure(self, model_ref: str, log_cb: LogCallback = None) -> None:
        """Start (or reuse) a vLLM server for ``model_ref``. Single-flight via lock."""
        async with self._lock:
            if self.model == model_ref:
                self._last_used = time.time()
                return  # already warm for this exact model
            await self._stop_locked()
            if log_cb:
                log_cb(f"Starting vLLM server for {model_ref} (first request after a model change is slower)…")
            log.info("starting vLLM server: model=%s port=%s", model_ref, _settings.vllm_port)
            self._proc = await asyncio.create_subprocess_exec(
                sys.executable, "-u", "-m", "vllm.entrypoints.openai.api_server",
                "--model", model_ref,
                "--port", str(_settings.vllm_port),
                "--gpu-memory-utilization", str(_settings.vllm_gpu_memory_utilization),
                "--max-model-len", str(_settings.vllm_max_model_len),
                "--dtype", "auto",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            )
            self._model = model_ref
            asyncio.create_task(self._drain_stdout(log_cb))
            try:
                await self._wait_healthy()
            except Exception:
                await self._stop_locked()
                raise
            self._last_used = time.time()
            self._schedule_idle_check()

    async def _drain_stdout(self, log_cb: LogCallback) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").strip()
            if line and log_cb:
                log_cb(line)

    async def _wait_healthy(self, timeout: float = 180.0) -> None:
        deadline = time.time() + timeout
        async with httpx.AsyncClient() as client:
            while time.time() < deadline:
                if self._proc is None or self._proc.returncode is not None:
                    raise RuntimeError(
                        f"vLLM server exited before becoming healthy (exit={self._proc.returncode if self._proc else '?'})."
                    )
                try:
                    r = await client.get(f"{self.base_url()}/health", timeout=2)
                    if r.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(1)
        raise TimeoutError("vLLM server did not become healthy in time — check the log output.")

    def _schedule_idle_check(self) -> None:
        if self._idle_task:
            self._idle_task.cancel()
        self._idle_task = asyncio.create_task(self._idle_watch())

    async def _idle_watch(self) -> None:
        while True:
            await asyncio.sleep(30)
            if self._proc is None or self._proc.returncode is not None:
                return
            if time.time() - self._last_used > _settings.vllm_idle_timeout_seconds:
                log.info("vLLM server idle for %ss — stopping to free VRAM", _settings.vllm_idle_timeout_seconds)
                await self.stop()
                return

    async def stream_chat(self, prompt: str, params: dict) -> AsyncIterator[str]:
        """Stream response text deltas from the warm server's chat endpoint."""
        self._last_used = time.time()
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": params.get("max_new_tokens", 256),
            "temperature": params.get("temperature", 0.7),
            "top_p": params.get("top_p", 0.95),
            # vLLM's OpenAI API has no repetition_penalty; approximate with
            # frequency_penalty (roughly comparable direction, not identical math).
            "frequency_penalty": max(0.0, (params.get("repetition_penalty", 1.1) - 1.0) * 2),
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", f"{self.base_url()}/v1/chat/completions", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[len("data: "):]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = (obj.get("choices") or [{}])[0].get("delta", {})
                    text = delta.get("content")
                    if text:
                        yield text
        self._last_used = time.time()

    async def stop(self) -> None:
        async with self._lock:
            await self._stop_locked()

    async def _stop_locked(self) -> None:
        if self._idle_task:
            self._idle_task.cancel()
            self._idle_task = None
        proc, self._proc = self._proc, None
        self._model = None
        if proc is not None and proc.returncode is None:
            log.info("stopping vLLM server")
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=10)
            except (asyncio.TimeoutError, ProcessLookupError):
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass


manager = VLLMServerManager()
