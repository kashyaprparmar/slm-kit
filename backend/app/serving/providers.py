"""Serving providers expose capabilities without loading ML libraries in the API."""
from __future__ import annotations

import asyncio
import re
import shutil
import tempfile
from pathlib import Path
from typing import Protocol

import httpx

from app.config import get_settings
from app.core.observability import activity
from app.core.resources import gpu


class ModelServingProvider(Protocol):
    name: str
    async def status(self) -> dict: ...
    async def start(self, model: str) -> dict: ...
    async def stop(self) -> bool: ...

class OllamaProvider:
    name = "ollama"
    def __init__(self):
        self._lease = None
        self._model = None
        self._lock = asyncio.Lock()

    @property
    def url(self):
        return get_settings().ollama_url.rstrip("/")

    async def call(self, method, path, payload=None, timeout=5):
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(method, self.url + path, json=payload)
            response.raise_for_status()
            data = response.json()
            if data.get("error"):
                raise RuntimeError(data["error"])
            return data

    async def status(self):
        installed = shutil.which("ollama") is not None
        result = {"provider": self.name, "installed": installed, "running": False,
                  "models": [], "loaded": [], "managed_model": self._model,
                  "endpoint": self.url, "capabilities": ["serve", "test", "unload"] + (["import_gguf"] if installed else []),
                  "guidance": "Install Ollama from ollama.com and start it. In Docker, set SLMKIT_OLLAMA_URL=http://host.docker.internal:11434."}
        try:
            tags, loaded = await asyncio.gather(self.call("GET", "/api/tags"), self.call("GET", "/api/ps"))
            result.update(running=True, models=tags.get("models", []), loaded=loaded.get("models", []), guidance="")
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            result["error"] = str(exc)
        return result

    async def start(self, model):
        async with self._lock:
            if self._model == model:
                return await self.status()
            if self._model:
                raise RuntimeError("Stop the current Ollama model before loading another.")
            state = await self.status()
            if not state["running"]:
                raise RuntimeError(state["guidance"])
            names = {m.get("name") for m in state["models"]}
            if model not in names:
                raise ValueError("Choose an installed Ollama model. Pull a model in Ollama first.")
            if state["loaded"]:
                raise RuntimeError("Ollama already has a model loaded outside SLM Kit. Unload it before starting a managed model.")
            self._lease = gpu.acquire("serving", "ollama:" + model)
            try:
                await self.call("POST", "/api/generate", {"model": model, "keep_alive": -1, "stream": False}, 180)
                self._model = model
                activity.add("Ollama", f"Model ready: {model}", "SUCCESS")
            except BaseException:
                # Loading can outlive a disconnected HTTP client. Try unloading before releasing.
                try:
                    await self.call("POST", "/api/generate", {"model": model, "keep_alive": 0, "stream": False}, 30)
                except Exception:
                    self._model = model  # retain the GPU lease until Stop confirms unload
                    raise
                gpu.release(self._lease)
                self._lease = None
                raise
        return await self.status()

    async def stop(self):
        async with self._lock:
            if not self._model:
                return False
            await self.call("POST", "/api/generate", {"model": self._model, "keep_alive": 0, "stream": False}, 60)
            activity.add("Ollama", f"Model unloaded: {self._model}")
            self._model = None
            gpu.release(self._lease)
            self._lease = None
            return True

    async def generate(self, prompt, max_tokens):
        async with self._lock:
            if not self._model:
                raise RuntimeError("Start an Ollama model first.")
            result = await self.call("POST", "/api/generate", {"model": self._model, "prompt": prompt,
                                    "stream": False, "keep_alive": -1, "options": {"num_predict": max_tokens}}, 180)
            activity.add("Ollama", "Inference completed", "SUCCESS")
            return result

    async def import_gguf(self, model: str, gguf_path: str) -> dict:
        """Import a local GGUF with Ollama's supported Modelfile workflow.

        This is intentionally CLI-backed: a remote Ollama daemon cannot see a
        backend/container path, while the native CLI is explicit about that
        filesystem boundary and provides actionable output.
        """
        executable = shutil.which("ollama")
        path = Path(gguf_path).resolve()
        if not executable:
            raise RuntimeError("Ollama CLI is not available to the backend. Run this action from a native install, or import the shown GGUF path with 'ollama create'.")
        if not path.is_file() or path.suffix.lower() != ".gguf":
            raise ValueError("The selected artifact is not a readable GGUF file.")
        if not re.fullmatch(r"[A-Za-z0-9._:-]+", model):
            raise ValueError("Ollama model names may contain letters, numbers, dots, underscores, colons and hyphens.")
        modelfile = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".Modelfile", delete=False, encoding="utf-8") as handle:
                handle.write(f'FROM "{path.as_posix()}"\n')
                modelfile = Path(handle.name)
            proc = await asyncio.create_subprocess_exec(
                executable,
                "create",
                model,
                "-f",
                str(modelfile),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                output, _ = await asyncio.wait_for(proc.communicate(), timeout=3600)
            except TimeoutError as exc:
                proc.kill()
                await proc.wait()
                raise RuntimeError("Ollama import exceeded the one-hour timeout and was stopped.") from exc
            text = output.decode(errors="replace").strip()
            if proc.returncode != 0:
                raise RuntimeError(text or f"ollama create exited with code {proc.returncode}")
            activity.add("Ollama", f"Imported GGUF as {model}", "SUCCESS")
            return {"model": model, "status": "ready", "output": text}
        finally:
            if modelfile:
                modelfile.unlink(missing_ok=True)

ollama = OllamaProvider()
