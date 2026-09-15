"""Serving providers expose capabilities without loading ML libraries in the API."""
from __future__ import annotations

import asyncio
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Protocol

import httpx

from app.capabilities import Capability, ServingProviderCapabilities, SupportState
from app.config import get_settings
from app.core.observability import activity
from app.core.resources import gpu


class ModelServingProvider(Protocol):
    name: str
    def capabilities(self) -> ServingProviderCapabilities: ...
    async def status(self) -> dict: ...
    async def start(self, model: str) -> dict: ...
    async def stop(self) -> bool: ...
    async def generate(self, prompt: str, max_tokens: int) -> dict: ...


class TransformersProvider:
    name = "transformers"

    def capabilities(self) -> ServingProviderCapabilities:
        supported = Capability(state=SupportState.SUPPORTED, reason="Implemented by the managed Transformers server.")
        return ServingProviderCapabilities(
            name=self.name,
            display_name="Transformers",
            availability=supported,
            operations={name: supported for name in ("serve", "test", "chat", "scratch", "adapter")},
            model_formats=["transformers", "peft_adapter", "slmkit_scratch"],
            required_dependencies=["torch", "transformers"],
        )

    async def status(self) -> dict:
        from app.core.deployment import manager
        capabilities = self.capabilities()
        return {**manager.status(), "provider": self.name,
                "managed": True, "capabilities": capabilities.enabled_operations,
                "provider_capabilities": capabilities.model_dump(mode="json")}

    async def start(self, model: str) -> dict:
        from app.core.deployment import manager
        owner = await external_gpu_owner(exclude=self.name)
        if owner:
            raise RuntimeError(f"GPU is already in use by external provider '{owner}'. Stop it before starting Transformers serving.")
        return await manager.start(model)

    async def stop(self) -> bool:
        from app.core.deployment import manager
        return await manager.stop()

    async def generate(self, prompt: str, max_tokens: int) -> dict:
        from app.core.deployment import manager
        if not manager.active:
            raise RuntimeError("Start a Transformers model server first.")
        started = time.perf_counter()
        url = f"http://127.0.0.1:{get_settings().deploy_port}/v1/chat/completions"
        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(url, json={"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens})
            response.raise_for_status()
            data = response.json()
        usage = data.get("usage") or {}
        return {"response": data["choices"][0]["message"]["content"], "eval_count": usage.get("completion_tokens"),
                "seconds": time.perf_counter() - started}


class OpenAICompatibleProvider:
    """Read-only lifecycle integration for the dedicated vLLM container."""

    name = "vllm"

    def capabilities(self) -> ServingProviderCapabilities:
        supported = Capability(
            state=SupportState.SUPPORTED,
            reason="Available through a separately managed OpenAI-compatible service.",
            requirements=["vLLM service"],
        )
        return ServingProviderCapabilities(
            name=self.name,
            display_name="vLLM",
            availability=supported,
            operations={name: supported for name in ("test", "chat", "stream")},
            model_formats=["transformers", "safetensors"],
            required_dependencies=["vLLM service"],
        )

    @property
    def url(self) -> str:
        return get_settings().vllm_url.rstrip("/")

    async def status(self) -> dict:
        capabilities = self.capabilities()
        result = {"provider": self.name, "available": False, "active": False, "managed": False,
                  "endpoint": self.url, "models": [], "capabilities": capabilities.enabled_operations,
                  "provider_capabilities": capabilities.model_dump(mode="json"),
                  "guidance": "Start the dedicated service with: docker compose --profile vllm up -d vllm"}
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                response = await client.get(self.url + "/v1/models")
                response.raise_for_status()
                models = response.json().get("data", [])
            result.update(available=True, active=True, models=models, guidance="")
        except (httpx.HTTPError, ValueError) as exc:
            result["error"] = str(exc)
        return result

    async def start(self, model: str) -> dict:
        raise RuntimeError("vLLM lifecycle is isolated from the API. Configure SLMKIT_VLLM_MODEL and start its Docker Compose profile.")

    async def stop(self) -> bool:
        raise RuntimeError("SLM Kit does not have unrestricted Docker control. Stop the vLLM Compose service explicitly.")

    async def generate(self, prompt: str, max_tokens: int) -> dict:
        state = await self.status()
        if not state["active"]:
            raise RuntimeError(state["guidance"])
        model = state["models"][0]["id"] if state["models"] else None
        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(self.url + "/v1/chat/completions", json={
                "model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            })
            response.raise_for_status()
            data = response.json()
        usage = data.get("usage") or {}
        return {"response": data["choices"][0]["message"]["content"], "eval_count": usage.get("completion_tokens"),
                "seconds": time.perf_counter() - started}

class OllamaProvider:
    name = "ollama"
    def __init__(self):
        self._lease = None
        self._model = None
        self._lock = asyncio.Lock()

    def capabilities(self) -> ServingProviderCapabilities:
        installed = shutil.which("ollama") is not None
        state = SupportState.SUPPORTED if installed else SupportState.NOT_INSTALLED
        support = Capability(
            state=state,
            reason="Ollama CLI is installed." if installed else "Install the Ollama CLI to manage local models.",
            requirements=["ollama"],
        )
        return ServingProviderCapabilities(
            name=self.name,
            display_name="Ollama",
            availability=support,
            operations={name: support for name in ("serve", "test", "unload", "import_gguf")},
            model_formats=["ollama", "gguf"],
            required_dependencies=["ollama"],
        )

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
        capabilities = self.capabilities()
        legacy_operations = ["serve", "test", "unload"] + (["import_gguf"] if installed else [])
        result = {"provider": self.name, "installed": installed, "running": False,
                  "models": [], "loaded": [], "managed_model": self._model,
                  "endpoint": self.url, "capabilities": legacy_operations,
                  "provider_capabilities": capabilities.model_dump(mode="json"),
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
            owner = await external_gpu_owner(exclude=self.name)
            if owner:
                raise RuntimeError(f"GPU is already in use by external provider '{owner}'. Stop it before loading Ollama.")
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
transformers = TransformersProvider()
vllm = OpenAICompatibleProvider()
PROVIDERS: dict[str, ModelServingProvider] = {
    transformers.name: transformers,
    ollama.name: ollama,
    vllm.name: vllm,
}


def get_provider(name: str) -> ModelServingProvider:
    try:
        return PROVIDERS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown inference provider '{name}'. Available: {', '.join(PROVIDERS)}") from exc


async def provider_statuses() -> dict[str, dict]:
    """Probe providers concurrently and isolate a broken optional provider."""
    names = list(PROVIDERS)
    results = await asyncio.gather(*(PROVIDERS[name].status() for name in names), return_exceptions=True)
    states: dict[str, dict] = {}
    for name, result in zip(names, results, strict=True):
        if isinstance(result, BaseException):
            states[name] = {"provider": name, "available": False, "active": False,
                            "error": str(result), "guidance": "Provider health probe failed; inspect backend logs."}
        else:
            states[name] = result
    return states


async def external_gpu_owner(*, exclude: str | None = None) -> str | None:
    """Return an unmanaged provider that may own GPU memory.

    Managed Transformers/Ollama instances hold the in-process GPU lease.  A
    Compose vLLM server and models loaded directly in Ollama do not, so they
    must be discovered before another local workload is admitted.
    """
    probes = []
    names = []
    if exclude != "vllm":
        names.append("vllm")
        probes.append(vllm.status())
    if exclude != "ollama":
        names.append("ollama")
        probes.append(ollama.status())
    if not probes:
        return None
    results = await asyncio.gather(*probes, return_exceptions=True)
    for name, state in zip(names, results, strict=True):
        if isinstance(state, BaseException):
            continue
        if name == "vllm" and state.get("active"):
            return name
        if name == "ollama" and state.get("loaded") and not ollama._model:
            return name
    return None
