"""Serving providers expose capabilities without loading ML libraries in the API."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Protocol

import httpx

from app.capabilities import (
    Capability,
    ServingProviderCapabilities,
    SupportState,
    dependency_statuses,
)
from app.config import get_settings
from app.core.observability import activity
from app.core.resources import gpu
from app.serving.runtime_options import (
    FLAGS,
    ServingOptions,
    installation,
    option_arguments,
    probe_flags,
)


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
        supported = Capability(
            state=SupportState.SUPPORTED, reason="Implemented by the managed Transformers server."
        )
        dependencies = dependency_statuses()
        missing = [
            name
            for name in ("torch", "transformers", "tokenizers")
            if not dependencies[name].installed
        ]
        availability = Capability(
            state=SupportState.MISSING_DEPENDENCY if missing else SupportState.SUPPORTED,
            reason="Install optional runtime: " + ", ".join(missing)
            if missing
            else "Managed runtime dependencies installed.",
        )
        return ServingProviderCapabilities(
            name=self.name,
            display_name="Transformers",
            availability=availability,
            operations={
                name: supported for name in ("serve", "test", "chat", "scratch", "adapter")
            },
            model_formats=["transformers", "peft_adapter", "slmkit_scratch"],
            required_dependencies=["torch", "transformers"],
            features={
                key: supported
                if key
                in ("chat_completion", "text_completion", "streaming", "lora", "reward_scoring")
                else Capability(
                    state=SupportState.UNSUPPORTED,
                    reason="No verified implementation in the managed native server.",
                )
                for key in (
                    "chat_completion",
                    "text_completion",
                    "streaming",
                    "lora",
                    "reward_scoring",
                    "quantized_models",
                    "tool_calling",
                    "reasoning",
                    "metrics",
                    "batching",
                    "prefix_caching",
                    "speculative_decoding",
                )
            },
        )

    async def status(self) -> dict:
        from app.core.deployment import manager

        capabilities = self.capabilities()
        return {
            **manager.status(),
            "provider": self.name,
            "managed": True,
            "capabilities": capabilities.enabled_operations,
            "provider_capabilities": capabilities.model_dump(mode="json"),
        }

    async def start(self, model: str) -> dict:
        from app.core.deployment import manager

        owner = await external_gpu_owner(exclude=self.name)
        if owner:
            raise RuntimeError(
                f"GPU is already in use by external provider '{owner}'. Stop it before starting Transformers serving."
            )
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
            response = await client.post(
                url,
                json={"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens},
            )
            response.raise_for_status()
            data = response.json()
        usage = data.get("usage") or {}
        return {
            "response": data["choices"][0]["message"]["content"],
            "eval_count": usage.get("completion_tokens"),
            "seconds": time.perf_counter() - started,
        }


class OpenAICompatibleProvider:
    """Read-only lifecycle integration for the dedicated vLLM container."""

    name = "vllm"

    def __init__(self, name="vllm"):
        self.name = name
        self._lease = None
        self._monitor = None
        self._lock = asyncio.Lock()

    @property
    def manager(self):
        from app.integrations.vllm_serve import manager, sglang_manager

        return manager if self.name == "vllm" else sglang_manager

    def capabilities(self) -> ServingProviderCapabilities:
        installed, version = installation(self.name)
        supported = Capability(
            state=SupportState.SUPPORTED,
            reason="Available through a separately managed OpenAI-compatible service.",
            requirements=["vLLM service"],
        )
        return ServingProviderCapabilities(
            name=self.name,
            display_name="vLLM" if self.name == "vllm" else "SGLang",
            availability=supported,
            operations={
                **{name: supported for name in ("test", "chat", "stream")},
                "serve": Capability(
                    state=SupportState.EXPERIMENTAL if installed else SupportState.NOT_INSTALLED,
                    reason="Optional installed engine; validate model and executable flags before launch."
                    if installed
                    else f"{self.name} is not installed. Optional.",
                ),
            },
            model_formats=["transformers", "safetensors"],
            required_dependencies=["vLLM service"],
            installed_version=version,
            features={
                key: Capability(
                    state=SupportState.EXPERIMENTAL
                    if key in ("chat_completion", "text_completion", "streaming", "batching")
                    else SupportState.UNSUPPORTED,
                    reason="Requires service/model verification."
                    if key in ("chat_completion", "text_completion", "streaming", "batching")
                    else "Not configured or verified for this deployment.",
                )
                for key in (
                    "chat_completion",
                    "text_completion",
                    "streaming",
                    "lora",
                    "quantized_models",
                    "tool_calling",
                    "reasoning",
                    "metrics",
                    "batching",
                    "prefix_caching",
                    "speculative_decoding",
                )
            },
        )

    @property
    def url(self) -> str:
        if self._lease:
            return self.manager.base_url()
        return getattr(get_settings(), self.name + "_url").rstrip("/")

    async def status(self) -> dict:
        capabilities = self.capabilities()
        result = {
            "provider": self.name,
            "available": False,
            "active": False,
            "managed": False,
            "endpoint": self.url,
            "models": [],
            "capabilities": capabilities.enabled_operations,
            "provider_capabilities": capabilities.model_dump(mode="json"),
            "guidance": "Start the dedicated service with: docker compose --profile vllm up -d vllm",
        }
        if self.name == "sglang":
            installed, version = installation("sglang")
            result.update(
                installed=installed,
                version=version,
                optional=True,
                guidance="Install optional SGLang in a compatible Linux GPU environment, launch its OpenAI server, and set SLMKIT_SGLANG_URL.",
            )
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                response = await client.get(self.url + "/v1/models")
                response.raise_for_status()
                models = response.json().get("data", [])
            result.update(available=True, active=True, models=models, guidance="")
        except (httpx.HTTPError, ValueError) as exc:
            result["error"] = str(exc)
        result["managed"] = bool(self._lease and self.manager.model)
        result["installed"], result["version"] = installation(self.name)
        result["logs"] = self.manager.logs if result["managed"] else []
        return result

    async def start(self, model: str, options: ServingOptions | None = None) -> dict:
        async with self._lock:
            if not installation(self.name)[0]:
                raise RuntimeError(
                    f"{self.name} backend: Not installed. Optional. Install it in a compatible Linux GPU environment."
                )
            options = options or ServingOptions()
            flags = await probe_flags(self.name)
            option_arguments(self.name, options, flags)
            from app.model_refs import resolve_model_ref

            spec = resolve_model_ref(model)
            if not spec.deployable or spec.kind == "scratch":
                raise ValueError("This engine requires a compatible causal Hugging Face model.")
            if spec.kind == "adapter":
                raise ValueError(
                    "Merge this adapter first; adapter loading must be explicitly configured for the engine."
                )
            from app.core.hardware import poller

            hardware = poller.latest
            if not hardware.cuda_available:
                raise ValueError(
                    "Managed optional engines require a detected CUDA GPU; external services remain usable."
                )
            if options.tensor_parallel_size and options.tensor_parallel_size > hardware.gpu_count:
                raise ValueError("Tensor parallel size exceeds the detected GPU count.")
            if spec.local_path:
                import json

                config_file = Path(spec.local_path) / "config.json"
                config = json.loads(config_file.read_text(encoding="utf-8"))
                context = config.get("max_position_embeddings") or config.get("n_positions")
                if options.max_model_len and context and options.max_model_len > context:
                    raise ValueError(
                        "Requested model length exceeds the model configuration; configure/validate context extension explicitly first."
                    )
                configured_quantization = (config.get("quantization_config") or {}).get(
                    "quant_method"
                )
                if options.quantization and options.quantization != configured_quantization:
                    raise ValueError(
                        "Requested quantization differs from the artifact's recorded quantization."
                    )
            elif options.quantization or options.speculative_config:
                raise ValueError(
                    "Quantization/speculative controls require a local inspected artifact."
                )
            if options.speculative_config:
                raise ValueError(
                    "Speculative decoding requires a verified draft-model compatibility profile; this integration is capability-gated."
                )
            state = await self.status()
            if state["active"] and not self._lease:
                raise RuntimeError(
                    "An external server already owns this endpoint; stop it explicitly before managed launch."
                )
            owner = await external_gpu_owner(exclude=self.name)
            if owner:
                raise RuntimeError(f"Stop external {owner} before starting this server.")
            if not self._lease:
                self._lease = gpu.acquire("serving", self.name + ":" + model)
            try:
                await self.manager.ensure(spec.load_ref, options=options, managed=True)
            except BaseException:
                gpu.release(self._lease)
                self._lease = None
                raise
            if not self._monitor or self._monitor.done():
                self._monitor = asyncio.create_task(self._release_on_exit())
        return await self.status()

    async def _release_on_exit(self):
        while True:
            await asyncio.sleep(1)
            async with self._lock:
                if not self.manager.model:
                    gpu.release(self._lease)
                    self._lease = None
                    return

    async def stop(self) -> bool:
        async with self._lock:
            if not self._lease:
                raise RuntimeError(
                    "This is an external service. Stop it explicitly; SLM Kit only stops its own workers."
                )
            await self.manager.stop()
            gpu.release(self._lease)
            self._lease = None
            if self._monitor:
                self._monitor.cancel()
                await asyncio.gather(self._monitor, return_exceptions=True)
                self._monitor = None
            return True

    async def options(self) -> dict:
        flags = await probe_flags(self.name)
        schema = ServingOptions.model_json_schema()
        supported = {
            key: value
            for key, value in schema["properties"].items()
            if FLAGS[self.name].get(key) in flags
            and key not in ("speculative_config", "quantization", "max_lora_rank")
        }
        # These options need an artifact-specific profile; advertise their capability
        # but do not expose a generic control that could misrepresent model support.
        return {
            "properties": supported,
            "installed_version": installation(self.name)[1],
            "guidance": "Flags verified against the installed CLI. Model/hardware compatibility is checked by the engine during supervised startup.",
        }

    async def generate(self, prompt: str, max_tokens: int) -> dict:
        state = await self.status()
        if not state["active"]:
            raise RuntimeError(state["guidance"])
        model = state["models"][0]["id"] if state["models"] else None
        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(
                self.url + "/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                },
            )
            response.raise_for_status()
            data = response.json()
        usage = data.get("usage") or {}
        return {
            "response": data["choices"][0]["message"]["content"],
            "eval_count": usage.get("completion_tokens"),
            "seconds": time.perf_counter() - started,
        }


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
            reason="Ollama CLI is installed."
            if installed
            else "Install the Ollama CLI to manage local models.",
            requirements=["ollama"],
        )
        return ServingProviderCapabilities(
            name=self.name,
            display_name="Ollama",
            availability=support,
            operations={name: support for name in ("serve", "test", "unload", "import_gguf")},
            model_formats=["ollama", "gguf"],
            required_dependencies=["ollama"],
            features={
                key: Capability(
                    state=SupportState.EXPERIMENTAL
                    if key
                    in ("chat_completion", "text_completion", "streaming", "quantized_models")
                    else SupportState.UNSUPPORTED,
                    reason="Requires a running compatible Ollama service/model."
                    if key
                    in ("chat_completion", "text_completion", "streaming", "quantized_models")
                    else "Not configured or verified.",
                )
                for key in (
                    "chat_completion",
                    "text_completion",
                    "streaming",
                    "lora",
                    "quantized_models",
                    "tool_calling",
                    "reasoning",
                    "metrics",
                    "batching",
                    "prefix_caching",
                    "speculative_decoding",
                )
            },
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
        result = {
            "provider": self.name,
            "installed": installed,
            "running": False,
            "models": [],
            "loaded": [],
            "managed_model": self._model,
            "endpoint": self.url,
            "capabilities": legacy_operations,
            "provider_capabilities": capabilities.model_dump(mode="json"),
            "guidance": "Install Ollama from ollama.com and start it. In Docker, set SLMKIT_OLLAMA_URL=http://host.docker.internal:11434.",
        }
        try:
            tags, loaded = await asyncio.gather(
                self.call("GET", "/api/tags"), self.call("GET", "/api/ps")
            )
            result.update(
                running=True,
                models=tags.get("models", []),
                loaded=loaded.get("models", []),
                guidance="",
            )
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
                raise RuntimeError(
                    f"GPU is already in use by external provider '{owner}'. Stop it before loading Ollama."
                )
            state = await self.status()
            if not state["running"]:
                raise RuntimeError(state["guidance"])
            names = {m.get("name") for m in state["models"]}
            if model not in names:
                raise ValueError("Choose an installed Ollama model. Pull a model in Ollama first.")
            if state["loaded"]:
                raise RuntimeError(
                    "Ollama already has a model loaded outside SLM Kit. Unload it before starting a managed model."
                )
            self._lease = gpu.acquire("serving", "ollama:" + model)
            try:
                await self.call(
                    "POST",
                    "/api/generate",
                    {"model": model, "keep_alive": -1, "stream": False},
                    180,
                )
                self._model = model
                activity.add("Ollama", f"Model ready: {model}", "SUCCESS")
            except BaseException:
                # Loading can outlive a disconnected HTTP client. Try unloading before releasing.
                try:
                    await self.call(
                        "POST",
                        "/api/generate",
                        {"model": model, "keep_alive": 0, "stream": False},
                        30,
                    )
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
            await self.call(
                "POST",
                "/api/generate",
                {"model": self._model, "keep_alive": 0, "stream": False},
                60,
            )
            activity.add("Ollama", f"Model unloaded: {self._model}")
            self._model = None
            gpu.release(self._lease)
            self._lease = None
            return True

    async def generate(self, prompt, max_tokens):
        async with self._lock:
            if not self._model:
                raise RuntimeError("Start an Ollama model first.")
            result = await self.call(
                "POST",
                "/api/generate",
                {
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": -1,
                    "options": {"num_predict": max_tokens},
                },
                180,
            )
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
            raise RuntimeError(
                "Ollama CLI is not available to the backend. Run this action from a native install, or import the shown GGUF path with 'ollama create'."
            )
        if not path.is_file() or path.suffix.lower() != ".gguf":
            raise ValueError("The selected artifact is not a readable GGUF file.")
        if not re.fullmatch(r"[A-Za-z0-9._:-]+", model):
            raise ValueError(
                "Ollama model names may contain letters, numbers, dots, underscores, colons and hyphens."
            )
        modelfile = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", suffix=".Modelfile", delete=False, encoding="utf-8"
            ) as handle:
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
            except (TimeoutError, asyncio.CancelledError) as exc:
                from app.core.runner import _terminate_process_tree

                await asyncio.to_thread(_terminate_process_tree, proc.pid)
                await proc.wait()
                if isinstance(exc, asyncio.CancelledError):
                    raise
                raise RuntimeError(
                    "Ollama import exceeded the one-hour timeout and was stopped."
                ) from exc
            text = output.decode(errors="replace").strip()
            if proc.returncode != 0:
                raise RuntimeError(text or f"ollama create exited with code {proc.returncode}")
            activity.add("Ollama", f"Imported GGUF as {model}", "SUCCESS")
            return {"model": model, "status": "ready", "output": text}
        finally:
            if modelfile:
                modelfile.unlink(missing_ok=True)

    async def import_package(self, model: str, package_path: str) -> dict:
        """Import an SLM Kit generated Ollama package without discarding policy."""
        package = Path(package_path).resolve()
        modelfile = package / "Modelfile"
        if not package.is_dir() or not modelfile.is_file():
            raise ValueError("The selected Ollama artifact does not contain a Modelfile package.")
        executable = shutil.which("ollama")
        if not executable:
            raise RuntimeError(
                "Ollama CLI is not available to the backend. Export completed; run 'ollama create <name> -f Modelfile' from the package directory."
            )
        if not re.fullmatch(r"[A-Za-z0-9._:-]+", model):
            raise ValueError(
                "Ollama model names may contain letters, numbers, dots, underscores, colons and hyphens."
            )
        proc = await asyncio.create_subprocess_exec(
            executable,
            "create",
            model,
            "-f",
            str(modelfile),
            cwd=str(package),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            output, _ = await asyncio.wait_for(proc.communicate(), timeout=3600)
        except (TimeoutError, asyncio.CancelledError) as exc:
            from app.core.runner import _terminate_process_tree

            await asyncio.to_thread(_terminate_process_tree, proc.pid)
            await proc.wait()
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise RuntimeError(
                "Ollama import exceeded the one-hour timeout and was stopped."
            ) from exc
        text = output.decode(errors="replace").strip()
        if proc.returncode != 0:
            raise RuntimeError(text or f"ollama create exited with code {proc.returncode}")
        activity.add("Ollama", f"Imported package as {model}", "SUCCESS")
        return {"model": model, "status": "ready", "output": text}


ollama = OllamaProvider()
transformers = TransformersProvider()
vllm = OpenAICompatibleProvider()
sglang = OpenAICompatibleProvider("sglang")
PROVIDERS: dict[str, ModelServingProvider] = {
    transformers.name: transformers,
    ollama.name: ollama,
    vllm.name: vllm,
    sglang.name: sglang,
}


def get_provider(name: str) -> ModelServingProvider:
    try:
        return PROVIDERS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown inference provider '{name}'. Available: {', '.join(PROVIDERS)}"
        ) from exc


async def benchmark_stream(provider_name: str, prompt: str, max_tokens: int):
    """Yield normalized streamed output events for the benchmark owner.

    Providers retain their own lifecycle/configuration; this only normalizes the
    already-public streaming protocol so timing is measured at receipt time.
    """
    provider = get_provider(provider_name)
    state = await provider.status()
    if provider_name == "ollama":
        model = getattr(ollama, "_model", None)
        if not state.get("running") or not model:
            raise RuntimeError("Start an Ollama model first.")
        async with httpx.AsyncClient(timeout=180) as client:
            async with client.stream(
                "POST",
                ollama.url + "/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": True,
                    "keep_alive": -1,
                    "options": {"num_predict": max_tokens},
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    item = json.loads(line)
                    yield {
                        "text": item.get("response"),
                        "done": bool(item.get("done")),
                        "completion_tokens": item.get("eval_count"),
                        "generation_seconds": (item.get("eval_duration") or 0) / 1e9,
                    }
        return
    if not state.get("active"):
        raise RuntimeError(state.get("guidance") or "Start this serving provider first.")
    if provider_name == "transformers":
        endpoint = f"http://127.0.0.1:{get_settings().deploy_port}"
        model = None
    else:
        endpoint = provider.url
        model = state.get("models", [{}])[0].get("id") if state.get("models") else None
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    if model:
        payload["model"] = model
    async with httpx.AsyncClient(timeout=180) as client:
        async with client.stream(
            "POST", endpoint + "/v1/chat/completions", json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    yield {"done": True}
                    break
                item = json.loads(data)
                usage = item.get("usage") or {}
                choices = item.get("choices") or []
                delta = choices[0].get("delta", {}) if choices else {}
                yield {
                    "text": delta.get("content"),
                    "done": bool(choices and choices[0].get("finish_reason")),
                    "completion_tokens": usage.get("completion_tokens"),
                }


async def provider_statuses() -> dict[str, dict]:
    """Probe providers concurrently and isolate a broken optional provider."""
    names = list(PROVIDERS)
    results = await asyncio.gather(
        *(PROVIDERS[name].status() for name in names), return_exceptions=True
    )
    states: dict[str, dict] = {}
    for name, result in zip(names, results, strict=True):
        if isinstance(result, BaseException):
            states[name] = {
                "provider": name,
                "available": False,
                "active": False,
                "error": str(result),
                "guidance": "Provider health probe failed; inspect backend logs.",
            }
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
    if exclude != "sglang":
        names.append("sglang")
        probes.append(sglang.status())
    if exclude != "ollama":
        names.append("ollama")
        probes.append(ollama.status())
    if not probes:
        return None
    results = await asyncio.gather(*probes, return_exceptions=True)
    for name, state in zip(names, results, strict=True):
        if isinstance(state, BaseException):
            continue
        if name in ("vllm", "sglang") and state.get("active"):
            return name
        if name == "ollama" and state.get("loaded") and not ollama._model:
            return name
    return None
