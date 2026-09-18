"""Validate engine controls against the installed executable's advertised flags."""
import asyncio
import json
import re
import sys
from importlib import metadata, util

from pydantic import BaseModel, ConfigDict, Field


class ServingOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gpu_memory_utilization: float | None = Field(default=None, gt=0, lt=1)
    max_model_len: int | None = Field(default=None, ge=1)
    tensor_parallel_size: int | None = Field(default=None, ge=1)
    max_lora_rank: int | None = Field(default=None, ge=1)
    enforce_eager: bool | None = None
    prefix_caching: bool | None = None
    quantization: str | None = None
    speculative_config: dict | None = None


FLAGS = {
    "vllm": {"gpu_memory_utilization": "--gpu-memory-utilization", "max_model_len": "--max-model-len",
             "tensor_parallel_size": "--tensor-parallel-size", "max_lora_rank": "--max-lora-rank",
             "enforce_eager": "--enforce-eager", "prefix_caching": "--enable-prefix-caching",
             "quantization": "--quantization", "speculative_config": "--speculative-config"},
    "sglang": {"gpu_memory_utilization": "--mem-fraction-static", "max_model_len": "--context-length",
               "tensor_parallel_size": "--tp-size"},
}
MODULES = {"vllm": "vllm.entrypoints.openai.api_server", "sglang": "sglang.launch_server"}


def installation(name: str) -> tuple[bool, str | None]:
    try:
        return util.find_spec(name) is not None, metadata.version(name)
    except (ValueError, ImportError, metadata.PackageNotFoundError):
        return False, None


async def probe_flags(name: str) -> set[str]:
    if not installation(name)[0]:
        return set()
    from app.core.runner import _terminate_process_tree
    proc = await asyncio.create_subprocess_exec(sys.executable, "-m", MODULES[name], "--help",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        output, _ = await asyncio.wait_for(proc.communicate(), 30)
        return set(re.findall(r"--[a-z][a-z0-9-]+", output.decode(errors="replace"))) if proc.returncode == 0 else set()
    finally:
        if proc.returncode is None:
            await asyncio.to_thread(_terminate_process_tree, proc.pid)
            await proc.wait()


def option_arguments(name: str, options: ServingOptions, available: set[str]) -> list[str]:
    result = []
    for key, value in options.model_dump(exclude_none=True).items():
        flag = FLAGS[name].get(key)
        if flag not in available:
            raise ValueError(f"Installed {name} does not advertise support for {key}.")
        # False would otherwise silently retain version-dependent defaults.
        if value is False:
            raise ValueError(f"Explicitly disabling {key} is not verified for this engine; leave it unspecified.")
        result.append(flag)
        if value is not True:
            result.append(json.dumps(value) if isinstance(value, dict) else str(value))
        if key == "max_lora_rank":
            if "--enable-lora" not in available:
                raise ValueError("Installed runtime does not support enabling LoRA serving.")
            result.append("--enable-lora")
    return result
