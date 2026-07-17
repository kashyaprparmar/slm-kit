"""GGUF conversion via llama.cpp, with graceful detection.

Like the llmfit integration, this degrades cleanly: if llama.cpp isn't present we
raise a clear, actionable error instead of failing obscurely. Point
``SLMKIT_LLAMACPP_DIR`` at a llama.cpp checkout that contains
``convert_hf_to_gguf.py`` and a built ``llama-quantize`` binary.

Note: GGUF conversion expects a *full/merged* model directory. LoRA/QLoRA runs
save adapters; those must be merged first (a documented follow-up). Full
fine-tunes and from-scratch/domain-adapted bases convert directly.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

# Common GGUF quantization types (q4_k_m is the 8GB-friendly default).
QUANT_TYPES = ["q4_k_m", "q5_k_m", "q8_0", "f16"]

LogCallback = Optional[Callable[[str], None]]


def llamacpp_dir() -> Path | None:
    d = os.environ.get("SLMKIT_LLAMACPP_DIR")
    return Path(d) if d else None


def _convert_script() -> Path | None:
    d = llamacpp_dir()
    if d and (d / "convert_hf_to_gguf.py").exists():
        return d / "convert_hf_to_gguf.py"
    return None


def _quantize_bin() -> str | None:
    found = shutil.which("llama-quantize")
    if found:
        return found
    d = llamacpp_dir()
    if d:
        for candidate in (d / "build" / "bin" / "llama-quantize", d / "llama-quantize"):
            if candidate.exists():
                return str(candidate)
    return None


def llamacpp_available() -> bool:
    return _convert_script() is not None


def _run_streamed(cmd: list[str], on_line: LogCallback, label: str) -> None:
    """Run a subprocess, calling ``on_line`` for every line of output as it
    happens (not just at the end) — this is what makes GGUF export show live,
    comprehensive progress in the Model Registry instead of going silent for
    the whole conversion."""
    if on_line:
        on_line(f"$ {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        if not line:
            continue
        tail.append(line)
        if len(tail) > 40:
            tail.pop(0)
        if on_line:
            on_line(line)
    returncode = proc.wait()
    if returncode != 0:
        detail = "\n".join(tail)
        raise RuntimeError(f"{label} failed (exit {returncode}). Last output:\n{detail}")


def quantize(src_dir: str, out_dir: str, quant_type: str = "q4_k_m", on_line: LogCallback = None) -> str:
    """Convert a HF model dir to GGUF and quantize. Returns the .gguf path.

    ``on_line`` (optional) receives every line of llama.cpp's output as it's
    produced, so a caller can stream live progress instead of waiting silently
    for the whole (often multi-minute) conversion to finish.
    """
    script = _convert_script()
    if script is None:
        raise RuntimeError(
            "llama.cpp not found. Clone https://github.com/ggerganov/llama.cpp, build it, "
            "and set SLMKIT_LLAMACPP_DIR to that folder (it must contain convert_hf_to_gguf.py "
            "and a built llama-quantize)."
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f16 = out / "model-f16.gguf"
    _run_streamed(
        [sys.executable, str(script), src_dir, "--outfile", str(f16), "--outtype", "f16"],
        on_line, "GGUF conversion",
    )
    if quant_type in ("f16", "fp16"):
        return str(f16)

    qbin = _quantize_bin()
    if qbin is None:
        raise RuntimeError("llama-quantize binary not found. Build llama.cpp so it produces llama-quantize.")
    target = out / f"model-{quant_type}.gguf"
    _run_streamed([qbin, str(f16), str(target), quant_type], on_line, "GGUF quantization")
    f16.unlink(missing_ok=True)  # keep only the quantized artifact
    return str(target)
