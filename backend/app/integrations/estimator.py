"""Fallback VRAM estimator — the local math used when llmfit isn't available.

This is deliberately conservative and explainable: it returns a per-component
breakdown so the UI can show *why* a run fits or won't. All figures are MB and
peak (training) unless noted. Numbers are approximations tuned to be slightly
pessimistic — better to warn about a run that would have just fit than to let one
OOM on first try.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.domain import FitLevel, HardwareProfile, MemoryEstimate, Method, RunConfig

_MB = 1024 * 1024
# CUDA context + driver + framework baseline that exists before any weights load.
_CUDA_CONTEXT_MB = 600.0


@dataclass
class ModelSpec:
    num_params: int          # total parameters
    hidden_size: int = 2048
    num_layers: int = 24
    num_kv_heads: int = 8    # for GQA-aware KV-cache sizing
    head_dim: int = 128


# Extracts a parameter count straight from the repo name (e.g. "7b", "0.6b",
# "350m") when we can't fetch a real config.json — used offline or for a typo'd
# repo id. Quantization suffixes are stripped first: without that, a bare "4b"
# pattern would false-match inside "...-bnb-4bit", which is now most curated
# repo ids. Requires a non-alphanumeric (or end-of-string) boundary after the
# unit so "4bit"/"8bit" can never be mistaken for a size.
_QUANT_SUFFIX = re.compile(r"[-_](?:unsloth-)?bnb-4bit|[-_]gguf|[-_][48]bit$", re.IGNORECASE)
_SIZE_B = re.compile(r"(\d+(?:\.\d+)?)\s*b(?:$|[^a-z0-9])", re.IGNORECASE)
_SIZE_M = re.compile(r"(\d+(?:\.\d+)?)\s*m(?:$|[^a-z0-9])", re.IGNORECASE)


def guess_params_from_name(name: str) -> int:
    cleaned = _QUANT_SUFFIX.sub("", name.lower())
    if m := _SIZE_B.search(cleaned):
        return int(float(m.group(1)) * 1_000_000_000)
    if m := _SIZE_M.search(cleaned):
        return int(float(m.group(1)) * 1_000_000)
    return 1_500_000_000  # unknown → assume a ~1.5B model


def spec_from_name(name: str) -> ModelSpec:
    """Very rough spec when we only have a repo id (no config.json fetched)."""
    n = guess_params_from_name(name)
    # Scale architecture guesses with param count.
    if n <= 200_000_000:
        return ModelSpec(n, hidden_size=576, num_layers=30, num_kv_heads=3, head_dim=64)
    if n <= 800_000_000:
        return ModelSpec(n, hidden_size=896, num_layers=24, num_kv_heads=2, head_dim=64)
    if n <= 2_000_000_000:
        return ModelSpec(n, hidden_size=2048, num_layers=28, num_kv_heads=4, head_dim=64)
    if n <= 4_000_000_000:
        return ModelSpec(n, hidden_size=3072, num_layers=28, num_kv_heads=8, head_dim=128)
    return ModelSpec(n, hidden_size=4096, num_layers=32, num_kv_heads=8, head_dim=128)


def _bytes_per_weight(cfg: RunConfig) -> float:
    if cfg.method == Method.QLORA or cfg.load_in_4bit:
        return 0.55       # 4-bit NF4 packed + per-block scales/zeros
    return 2.0            # fp16 / bf16


def estimate(cfg: RunConfig, hw: HardwareProfile, spec: ModelSpec) -> MemoryEstimate:
    notes: list[str] = []
    n = spec.num_params

    # --- Base weights ---------------------------------------------------
    weights_mb = n * _bytes_per_weight(cfg) / _MB

    # --- Trainable params + optimizer/grad ------------------------------
    if cfg.method in (Method.LORA, Method.QLORA, Method.DORA):
        # Adapter params ≈ 2 * r * hidden per target module per layer.
        n_targets = max(1, len(cfg.lora.target_modules))
        trainable = 2 * cfg.lora.r * spec.hidden_size * spec.num_layers * n_targets
        if cfg.method == Method.DORA:
            trainable = int(trainable * 1.5)  # DoRA adds a magnitude vector
        # LoRA optimizer/grad are fp32-ish but tiny relative to base.
        optimizer_mb = trainable * (4 + 4 + 4) / _MB   # grad + m + v (fp32)
        notes.append(f"~{trainable/1e6:.1f}M trainable adapter params")
    elif cfg.method == Method.FULL:
        trainable = n
        # adamw_8bit: 8-bit m+v (2B) + fp16 grad (2B) + fp32 master (4B).
        optimizer_mb = trainable * (2 + 2 + 4) / _MB
        notes.append("Full fine-tune: optimizer state dominates VRAM")
    else:  # prompt/prefix tuning — trainable is negligible
        optimizer_mb = 8.0

    # --- Activations (with gradient checkpointing assumed, as Unsloth does) ---
    b = cfg.train.per_device_batch_size
    s = cfg.train.max_seq_length
    # Checkpointed activations ~ store per-layer inputs: b * s * hidden * 2 bytes,
    # times a modest constant for attention/MLP temporaries.
    activations_mb = b * s * spec.hidden_size * 2 * 3.0 / _MB

    # --- KV cache (small during checkpointed training; kept for realism) ---
    kv_cache_mb = 2 * b * s * spec.num_kv_heads * spec.head_dim * spec.num_layers * 2 / _MB

    subtotal = weights_mb + optimizer_mb + activations_mb + kv_cache_mb
    overhead_mb = _CUDA_CONTEXT_MB + 0.15 * subtotal
    total_mb = subtotal + overhead_mb

    budget_mb = float(hw.vram_total_mb or 8192)
    safe = budget_mb * 0.90
    if total_mb <= safe * 0.80:
        fit = FitLevel.FITS
    elif total_mb <= safe:
        fit = FitLevel.TIGHT
        notes.append("Tight — close the browser's GPU-heavy tabs before launching")
    else:
        fit = FitLevel.WONT_FIT
        notes.append("Predicted to exceed usable VRAM; expect OOM")

    return MemoryEstimate(
        weights_mb=round(weights_mb, 1),
        optimizer_mb=round(optimizer_mb, 1),
        activations_mb=round(activations_mb, 1),
        kv_cache_mb=round(kv_cache_mb, 1),
        overhead_mb=round(overhead_mb, 1),
        total_mb=round(total_mb, 1),
        budget_mb=round(budget_mb, 1),
        fit=fit,
        source="fallback",
        notes=notes,
    )


def suggest_seq_len_for_budget(cfg: RunConfig, hw: HardwareProfile, spec: ModelSpec) -> int:
    """Largest power-of-two-ish max_seq_length that still fits, for auto-tuning."""
    for s in (2048, 1536, 1024, 768, 512, 256):
        trial = cfg.model_copy(deep=True)
        trial.train.max_seq_length = s
        if estimate(trial, hw, spec).fit != FitLevel.WONT_FIT:
            return s
    return 256
