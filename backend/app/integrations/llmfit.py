"""llmfit integration with graceful fallback.

Primary path: shell out to the ``llmfit`` binary (JSON mode) for hardware-aware
fit scoring. If the binary isn't installed, everything falls back to the internal
estimator (``estimator.py``) so the fit/advisor features keep working. Both paths
return the same ``MemoryEstimate`` shape.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from functools import lru_cache
from typing import Any

from app.config import get_settings
from app.domain import HardwareProfile, MemoryEstimate, RunConfig
from app.integrations import estimator, hf_hub
from app.model_refs import ModelReferenceError, resolve_model_ref

_settings = get_settings()


@lru_cache(maxsize=1)
def llmfit_available() -> bool:
    return shutil.which(_settings.llmfit_bin) is not None


def _run_json(args: list[str], timeout: float = 30.0) -> Any | None:
    if not llmfit_available():
        return None
    try:
        proc = subprocess.run(
            [_settings.llmfit_bin, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            return None
        return json.loads(proc.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        return None


def estimate_fit(cfg: RunConfig, hw: HardwareProfile) -> MemoryEstimate:
    """Best-effort fit estimate for a run config on the given hardware.

    Tries llmfit first; on any failure or absence, uses the local estimator.
    """
    model_ref = cfg.base_model
    if cfg.base_model:
        try:
            resolved = resolve_model_ref(cfg.base_model)
            # Adapter configs are tiny; training memory is determined by their
            # base architecture, not the adapter directory/name.
            model_ref = resolved.base_model if resolved.kind == "adapter" else resolved.load_ref
        except ModelReferenceError:
            # Validation reports the actionable error; estimates should remain
            # responsive while a user types an incomplete custom path.
            model_ref = cfg.base_model
    # Generic llmfit inference totals omit this run's optimizer and batch.
    # Use the training-aware breakdown; llmfit remains available for discovery.
    spec = hf_hub.get_model_spec(model_ref) if model_ref else estimator.spec_from_name("1.5b")
    return estimator.estimate(cfg, hw, spec)


def _parse_llmfit_fit(data: Any, hw: HardwareProfile) -> MemoryEstimate | None:
    """Map llmfit's JSON into our MemoryEstimate. Schema-tolerant.

    llmfit's exact JSON keys vary by version, so we probe a few likely fields and
    bail (returning None → fallback) if we can't find a total.
    """
    if not isinstance(data, dict):
        return None
    total = _first_number(data, ["total_vram_mb", "vram_mb", "required_mb", "memory_mb"])
    if total is None:
        return None
    budget = float(hw.vram_total_mb or _settings.vram_budget_mb)
    from app.domain import FitLevel

    safe = budget * _settings.vram_safe_fraction
    fit = FitLevel.FITS if total <= safe * 0.8 else FitLevel.TIGHT if total <= safe else FitLevel.WONT_FIT
    return MemoryEstimate(
        weights_mb=_first_number(data, ["weights_mb"]) or 0.0,
        total_mb=float(total),
        budget_mb=budget,
        fit=fit,
        source="llmfit",
        notes=["Estimate from llmfit"],
    )


def _first_number(d: dict, keys: list[str]) -> float | None:
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def recommend(hw: HardwareProfile) -> list[dict[str, Any]] | None:
    """Hardware-aware model recommendations from llmfit, if available."""
    data = _run_json(["recommend", "--json"])
    if isinstance(data, dict):
        return data.get("recommendations") or data.get("models")
    if isinstance(data, list):
        return data
    return None
