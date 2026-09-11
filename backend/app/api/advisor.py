"""Model Advisor: recommend a base-model + method combo for a task on this box."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel
from sqlmodel import Session

from app.core.hardware import read_hardware
from app.db.models import Dataset
from app.db.session import engine
from app.domain import FitLevel, Method, RunConfig, TaskType
from app.integrations import llmfit

router = APIRouter(prefix="/api/advisor", tags=["advisor"])


class SettingsBody(BaseModel):
    config: RunConfig
    preset: Literal["fast", "balanced", "best_quality", "lowest_memory"] = "balanced"


@router.post("/settings")
def training_settings(body: SettingsBody):
    from fastapi import HTTPException

    from app.integrations.recommendations import recommend_settings
    from app.model_refs import ModelReferenceError
    with Session(engine) as db:
        dataset = db.get(Dataset, body.config.dataset_id) if body.config.dataset_id else None
    try:
        return recommend_settings(body.config, read_hardware(), body.preset,
                                  (dataset.num_rows or 0) if dataset else 0,
                                  (dataset.num_tokens_est or 0) if dataset else 0)
    except ModelReferenceError as exc:
        raise HTTPException(400, str(exc)) from exc

# Curated 8GB-friendly candidates with rough quality tiers (1=basic … 5=strong).
# Repo ids are pre-quantized (unsloth-bnb-4bit) where available for faster download.
CANDIDATES: list[dict] = [
    # --- Latest small models (2026) ---
    {"repo": "LiquidAI/LFM2-350M", "params_b": 0.35, "quality": 2, "ctx": 32768, "latest": True},
    {"repo": "unsloth/Qwen3-0.6B-unsloth-bnb-4bit", "params_b": 0.6, "quality": 2, "ctx": 32768, "latest": True},
    {"repo": "LiquidAI/LFM2-700M", "params_b": 0.7, "quality": 3, "ctx": 32768, "latest": True},
    {"repo": "LiquidAI/LFM2-1.2B", "params_b": 1.2, "quality": 3, "ctx": 32768, "latest": True},
    {"repo": "unsloth/Qwen3-1.7B-unsloth-bnb-4bit", "params_b": 1.7, "quality": 4, "ctx": 32768, "latest": True},
    {"repo": "unsloth/gemma-3n-E2B-unsloth-bnb-4bit", "params_b": 2.0, "quality": 4, "ctx": 32768, "latest": True},
    {"repo": "unsloth/SmolLM3-3B", "params_b": 3.0, "quality": 4, "ctx": 65536, "latest": True},
    {"repo": "unsloth/Phi-4-mini-instruct-bnb-4bit", "params_b": 3.8, "quality": 5, "ctx": 131072, "latest": True},
    {"repo": "unsloth/Qwen3-4B-unsloth-bnb-4bit", "params_b": 4.0, "quality": 5, "ctx": 32768, "latest": True},
    {"repo": "unsloth/Qwen3-8B-unsloth-bnb-4bit", "params_b": 8.0, "quality": 5, "ctx": 32768, "latest": True},
    # --- Established / well-tested ---
    {"repo": "unsloth/Qwen2.5-0.5B-Instruct", "params_b": 0.5, "quality": 2, "ctx": 32768},
    {"repo": "unsloth/SmolLM2-360M-Instruct", "params_b": 0.36, "quality": 1, "ctx": 8192},
    {"repo": "unsloth/Llama-3.2-1B-Instruct", "params_b": 1.2, "quality": 3, "ctx": 131072},
    {"repo": "unsloth/Qwen2.5-1.5B-Instruct", "params_b": 1.5, "quality": 3, "ctx": 32768},
    {"repo": "unsloth/Llama-3.2-3B-Instruct", "params_b": 3.2, "quality": 4, "ctx": 131072},
    {"repo": "unsloth/Qwen2.5-3B-Instruct", "params_b": 3.1, "quality": 4, "ctx": 32768},
    {"repo": "unsloth/gemma-2-2b-it", "params_b": 2.6, "quality": 4, "ctx": 8192},
    {"repo": "unsloth/Phi-3.5-mini-instruct", "params_b": 3.8, "quality": 4, "ctx": 131072},
    {"repo": "unsloth/mistral-7b-instruct-v0.3", "params_b": 7.2, "quality": 5, "ctx": 32768},
    {"repo": "unsloth/Qwen2.5-7B-Instruct", "params_b": 7.6, "quality": 5, "ctx": 32768},
]


class AdvisorBody(BaseModel):
    task: TaskType = TaskType.FINETUNE
    method: Method = Method.QLORA
    priority: str = "balanced"        # fastest | balanced | best_quality
    max_seq_length: int = 1024


@router.post("/recommend")
def recommend(body: AdvisorBody):
    hw = read_hardware()
    results = []
    for cand in CANDIDATES:
        cfg = RunConfig(
            backend="unsloth",
            task=body.task,
            method=body.method,
            base_model=cand["repo"],
            output_name="advisor-probe",
        )
        cfg.train.max_seq_length = body.max_seq_length
        est = llmfit.estimate_fit(cfg, hw)
        if est.fit == FitLevel.WONT_FIT:
            continue
        # Speed proxy: smaller is faster (inverse of params).
        speed_score = 5.0 / (0.4 + cand["params_b"])
        quality_score = cand["quality"]
        headroom = max(0.0, est.budget_mb - est.total_mb) / max(1.0, est.budget_mb)

        if body.priority == "fastest":
            score = speed_score * 2 + headroom * 3
        elif body.priority == "best_quality":
            score = quality_score * 2 + headroom
        else:  # balanced
            score = speed_score + quality_score + headroom * 2
        if cand.get("latest"):
            score += 0.3  # small nudge toward newer architectures, all else equal

        results.append({
            "repo": cand["repo"],
            "params_b": cand["params_b"],
            "method": body.method.value,
            "context_length": cand["ctx"],
            "estimate": est.model_dump(),
            "quality_tier": cand["quality"],
            "fit": est.fit.value,
            "latest": bool(cand.get("latest")),
            "score": round(score, 3),
            "reasoning": _reason(cand, est, body.priority),
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return {
        "hardware": hw,
        "source": "llmfit" if llmfit.llmfit_available() else "fallback-estimator",
        "recommendations": results,
    }


def _reason(cand: dict, est, priority: str) -> str:
    fit_word = {"fits": "fits comfortably", "tight": "a tight fit", "wont_fit": "won't fit"}[est.fit.value]
    return (
        f"{cand['params_b']}B params, quality tier {cand['quality']}/5, {cand['ctx']:,}-token context. "
        f"Predicted ~{est.total_mb:.0f} MB — {fit_word} on your {est.budget_mb:.0f} MB budget. "
        f"Ranked for '{priority}'."
    )
