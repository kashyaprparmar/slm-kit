"""Eval Lab endpoints: playground generation + eval harness + results."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.core.eval_manager import log_path, manager
from app.core.log_capture import read_log_tail
from app.db.models import Dataset, EvalResult
from app.db.session import engine
from app.integrations import judge
from app.model_refs import ModelReferenceError, resolve_model_ref

router = APIRouter(prefix="/api/eval", tags=["eval"])


@router.get("/status")
def status():
    return {
        "busy": manager.busy(),
        "judge_available": judge.judge_available(),
        "current": manager.current(),  # {"kind","id","topic"} or null
        **manager.engine_info(),        # vllm_available, serve_engine_setting, vllm_warm_model
    }


@router.post("/cancel")
async def cancel():
    """Stop the currently-running eval/generation and free the GPU. Also clears a
    stuck 'busy' flag if a subprocess died without releasing it."""
    cancelled = await manager.cancel_current()
    return {"cancelled": cancelled}


@router.get("/logs/{kind}/{ident}")
def get_logs(kind: str, ident: str, lines: int = 2000):
    """Full persisted log history for an eval or playground-generate job —
    works after the job finishes too, not just while it's streaming."""
    if kind not in ("eval", "gen"):
        raise HTTPException(400, "kind must be 'eval' or 'gen'.")
    return {"lines": read_log_tail(log_path(kind, ident), lines)}


class GenerateBody(BaseModel):
    model_config = {"protected_namespaces": ()}  # allow the model_ref field name

    model_ref: str = Field(min_length=1)
    prompt: str = Field(min_length=1, max_length=100_000)
    max_new_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.95, ge=0, le=1)
    top_k: int = Field(default=50, ge=0, le=500)
    repetition_penalty: float = Field(default=1.1, ge=1, le=2)
    system_prompt: str = Field(default="", max_length=32000)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)


@router.post("/generate")
async def generate(body: GenerateBody):
    if not body.model_ref.strip():
        raise HTTPException(400, "A model reference is required.")
    try:
        resolve_model_ref(body.model_ref)
    except ModelReferenceError as exc:
        raise HTTPException(400, str(exc))
    try:
        gen_id = await manager.start_generate(body.model_dump())
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"gen_id": gen_id}


class EvalBody(BaseModel):
    models: list[str] = Field(min_length=1, max_length=8)
    dataset_id: int = Field(ge=1)
    max_samples: int = Field(default=50, ge=1, le=1_000)
    max_new_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0, le=2)
    top_p: float = Field(default=0.95, ge=0, le=1)
    metrics: list[Literal["exact_match", "token_f1", "rouge_l", "bleu", "perplexity"]] = Field(
        default_factory=lambda: ["exact_match", "token_f1", "rouge_l", "bleu"],
        max_length=6,
        min_length=1,
    )
    judge: bool = False


@router.post("/run")
async def run_eval(body: EvalBody):
    models = [m.strip() for m in body.models if m.strip()]
    if not models:
        raise HTTPException(400, "Provide at least one model.")
    if body.judge and not judge.judge_available():
        raise HTTPException(400, "LLM-as-judge requested but no judge API key is configured.")
    if not 1 <= body.max_samples <= 1_000:
        raise HTTPException(400, "max_samples must be between 1 and 1,000.")
    for model_ref in models:
        try:
            resolve_model_ref(model_ref)
        except ModelReferenceError as exc:
            raise HTTPException(400, str(exc))
    with Session(engine) as db:
        dataset = db.get(Dataset, body.dataset_id)
    if dataset is None:
        raise HTTPException(404, "Evaluation dataset not found.")
    if dataset.kind not in {"eval", "instruction"}:
        raise HTTPException(400, "Evaluation requires an eval or instruction dataset.")
    if dataset.validation and dataset.validation.get("ok") is False:
        raise HTTPException(400, "The selected dataset has validation errors.")
    cfg = body.model_dump()
    cfg["models"] = list(dict.fromkeys(models))
    try:
        eval_id = await manager.start_eval(cfg)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"eval_id": eval_id}


@router.get("/results")
def list_results():
    with Session(engine) as db:
        return db.exec(select(EvalResult).order_by(EvalResult.created_at.desc())).all()


@router.get("/generation/{gen_id}")
def generation_result(gen_id: str):
    import json
    import re
    if not re.fullmatch(r"[a-f0-9]{12}", gen_id):
        raise HTTPException(400, "Invalid generation ID.")
    path = log_path("gen", gen_id).parent / "result.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"status": "running" if (manager.current() or {}).get("id") == gen_id else "unknown"}


@router.get("/results/{eval_id}")
def get_result(eval_id: int):
    with Session(engine) as db:
        row = db.get(EvalResult, eval_id)
        if not row:
            raise HTTPException(404, "Eval result not found")
        return row
