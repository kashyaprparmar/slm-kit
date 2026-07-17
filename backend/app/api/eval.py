"""Eval Lab endpoints: playground generation + eval harness + results."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from app.core.eval_manager import log_path, manager
from app.core.log_capture import read_log_tail
from app.db.models import EvalResult
from app.db.session import engine
from app.integrations import judge

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

    model_ref: str
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.1


@router.post("/generate")
async def generate(body: GenerateBody):
    if not body.model_ref.strip():
        raise HTTPException(400, "A model reference is required.")
    try:
        gen_id = await manager.start_generate(body.model_dump())
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"gen_id": gen_id}


class EvalBody(BaseModel):
    models: list[str]                     # HF repo ids or local paths (1 = single, 2+ = comparison)
    dataset_id: int
    max_samples: int = 50
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 0.95
    metrics: list[str] = ["exact_match", "token_f1", "rouge_l", "bleu"]
    judge: bool = False


@router.post("/run")
async def run_eval(body: EvalBody):
    models = [m.strip() for m in body.models if m.strip()]
    if not models:
        raise HTTPException(400, "Provide at least one model.")
    if body.judge and not judge.judge_available():
        raise HTTPException(400, "LLM-as-judge requested but no judge API key is configured.")
    cfg = body.model_dump()
    cfg["models"] = models
    try:
        eval_id = await manager.start_eval(cfg)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"eval_id": eval_id}


@router.get("/results")
def list_results():
    with Session(engine) as db:
        return db.exec(select(EvalResult).order_by(EvalResult.created_at.desc())).all()


@router.get("/results/{eval_id}")
def get_result(eval_id: int):
    with Session(engine) as db:
        row = db.get(EvalResult, eval_id)
        if not row:
            raise HTTPException(404, "Eval result not found")
        return row
