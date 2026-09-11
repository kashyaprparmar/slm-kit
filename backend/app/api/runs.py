"""Training run lifecycle: estimate, create/enqueue, inspect, cancel, clone."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException
from sqlmodel import Session, select

from app.backends.base import get_backend, list_backends, load_builtin_backends
from app.core.hardware import read_hardware
from app.core.log_capture import read_log_tail
from app.core.queue import queue
from app.core.reproducibility import snapshot as reproducibility_snapshot
from app.core.runner import log_path
from app.db.models import Checkpoint, Dataset, ModelArtifact, Run
from app.db.session import engine
from app.domain import DatasetKind, Method, RunConfig, RunStatus, TaskType, ValidationReport

router = APIRouter(prefix="/api/runs", tags=["runs"])

load_builtin_backends()

_ALLOWED_DATASET_KINDS = {
    TaskType.PRETRAIN: {DatasetKind.PRETRAIN_CORPUS.value, DatasetKind.DOMAIN_CORPUS.value},
    TaskType.CONTINUED_PRETRAIN: {DatasetKind.DOMAIN_CORPUS.value, DatasetKind.PRETRAIN_CORPUS.value},
    TaskType.FINETUNE: {DatasetKind.INSTRUCTION.value},
}


def _apply_method_defaults(cfg: RunConfig) -> RunConfig:
    if cfg.method == Method.QLORA:
        cfg.load_in_4bit = True
    elif cfg.method in (Method.LORA, Method.DORA, Method.FULL):
        cfg.load_in_4bit = False
    if cfg.method == Method.DORA:
        cfg.lora.use_dora = True
    return cfg


def _validate_dataset(cfg: RunConfig, report: ValidationReport) -> None:
    if cfg.dataset_id is None:
        report.error("A dataset is required for this training run.")
        return
    with Session(engine) as db:
        dataset = db.get(Dataset, cfg.dataset_id)
    if dataset is None:
        report.error(f"Dataset {cfg.dataset_id} was not found.")
        return
    allowed = _ALLOWED_DATASET_KINDS[cfg.task]
    if dataset.kind not in allowed:
        report.error(
            f"Dataset '{dataset.name}' is {dataset.kind}, but {cfg.task.value} expects "
            f"one of: {', '.join(sorted(allowed))}."
        )
    if dataset.validation and dataset.validation.get("ok") is False:
        report.error(f"Dataset '{dataset.name}' has validation errors. Fix or replace it before training.")


@router.get("/backends")
def backends():
    return [
        {
            "name": b.name,
            "tasks": [t.value for t in b.supported_tasks],
            "methods": [m.value for m in b.supported_methods],
            "platforms": ["linux", "windows", "wsl"],
            "quantization": ["nf4", "fp16", "bf16"] if b.name != "scratch" else ["fp32"],
            "architectures": ["decoder-only causal language models"] if b.name != "scratch" else ["SLM Kit GPT"],
            "requirements": ["torch", "tokenizers"] if b.name == "scratch" else ["torch", "transformers", "trl", "peft"],
        }
        for b in list_backends()
    ]


@router.post("/estimate")
def estimate(cfg: RunConfig):
    cfg = _apply_method_defaults(cfg)
    hw = read_hardware()
    try:
        backend = get_backend(cfg.backend)
    except KeyError as e:
        raise HTTPException(400, str(e))
    est = backend.estimate_footprint(cfg, hw)
    report = backend.validate_config(cfg, hw)
    _validate_dataset(cfg, report)
    return {"estimate": est, "validation": report, "hardware": hw}


@router.get("")
def list_runs():
    with Session(engine) as db:
        return db.exec(select(Run).order_by(Run.created_at.desc())).all()


@router.get("/{run_id}")
def get_run(run_id: int):
    with Session(engine) as db:
        run = db.get(Run, run_id)
        if not run:
            raise HTTPException(404, "Run not found")
        checkpoints = db.exec(select(Checkpoint).where(Checkpoint.run_id == run_id)).all()
    return {"run": run, "checkpoints": checkpoints, "queue_position": queue.position(run_id)}


@router.post("", status_code=201)
async def create_run(cfg: RunConfig):
    cfg = _apply_method_defaults(cfg)
    hw = read_hardware()
    try:
        backend = get_backend(cfg.backend)
    except KeyError as e:
        raise HTTPException(400, str(e))

    report = backend.validate_config(cfg, hw)
    _validate_dataset(cfg, report)
    if not report.ok:
        raise HTTPException(422, detail={"validation": report.model_dump()})
    est = backend.estimate_footprint(cfg, hw)

    with Session(engine) as db:
        dataset = db.get(Dataset, cfg.dataset_id) if cfg.dataset_id else None
        previous_snapshot = cfg.extra.get("reproducibility")
        cfg.extra = {
            **cfg.extra,
            **({"source_reproducibility": previous_snapshot} if previous_snapshot else {}),
            "reproducibility": reproducibility_snapshot(
                dataset,
                backend=cfg.backend,
                revision=cfg.revision,
            ),
        }
        run = Run(
            name=cfg.output_name,
            task=cfg.task.value,
            method=cfg.method.value,
            backend=cfg.backend,
            base_model=cfg.base_model,
            dataset_id=cfg.dataset_id,
            status=RunStatus.QUEUED.value,
            config=cfg.model_dump(mode="json"),
            estimate=est.model_dump(),
            hardware=hw.model_dump(),
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        run_id = run.id

    pos = await queue.enqueue(run_id)
    return {"run_id": run_id, "queue_size": pos, "estimate": est}


@router.post("/{run_id}/cancel")
async def cancel_run(run_id: int):
    ok = await queue.cancel(run_id)
    if not ok:
        raise HTTPException(409, "Run is not queued or running.")
    return {"cancelled": run_id}


@router.delete("/{run_id}")
def delete_run(run_id: int):
    """Delete an inactive, unregistered run and its files after safety checks."""
    from app.config import get_settings

    with Session(engine) as db:
        run = db.get(Run, run_id)
        if not run:
            raise HTTPException(404, "Run not found")
        if run.status in {RunStatus.QUEUED.value, RunStatus.RUNNING.value}:
            raise HTTPException(409, "Stop this run before deleting it.")
        if db.exec(select(ModelArtifact).where(ModelArtifact.run_id == run_id)).first():
            raise HTTPException(409, "This run has registered model artifacts. Keep it to preserve model lineage.")
        for checkpoint in db.exec(select(Checkpoint).where(Checkpoint.run_id == run_id)).all():
            db.delete(checkpoint)
        output_dir = Path(run.output_dir).resolve() if run.output_dir else None
        runs_root = get_settings().runs_dir.resolve()
        if output_dir and runs_root not in output_dir.parents:
            raise HTTPException(409, "Run files are outside the managed runs directory and were not deleted.")
        db.delete(run)
        db.commit()
    if output_dir and output_dir.is_dir():
        shutil.rmtree(output_dir)
    return {"deleted": run_id}


@router.post("/{run_id}/clone")
def clone_run(run_id: int):
    with Session(engine) as db:
        run = db.get(Run, run_id)
        if not run:
            raise HTTPException(404, "Run not found")
        cfg = RunConfig(**run.config)
    cfg.output_name = f"{cfg.output_name}-clone"
    return {"config": cfg}


@router.get("/{run_id}/export-config")
def export_config(run_id: int):
    with Session(engine) as db:
        run = db.get(Run, run_id)
        if not run:
            raise HTTPException(404, "Run not found")
        cfg = RunConfig(**run.config)
    backend = get_backend(cfg.backend)
    return backend.export_config(cfg)


@router.get("/{run_id}/metrics")
def run_metrics(run_id: int):
    """Full metric history for replotting a finished run's curves."""
    from app.config import get_settings

    path = get_settings().runs_dir / str(run_id) / "metrics.jsonl"
    rows: list[dict] = []
    if path.exists():
        import json

        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return {"metrics": rows}


@router.get("/{run_id}/logs")
def run_logs(run_id: int, lines: int = 2000):
    """Full persisted log history for a run — works for finished runs too, not
    just ones currently streaming over the WebSocket."""
    return {"lines": read_log_tail(log_path(run_id), lines)}


@router.post("/{run_id}/rerun", status_code=201)
async def rerun(run_id: int):
    """Re-queue a new run from a previous run's stored config (config-as-data)."""
    with Session(engine) as db:
        run = db.get(Run, run_id)
        if not run:
            raise HTTPException(404, "Run not found")
        cfg = RunConfig(**run.config)
    return await create_run(cfg)
