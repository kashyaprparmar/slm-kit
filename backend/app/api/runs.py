"""Training run lifecycle: estimate, create/enqueue, inspect, cancel, clone."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException
from sqlmodel import Session, select

from app.backends.base import backend_selector, list_backends, load_builtin_backends
from app.core.hardware import read_hardware
from app.core.log_capture import read_log_tail
from app.core.queue import queue
from app.core.reproducibility import snapshot as reproducibility_snapshot
from app.core.runner import log_path
from app.db.models import Checkpoint, Dataset, EvalResult, ModelArtifact, Project, ProjectRun, Run
from app.db.session import engine
from app.domain import DatasetKind, Method, RunConfig, RunStatus, TaskType, ValidationReport
from app.integrations.hf_hub import get_model_capabilities
from app.model_refs import ModelReferenceError, resolve_model_ref
from app.models.capabilities import SupportState, is_allowed

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


def _validate_model_capabilities(cfg: RunConfig, report: ValidationReport) -> None:
    """Apply the same authoritative compatibility rules to estimate and create."""
    if cfg.task == TaskType.PRETRAIN or cfg.backend == "scratch":
        return
    try:
        resolved = resolve_model_ref(cfg.base_model)
        capabilities = get_model_capabilities(
            resolved.base_model or resolved.load_ref,
            None if resolved.kind == "adapter" else cfg.revision,
        )
    except ModelReferenceError as exc:
        report.error(str(exc))
        return
    if capabilities.architecture_kind == "encoder_decoder":
        report.error(
            "Sequence-to-sequence (encoder-decoder) architectures are not supported by the "
            "causal LM training worker. A separate seq2seq loader, trainer, and evaluation "
            "contract is required."
        )
        return
    operation = {
        TaskType.FINETUNE: "sft",
        TaskType.CONTINUED_PRETRAIN: "continued_pretraining",
    }.get(cfg.task, "full")
    if cfg.method in {Method.LORA, Method.QLORA, Method.DORA}:
        operation = cfg.method.value
    capability = capabilities.training[operation]
    if not is_allowed(capability):
        report.error(f"{capabilities.family}: {capability.reason}")
    elif capability.state == SupportState.EXPERIMENTAL:
        report.warn(f"{capabilities.family} support is experimental: {capability.reason}")
    backend_capability = capabilities.backends.get(cfg.backend)
    if backend_capability and not is_allowed(backend_capability):
        report.error(f"Backend '{cfg.backend}' cannot handle this model: {backend_capability.reason}")
    if cfg.train.max_seq_length and capabilities.context_length and cfg.train.max_seq_length > capabilities.context_length:
        report.error(
            f"Context length {cfg.train.max_seq_length:,} exceeds the model limit of "
            f"{capabilities.context_length:,}."
        )


def _run_plan(cfg: RunConfig, estimate, hardware, report: ValidationReport) -> dict:
    with Session(engine) as db:
        dataset = db.get(Dataset, cfg.dataset_id) if cfg.dataset_id else None
    model = None
    if cfg.task != TaskType.PRETRAIN and cfg.base_model:
        try:
            resolved = resolve_model_ref(cfg.base_model)
            model = get_model_capabilities(
                resolved.base_model or resolved.load_ref,
                None if resolved.kind == "adapter" else cfg.revision,
            )
        except ModelReferenceError:
            pass
    effective_batch = cfg.train.per_device_batch_size * cfg.train.gradient_accumulation * max(1, hardware.gpu_count)
    return {
        "model": cfg.base_model or "new scratch model",
        "revision": cfg.revision,
        "family": model.family if model else "SLM Kit GPT",
        "architecture": model.architecture_kind if model else "scratch",
        "backend": cfg.backend,
        "task": cfg.task.value,
        "method": cfg.method.value,
        "dataset": dataset.name if dataset else None,
        "dataset_rows": dataset.num_rows if dataset else None,
        "dataset_tokens": dataset.num_tokens_est if dataset else None,
        "context_length": cfg.train.max_seq_length,
        "batch_size": cfg.train.per_device_batch_size,
        "effective_batch_size": effective_batch,
        "epochs": cfg.train.epochs,
        "max_steps": cfg.train.max_steps,
        "precision": "nf4" if cfg.load_in_4bit else "bf16/fp16 selected by worker hardware",
        "estimated_vram_mb": estimate.total_mb,
        "safe_vram_mb": estimate.safe_budget_mb,
        "headroom_mb": estimate.headroom_mb,
        "fit": estimate.verdict,
        "expected_outputs": ["run.json", "output.log", "metrics.jsonl", "checkpoints", "final model or adapter"],
        "warnings": [issue.message for issue in report.issues if issue.level != "info"],
        "valid": report.ok,
    }


@router.get("/backends")
def backends():
    result = []
    for backend in list_backends():
        descriptor = backend.capabilities()
        data = descriptor.model_dump(mode="json")
        task_capabilities = data.pop("tasks")
        method_capabilities = data.pop("methods")
        quantization_capabilities = data.pop("quantization")
        # Keep the original array fields while exposing the structured source
        # used to derive them. Existing clients can upgrade incrementally.
        data.update(
            tasks=sorted(task.value for task in descriptor.supported_tasks),
            methods=sorted(method.value for method in descriptor.supported_methods),
            quantization=sorted(quantization_capabilities),
            requirements=descriptor.required_dependencies,
            task_capabilities=task_capabilities,
            method_capabilities=method_capabilities,
            quantization_capabilities=quantization_capabilities,
        )
        result.append(data)
    return result


@router.post("/estimate")
def estimate(cfg: RunConfig):
    cfg = _apply_method_defaults(cfg)
    hw = read_hardware()
    try:
        selection = backend_selector.select(cfg)
    except KeyError as e:
        raise HTTPException(400, str(e))
    backend = selection.backend
    est = backend.estimate_footprint(cfg, hw)
    report = backend.validate_config(cfg, hw)
    selection.capabilities.validate_availability(report, required=False)
    _validate_model_capabilities(cfg, report)
    _validate_dataset(cfg, report)
    return {"estimate": est, "validation": report, "hardware": hw, "plan": _run_plan(cfg, est, hw, report)}


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
        selection = backend_selector.select(cfg)
    except KeyError as e:
        raise HTTPException(400, str(e))
    backend = selection.backend

    report = backend.validate_config(cfg, hw)
    selection.capabilities.validate_availability(report, required=True)
    _validate_model_capabilities(cfg, report)
    _validate_dataset(cfg, report)
    if not report.ok:
        raise HTTPException(422, detail={"validation": report.model_dump()})
    est = backend.estimate_footprint(cfg, hw)

    from app.serving.providers import external_gpu_owner

    owner = await external_gpu_owner()
    if owner:
        raise HTTPException(409, f"GPU is already in use by external provider '{owner}'. Stop it before starting training.")

    with Session(engine) as db:
        dataset = db.get(Dataset, cfg.dataset_id) if cfg.dataset_id else None
        project_id = cfg.extra.get("project_id")
        if project_id is not None:
            if isinstance(project_id, bool) or not isinstance(project_id, int) or not db.get(Project, project_id):
                raise HTTPException(422, "The selected project no longer exists. Select another project or clear the project selection.")
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
        db.flush()
        run_id = run.id
        if isinstance(project_id, int):
            db.add(ProjectRun(project_id=project_id, run_id=run_id))
        db.commit()
        db.refresh(run)

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
        if db.exec(select(EvalResult).where(EvalResult.run_id == run_id)).first():
            raise HTTPException(409, "This run has recorded evaluations. Keep it to preserve evaluation lineage.")
        for link in db.exec(select(ProjectRun).where(ProjectRun.run_id == run_id)).all():
            db.delete(link)
        for checkpoint in db.exec(select(Checkpoint).where(Checkpoint.run_id == run_id)).all():
            db.delete(checkpoint)
        output_dir = Path(run.output_dir).resolve() if run.output_dir else None
        runs_root = get_settings().runs_dir.resolve()
        if output_dir and runs_root not in output_dir.parents:
            raise HTTPException(409, "Run files are outside the managed runs directory and were not deleted.")
        # SQLModel tables have no ORM relationships to order these deletes.
        db.flush()
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
    return backend_selector.select(cfg).backend.export_config(cfg)


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
