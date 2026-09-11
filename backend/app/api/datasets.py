"""Dataset management: upload, validate, preview, samples, HF Hub import."""

from __future__ import annotations

import asyncio
import re
import uuid
from dataclasses import asdict
from pathlib import Path

import aiofiles
from fastapi import APIRouter, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.config import get_settings
from app.datasets import hf_import
from app.datasets import samples as sample_mod
from app.datasets import validate as dsvalidate
from app.db.models import Dataset, EvalResult, Run
from app.db.session import engine
from app.domain import DatasetKind

router = APIRouter(prefix="/api/datasets", tags=["datasets"])
_settings = get_settings()
_REPO_ID_RE = re.compile(r"^[\w.\-]+(/[\w.\-]+)?$")


@router.get("")
def list_datasets():
    with Session(engine) as db:
        return db.exec(select(Dataset).order_by(Dataset.created_at.desc())).all()


# NOTE: every literal-path route (hf-capabilities, upload, install-samples,
# import-hf) MUST be declared before the parameterized `/{dataset_id}` routes
# below. FastAPI/Starlette matches routes in registration order, and
# `dataset_id: int` would otherwise greedily match e.g. "hf-capabilities" as
# its path param and fail with a 422 before ever reaching the literal route.
@router.get("/hf-capabilities")
def hf_capabilities():
    return {"available": hf_import.hf_datasets_available()}


class HFImportBody(BaseModel):
    repo_id: str
    config: str | None = None
    split: str = "train"
    kind: str
    name: str | None = None
    max_rows: int = 2000


@router.post("/import-hf")
def import_from_hf(body: HFImportBody):
    try:
        dk = DatasetKind(body.kind)
    except ValueError:
        raise HTTPException(400, f"Invalid kind '{body.kind}'. Expected one of {[k.value for k in DatasetKind]}")
    if not _REPO_ID_RE.match(body.repo_id):
        raise HTTPException(400, "repo_id must look like 'username/dataset-name' (or 'dataset-name').")
    if not (1 <= body.max_rows <= 50_000):
        raise HTTPException(400, "max_rows must be between 1 and 50,000.")

    dest_dir = _settings.datasets_dir / "hf-imports"
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe_name = body.repo_id.replace("/", "__")
    dest = dest_dir / f"{safe_name}.jsonl"
    counter = 1
    while dest.exists():
        dest = dest_dir / f"{safe_name}-{counter}.jsonl"
        counter += 1

    try:
        n_rows = hf_import.import_dataset(body.repo_id, dest, body.config, body.split, body.max_rows)
    except hf_import.HFImportError as e:
        raise HTTPException(400, str(e))

    report, stats = dsvalidate.validate(dest, dk)
    with Session(engine) as db:
        row = Dataset(
            name=body.name or f"HF: {body.repo_id}",
            kind=dk.value,
            path=str(dest),
            fmt=stats.fmt,
            num_rows=stats.num_rows,
            num_tokens_est=stats.num_tokens_est,
            size_bytes=stats.size_bytes,
            is_sample=False,
            validation=report.model_dump(),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
    return {"dataset": row, "validation": report, "rows_imported": n_rows}


@router.post("/install-samples")
def install_samples():
    return {"installed": sample_mod.install_samples()}


@router.get("/{dataset_id}")
def get_dataset(dataset_id: int):
    with Session(engine) as db:
        row = db.get(Dataset, dataset_id)
        if not row:
            raise HTTPException(404, "Dataset not found")
        return row


@router.get("/{dataset_id}/preview")
def preview(dataset_id: int):
    with Session(engine) as db:
        row = db.get(Dataset, dataset_id)
        if not row:
            raise HTTPException(404, "Dataset not found")
    report, stats = dsvalidate.validate(row.path, DatasetKind(row.kind))
    return {
        "validation": report,
        "stats": asdict(stats),
    }


@router.post("/upload")
async def upload(file: UploadFile, kind: str = Form(...), name: str | None = Form(None)):
    try:
        dk = DatasetKind(kind)
    except ValueError:
        raise HTTPException(400, f"Invalid kind '{kind}'. Expected one of {[k.value for k in DatasetKind]}")

    dest_dir = _settings.datasets_dir / "uploads"
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Sanitize: strip any path components from the client-supplied name, and
    # never overwrite an existing upload with the same name.
    safe_name = Path((file.filename or "upload.dat").replace("\\", "/")).name
    if Path(safe_name).suffix.lower() not in {".json", ".jsonl", ".csv", ".txt", ".parquet"}:
        raise HTTPException(400, "Choose a JSON, JSONL, CSV, TXT or Parquet file.")
    safe_name = f"{uuid.uuid4().hex[:10]}-{safe_name}"
    dest = dest_dir / safe_name
    counter = 1
    while dest.exists():
        dest = dest_dir / f"{Path(safe_name).stem}-{counter}{Path(safe_name).suffix}"
        counter += 1
    max_bytes = _settings.max_upload_mb * 1024 * 1024
    written = 0
    try:
        async with aiofiles.open(dest, "wb") as output:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        413, f"Upload exceeds the {_settings.max_upload_mb} MB limit."
                    )
                await output.write(chunk)
    except Exception:
        dest.unlink(missing_ok=True)
        raise

    report, stats = await asyncio.to_thread(dsvalidate.validate, dest, dk)
    with Session(engine) as db:
        row = Dataset(
            name=name or (file.filename or "dataset"),
            kind=dk.value,
            path=str(dest),
            fmt=stats.fmt,
            num_rows=stats.num_rows,
            num_tokens_est=stats.num_tokens_est,
            size_bytes=stats.size_bytes,
            is_sample=False,
            validation=report.model_dump(),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
    return {"dataset": row, "validation": report}


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: int):
    with Session(engine) as db:
        row = db.get(Dataset, dataset_id)
        if not row:
            raise HTTPException(404, "Dataset not found")
        if row.is_sample:
            raise HTTPException(409, "Bundled sample datasets cannot be deleted.")
        used = db.exec(select(Run).where(Run.dataset_id == dataset_id)).first()
        evaluated = db.exec(select(EvalResult).where(EvalResult.dataset_id == dataset_id)).first()
        if used or evaluated:
            raise HTTPException(409, "This dataset belongs to recorded runs or evaluations. Keep it for reproducibility.")
        # Remove the uploaded/copied file, but never touch files outside our home.
        try:
            p = Path(row.path).resolve()
            if _settings.datasets_dir.resolve() in p.parents:
                p.unlink(missing_ok=True)
        except OSError as exc:
            raise HTTPException(500, "Could not remove the dataset file.") from exc
        db.delete(row)
        db.commit()
    return {"deleted": dataset_id}


class PrepareBody(BaseModel):
    columns: dict[str, str] = Field(default_factory=dict)
    test_fraction: float = Field(default=.1, ge=0, le=.5)
    shuffle: bool = True
    deduplicate: bool = True
    drop_invalid: bool = False
    seed: int = Field(default=42, ge=0)


@router.post("/{dataset_id}/prepare", status_code=201)
def prepare_dataset(dataset_id: int, body: PrepareBody):
    from app.datasets.prepare import prepare
    with Session(engine) as db:
        source = db.get(Dataset, dataset_id)
        if not source:
            raise HTTPException(404, "Dataset not found.")
        if set(body.columns) - {"prompt", "response", "text", "messages"}:
            raise HTTPException(422, "Map prompt, response, text or messages columns.")
        destination = _settings.datasets_dir / "prepared"
        destination.mkdir(parents=True, exist_ok=True)
        try:
            prepared, dropped = prepare(source.path, destination, DatasetKind(source.kind), body.columns,
                                         body.test_fraction, body.shuffle, body.deduplicate, body.drop_invalid, body.seed)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc
        records = []
        for split, path, kind, report, stats in prepared:
            row = Dataset(name=f"{source.name} ({split})", kind=kind.value, path=str(path), fmt=stats.fmt,
                          num_rows=stats.num_rows, num_tokens_est=stats.num_tokens_est,
                          size_bytes=stats.size_bytes, validation={**report.model_dump(), "preparation": body.model_dump(),
                          "source_dataset_id": source.id, "fingerprint": stats.fingerprint})
            db.add(row)
            records.append(row)
        db.commit()
        for row in records:
            db.refresh(row)
    return {"datasets": records, "dropped_rows": dropped}
