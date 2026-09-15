"""Dataset version/recipe persistence without changing legacy Dataset handles."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sqlmodel import Session, select

from app.db.models import Dataset, DatasetVersion


def file_fingerprint(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def register_version(
    db: Session,
    dataset: Dataset,
    *,
    parent_version_id: int | None = None,
    split: str = "source",
    schema: dict | None = None,
) -> DatasetVersion:
    fingerprint = file_fingerprint(dataset.path)
    existing = db.exec(select(DatasetVersion).where(
        DatasetVersion.dataset_id == dataset.id,
        DatasetVersion.fingerprint == fingerprint,
        DatasetVersion.split == split,
    )).first()
    if existing:
        return existing
    version = DatasetVersion(
        dataset_id=dataset.id,
        parent_version_id=parent_version_id,
        fingerprint=fingerprint,
        path=dataset.path,
        fmt=dataset.fmt,
        schema_info=schema or {},
        split=split,
        num_rows=dataset.num_rows,
        num_tokens_est=dataset.num_tokens_est,
        size_bytes=dataset.size_bytes,
    )
    db.add(version)
    db.flush()
    return version


def current_version(db: Session, dataset: Dataset) -> DatasetVersion:
    version = db.exec(select(DatasetVersion).where(
        DatasetVersion.dataset_id == dataset.id,
    ).order_by(DatasetVersion.created_at.desc(), DatasetVersion.id.desc())).first()
    return version or register_version(db, dataset)


def backfill_legacy_versions(engine) -> int:
    created = 0
    with Session(engine) as db:
        versioned = set(db.exec(select(DatasetVersion.dataset_id)).all())
        for dataset in db.exec(select(Dataset)).all():
            if dataset.id in versioned or not Path(dataset.path).is_file():
                continue
            schema = {}
            validation = dataset.validation or {}
            if isinstance(validation, dict):
                schema = {"legacy_validation": validation}
            register_version(db, dataset, schema=schema)
            created += 1
        db.commit()
    return created
