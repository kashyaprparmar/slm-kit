"""SQLModel tables — the local metadata/lineage registry.

Source of truth for *model files* is Hugging Face Hub (once published) or the
local ``models/`` dir; this DB stores run metadata, config-as-data, dataset
records, and the lineage that ties them together.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Column
from sqlalchemy.types import JSON
from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Dataset(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    kind: str                      # DatasetKind value
    path: str                      # absolute path on disk
    fmt: str                       # txt | jsonl | json | csv | parquet
    num_rows: Optional[int] = None
    num_tokens_est: Optional[int] = None
    size_bytes: Optional[int] = None
    is_sample: bool = False
    validation: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=_utcnow)


class Run(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    task: str                      # TaskType value
    method: str                    # Method value
    backend: str
    base_model: str
    dataset_id: Optional[int] = Field(default=None, foreign_key="dataset.id")
    status: str = "queued"         # RunStatus value

    config: dict = Field(default_factory=dict, sa_column=Column(JSON))
    estimate: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    metrics: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    hardware: Optional[dict] = Field(default=None, sa_column=Column(JSON))

    output_dir: Optional[str] = None
    hf_repo: Optional[str] = None
    error: Optional[str] = None

    created_at: datetime = Field(default_factory=_utcnow)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class Checkpoint(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="run.id")
    step: int
    path: str
    is_final: bool = False
    created_at: datetime = Field(default_factory=_utcnow)


class ModelArtifact(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    kind: str                      # ArtifactKind value
    run_id: Optional[int] = Field(default=None, foreign_key="run.id")
    base_model: Optional[str] = None
    local_path: Optional[str] = None
    hf_repo: Optional[str] = None
    published: bool = False
    status: str = "ready"          # ready | quantizing | failed
    error: Optional[str] = None
    meta: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=_utcnow)


class EvalResult(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    model_ref: str                 # HF repo id or local path evaluated
    dataset_id: Optional[int] = Field(default=None, foreign_key="dataset.id")
    run_id: Optional[int] = Field(default=None, foreign_key="run.id")
    scores: dict = Field(default_factory=dict, sa_column=Column(JSON))
    detail: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=_utcnow)
