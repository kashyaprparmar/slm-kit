"""SQLModel tables — the local metadata/lineage registry.

Source of truth for *model files* is Hugging Face Hub (once published) or the
local ``models/`` dir; this DB stores run metadata, config-as-data, dataset
records, and the lineage that ties them together.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Column
from sqlalchemy.types import JSON
from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Dataset(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    kind: str                      # DatasetKind value
    path: str                      # absolute path on disk
    fmt: str                       # txt | jsonl | json | csv | parquet
    num_rows: int | None = None
    num_tokens_est: int | None = None
    size_bytes: int | None = None
    is_sample: bool = False
    validation: dict | None = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=_utcnow)


class DatasetRecipe(SQLModel, table=True):
    """Immutable transformation intent; outputs point back to this recipe."""

    id: int | None = Field(default=None, primary_key=True)
    source_version_id: int = Field(foreign_key="datasetversion.id")
    name: str
    config: dict = Field(default_factory=dict, sa_column=Column(JSON))
    fingerprint: str
    created_at: datetime = Field(default_factory=_utcnow)


class DatasetVersion(SQLModel, table=True):
    """Immutable reference to one source or prepared dataset file."""

    id: int | None = Field(default=None, primary_key=True)
    dataset_id: int = Field(foreign_key="dataset.id")
    parent_version_id: int | None = Field(default=None, foreign_key="datasetversion.id")
    fingerprint: str
    path: str
    fmt: str
    schema_info: dict = Field(default_factory=dict, sa_column=Column("schema", JSON))
    split: str = "source"
    num_rows: int | None = None
    num_tokens_est: int | None = None
    size_bytes: int | None = None
    created_at: datetime = Field(default_factory=_utcnow)


class DatasetSplit(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    recipe_id: int = Field(foreign_key="datasetrecipe.id")
    train_version_id: int = Field(foreign_key="datasetversion.id")
    validation_version_id: int | None = Field(default=None, foreign_key="datasetversion.id")
    test_version_id: int | None = Field(default=None, foreign_key="datasetversion.id")
    seed: int
    fractions: dict = Field(default_factory=dict, sa_column=Column(JSON))
    fingerprint: str
    created_at: datetime = Field(default_factory=_utcnow)


class TokenizerArtifact(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    model_ref: str
    revision: str | None = None
    resolved_revision: str | None = None
    fingerprint: str
    path: str | None = None
    config: dict = Field(default_factory=dict, sa_column=Column(JSON))
    status: str = "ready"
    created_at: datetime = Field(default_factory=_utcnow)


class DatasetProfile(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    dataset_version_id: int = Field(foreign_key="datasetversion.id")
    tokenizer_artifact_id: int | None = Field(default=None, foreign_key="tokenizerartifact.id")
    kind: str
    cache_key: str
    config: dict = Field(default_factory=dict, sa_column=Column(JSON))
    stats: dict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=_utcnow)


class Project(SQLModel, table=True):
    """Persistent workspace state without coupling existing run/dataset rows."""

    id: int | None = Field(default=None, primary_key=True)
    name: str
    description: str = ""
    state: dict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class ProjectRun(SQLModel, table=True):
    project_id: int = Field(foreign_key="project.id", primary_key=True)
    run_id: int = Field(foreign_key="run.id", primary_key=True)
    created_at: datetime = Field(default_factory=_utcnow)


class Run(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    task: str                      # TaskType value
    method: str                    # Method value
    backend: str
    base_model: str
    dataset_id: int | None = Field(default=None, foreign_key="dataset.id")
    status: str = "queued"         # RunStatus value

    config: dict = Field(default_factory=dict, sa_column=Column(JSON))
    estimate: dict | None = Field(default=None, sa_column=Column(JSON))
    metrics: dict | None = Field(default=None, sa_column=Column(JSON))
    hardware: dict | None = Field(default=None, sa_column=Column(JSON))

    output_dir: str | None = None
    hf_repo: str | None = None
    error: str | None = None

    created_at: datetime = Field(default_factory=_utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None


class Checkpoint(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="run.id")
    step: int
    path: str
    is_final: bool = False
    created_at: datetime = Field(default_factory=_utcnow)


class ModelArtifact(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    kind: str                      # ArtifactKind value
    run_id: int | None = Field(default=None, foreign_key="run.id")
    base_model: str | None = None
    local_path: str | None = None
    hf_repo: str | None = None
    published: bool = False
    status: str = "ready"          # ready | quantizing | failed
    error: str | None = None
    meta: dict | None = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=_utcnow)


class EvalResult(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    model_ref: str                 # HF repo id or local path evaluated
    dataset_id: int | None = Field(default=None, foreign_key="dataset.id")
    run_id: int | None = Field(default=None, foreign_key="run.id")
    scores: dict = Field(default_factory=dict, sa_column=Column(JSON))
    detail: dict | None = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=_utcnow)
