"""Bundled starter datasets — one per pillar — so the pipeline runs on day one.

Installing copies each file into the datasets dir, validates it, and registers a
``Dataset`` row (idempotent: re-installing updates existing sample rows).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from sqlmodel import Session, select

from app.config import get_settings
from app.datasets import validate as dsvalidate
from app.db.models import Dataset
from app.db.session import engine
from app.domain import DatasetKind

_settings = get_settings()
_SRC = Path(__file__).resolve().parents[2] / "data" / "samples"


@dataclass
class SampleSpec:
    name: str
    filename: str
    kind: DatasetKind
    description: str


SAMPLES: list[SampleSpec] = [
    # --- Pillar 1: from-scratch pretraining corpora ---
    SampleSpec(
        name="Sample: Tiny Story Corpus",
        filename="tiny_corpus.txt",
        kind=DatasetKind.PRETRAIN_CORPUS,
        description="Small original prose corpus for a from-scratch pretraining smoke test.",
    ),
    SampleSpec(
        name="Sample: Science Facts Corpus",
        filename="science_facts_corpus.txt",
        kind=DatasetKind.PRETRAIN_CORPUS,
        description="Explanatory science prose (biology, physics, chemistry) — a second, distinct pretraining domain.",
    ),
    # --- Pillar 2: domain-adaptive continued pretraining corpora ---
    SampleSpec(
        name="Sample: Finance Domain Corpus",
        filename="finance_domain_corpus.txt",
        kind=DatasetKind.DOMAIN_CORPUS,
        description="Financial-report-style prose for continued/domain-adaptive pretraining.",
    ),
    SampleSpec(
        name="Sample: Legal Contract Corpus",
        filename="legal_contract_corpus.txt",
        kind=DatasetKind.DOMAIN_CORPUS,
        description="Legal/contract-style prose — a second domain-adaptation corpus distinct from finance.",
    ),
    # --- Pillar 3: instruction fine-tuning sets ---
    SampleSpec(
        name="Sample: Finance QA (instruction)",
        filename="finance_qa.jsonl",
        kind=DatasetKind.INSTRUCTION,
        description="Finance question/answer pairs for a QLoRA instruction fine-tune.",
    ),
    SampleSpec(
        name="Sample: Coding QA (instruction)",
        filename="coding_qa.jsonl",
        kind=DatasetKind.INSTRUCTION,
        description="Programming concept Q&A pairs — a general-purpose coding-assistant fine-tune set.",
    ),
    SampleSpec(
        name="Sample: Customer Support QA (instruction)",
        filename="support_qa.jsonl",
        kind=DatasetKind.INSTRUCTION,
        description="Customer-support response examples — tone/empathy-focused instruction fine-tune set.",
    ),
    # --- Held-out eval sets, matched to the instruction sets above ---
    SampleSpec(
        name="Sample: Finance QA (eval)",
        filename="finance_qa_eval.jsonl",
        kind=DatasetKind.EVAL,
        description="Held-out finance QA set for evaluation and side-by-side comparison.",
    ),
    SampleSpec(
        name="Sample: Coding QA (eval)",
        filename="coding_qa_eval.jsonl",
        kind=DatasetKind.EVAL,
        description="Held-out coding QA set, disjoint from the coding instruction set above.",
    ),
    SampleSpec(
        name="Sample: Customer Support QA (eval)",
        filename="support_qa_eval.jsonl",
        kind=DatasetKind.EVAL,
        description="Held-out support QA set, disjoint from the support instruction set above.",
    ),
]


def install_samples() -> list[Dataset]:
    dest_dir = _settings.datasets_dir / "samples"
    dest_dir.mkdir(parents=True, exist_ok=True)
    installed: list[Dataset] = []

    # One commit at the end, then refresh every row while the session is still
    # open. Committing per-row (as this used to) expires ALL objects already
    # tracked by the session on each commit — not just the one just written —
    # so any row appended before the last one would raise DetachedInstanceError
    # the moment a caller reads it after this function returns.
    with Session(engine) as db:
        for spec in SAMPLES:
            src = _SRC / spec.filename
            if not src.exists():
                continue
            dest = dest_dir / spec.filename
            shutil.copyfile(src, dest)

            report, stats = dsvalidate.validate(dest, spec.kind)
            existing = db.exec(
                select(Dataset).where(Dataset.name == spec.name, Dataset.is_sample)
            ).first()
            row = existing or Dataset(name=spec.name, kind=spec.kind.value, path=str(dest),
                                      fmt=stats.fmt, is_sample=True)
            row.path = str(dest)
            row.kind = spec.kind.value
            row.fmt = stats.fmt
            row.num_rows = stats.num_rows
            row.num_tokens_est = stats.num_tokens_est
            row.size_bytes = stats.size_bytes
            row.validation = report.model_dump()
            db.add(row)
            installed.append(row)
        db.commit()
        for row in installed:
            db.refresh(row)
    return installed
