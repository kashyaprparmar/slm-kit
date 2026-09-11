"""Small, dependency-light snapshots attached to every training config."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import sys
from pathlib import Path

from app.db.models import Dataset

_PACKAGES = ("torch", "transformers", "peft", "trl", "unsloth", "datasets", "tokenizers")


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _fingerprint(path: str) -> str | None:
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def snapshot(dataset: Dataset | None, *, backend: str, revision: str | None) -> dict:
    validation = dataset.validation or {} if dataset else {}
    fingerprint = validation.get("fingerprint") if isinstance(validation, dict) else None
    if dataset and not fingerprint:
        fingerprint = _fingerprint(dataset.path)
    return {
        "schema_version": 1,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "backend": backend,
        "model_revision": revision,
        "dataset_id": dataset.id if dataset else None,
        "dataset_fingerprint_sha256": fingerprint,
        "packages": {name: version for name in _PACKAGES if (version := _version(name))},
    }
