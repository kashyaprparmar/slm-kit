"""Import a dataset directly from the Hugging Face Hub into the local registry.

Uses the ``datasets`` library in *streaming* mode so we only pull the first
``max_rows`` examples instead of downloading a (potentially huge) dataset in
full — HF Hub hosts everything from 20-row toy sets to multi-terabyte corpora.

``datasets`` isn't a core dependency (kept optional, like every other
network/GPU-adjacent integration in this app) — if it's missing, this raises a
clear, actionable error instead of failing to import at module load time.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from app.core.logging_config import get_logger

log = get_logger(__name__)


class HFImportError(Exception):
    """Raised for any import failure the caller should show to the user."""


def hf_datasets_available() -> bool:
    return importlib.util.find_spec("datasets") is not None


def import_dataset(
    repo_id: str,
    dest_path: Path,
    config: str | None = None,
    split: str = "train",
    max_rows: int = 2000,
) -> int:
    """Stream up to ``max_rows`` rows from an HF dataset repo into a local JSONL
    file. Returns the number of rows written."""
    if not hf_datasets_available():
        raise HFImportError(
            "The 'datasets' package is required to import from Hugging Face. "
            "Install it with: pip install datasets"
        )
    from datasets import load_dataset  # lazy — heavy-ish import, network on first call

    try:
        ds = load_dataset(repo_id, config, split=split, streaming=True)
    except Exception as e:
        raise HFImportError(f"Could not load '{repo_id}' (config={config!r}, split={split!r}): {e}") from e

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with dest_path.open("w", encoding="utf-8") as f:
            for row in ds:
                if written >= max_rows:
                    break
                if not isinstance(row, dict):
                    continue
                f.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
                written += 1
    except Exception as e:
        dest_path.unlink(missing_ok=True)
        raise HFImportError(f"Import from '{repo_id}' failed partway through: {e}") from e

    if written == 0:
        dest_path.unlink(missing_ok=True)
        raise HFImportError(f"'{repo_id}' (split={split!r}) yielded no usable rows.")
    log.info("imported %s rows from %s (split=%s) -> %s", written, repo_id, split, dest_path)
    return written
