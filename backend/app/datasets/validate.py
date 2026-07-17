"""Dataset format/schema/content validation, token estimation, and preview.

Runs in the API process (no torch), so token counts are estimated with a cheap
chars/token heuristic rather than a real tokenizer. The goal is fast, specific
feedback *before* a run is launched — with line numbers where possible.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from app.domain import DatasetKind, ValidationReport

# Rough "is this enough?" thresholds, tuned for 8GB-scale runs.
MIN_INSTRUCTION_ROWS = 50
GOOD_INSTRUCTION_ROWS = 300
MIN_DOMAIN_TOKENS = 100_000
MIN_PRETRAIN_TOKENS = 1_000_000

_CHARS_PER_TOKEN = 4  # heuristic for English-ish text


@dataclass
class DatasetStats:
    fmt: str
    num_rows: int = 0
    num_tokens_est: int = 0
    size_bytes: int = 0
    sample_rows: list = field(default_factory=list)
    token_histogram: list[int] = field(default_factory=list)  # per-row token estimate


def detect_format(path: str | Path) -> str:
    ext = Path(path).suffix.lower().lstrip(".")
    return {"txt": "txt", "jsonl": "jsonl", "json": "json",
            "csv": "csv", "parquet": "parquet"}.get(ext, "txt")


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _extract_text(row: dict) -> str:
    if "messages" in row and isinstance(row["messages"], list):
        return " ".join(str(m.get("content", "")) for m in row["messages"])
    keys = ("instruction", "input", "context", "prompt", "question",
            "output", "response", "answer", "text")
    return " ".join(str(row.get(k, "")) for k in keys if row.get(k))


def _validate_instruction_row(row: dict) -> str | None:
    """Return an error string if the row isn't a usable instruction/chat example."""
    if "messages" in row:
        msgs = row["messages"]
        if not isinstance(msgs, list) or not msgs:
            return "'messages' must be a non-empty list"
        for m in msgs:
            if not isinstance(m, dict) or "role" not in m or "content" not in m:
                return "each message needs 'role' and 'content'"
        return None
    has_prompt = any(row.get(k) for k in ("instruction", "prompt", "question"))
    has_answer = any(row.get(k) for k in ("output", "response", "answer"))
    if not has_prompt:
        return "missing an instruction/prompt/question field"
    if not has_answer:
        return "missing an output/response/answer field"
    return None


def validate(path: str | Path, kind: DatasetKind) -> tuple[ValidationReport, DatasetStats]:
    path = Path(path)
    report = ValidationReport()
    fmt = detect_format(path)
    stats = DatasetStats(fmt=fmt)

    if not path.exists():
        report.error(f"File not found: {path}")
        return report, stats
    stats.size_bytes = path.stat().st_size
    if stats.size_bytes == 0:
        report.error("File is empty.")
        return report, stats

    if fmt == "parquet":
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            report.error(
                "Parquet files need the 'pyarrow' package (pip install pyarrow) — "
                "or convert the dataset to JSONL/CSV."
            )
            return report, stats

    if kind in (DatasetKind.PRETRAIN_CORPUS, DatasetKind.DOMAIN_CORPUS):
        _validate_corpus(path, kind, report, stats)
    else:
        _validate_records(path, fmt, kind, report, stats)

    return report, stats


def _validate_corpus(path: Path, kind: DatasetKind, report: ValidationReport, stats: DatasetStats) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        report.error("Corpus contains only whitespace.")
        return
    stats.num_rows = text.count("\n") + 1
    stats.num_tokens_est = estimate_tokens(text)
    stats.sample_rows = [ln for ln in text.splitlines()[:20] if ln.strip()][:10]
    stats.token_histogram = [estimate_tokens(ln) for ln in text.splitlines()[:500] if ln.strip()]

    threshold = MIN_PRETRAIN_TOKENS if kind == DatasetKind.PRETRAIN_CORPUS else MIN_DOMAIN_TOKENS
    if stats.num_tokens_est < threshold:
        report.warn(
            f"~{stats.num_tokens_est:,} estimated tokens is below the ~{threshold:,} suggested for "
            f"{kind.value}. Training will run, but expect a toy-quality result."
        )
    report.info(f"~{stats.num_tokens_est:,} estimated tokens across {stats.num_rows:,} lines.")


def _iter_records(path: Path, fmt: str):
    if fmt == "jsonl":
        with path.open(encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield i, json.loads(line)
                except json.JSONDecodeError as e:
                    yield i, e
    elif fmt == "json":
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        rows = data if isinstance(data, list) else data.get("data", [])
        for i, row in enumerate(rows, start=1):
            yield i, row
    elif fmt == "csv":
        with path.open(encoding="utf-8", errors="replace", newline="") as f:
            for i, row in enumerate(csv.DictReader(f), start=1):
                yield i, dict(row)
    elif fmt == "parquet":
        import pyarrow.parquet as pq  # noqa: PLC0415 — optional dep, guarded in validate()

        for i, row in enumerate(pq.read_table(path).to_pylist(), start=1):
            yield i, row
    else:  # txt fallback: each line a record with a 'text' field
        with path.open(encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, start=1):
                if line.strip():
                    yield i, {"text": line.strip()}


def _validate_records(path: Path, fmt: str, kind: DatasetKind,
                       report: ValidationReport, stats: DatasetStats) -> None:
    seen: set[str] = set()
    duplicates = 0
    errors = 0
    for lineno, row in _iter_records(path, fmt):
        if isinstance(row, json.JSONDecodeError):
            report.error(f"Invalid JSON: {row.msg}", line=lineno)
            errors += 1
            if errors >= 20:
                report.error("Too many parse errors — stopping validation.")
                break
            continue
        if not isinstance(row, dict):
            report.error("Row is not an object.", line=lineno)
            errors += 1
            continue

        if kind == DatasetKind.INSTRUCTION or kind == DatasetKind.EVAL:
            err = _validate_instruction_row(row)
            if err and errors < 20:
                report.error(err, line=lineno)
                errors += 1

        text = _extract_text(row)
        toks = estimate_tokens(text)
        stats.num_rows += 1
        stats.num_tokens_est += toks
        if len(stats.token_histogram) < 500:
            stats.token_histogram.append(toks)
        if len(stats.sample_rows) < 10:
            stats.sample_rows.append(row)

        key = text.strip()[:200]
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)

    if stats.num_rows == 0:
        report.error("No usable rows found.")
        return
    if duplicates:
        report.warn(f"{duplicates} duplicate/near-duplicate rows detected.")
    if kind == DatasetKind.INSTRUCTION:
        if stats.num_rows < MIN_INSTRUCTION_ROWS:
            report.warn(
                f"Only {stats.num_rows} examples. A QLoRA instruction run usually wants at least "
                f"~{MIN_INSTRUCTION_ROWS} (ideally {GOOD_INSTRUCTION_ROWS}+) for a meaningful result."
            )
        elif stats.num_rows < GOOD_INSTRUCTION_ROWS:
            report.info(f"{stats.num_rows} examples — workable; {GOOD_INSTRUCTION_ROWS}+ is better.")
    report.info(f"{stats.num_rows:,} rows, ~{stats.num_tokens_est:,} estimated tokens.")
