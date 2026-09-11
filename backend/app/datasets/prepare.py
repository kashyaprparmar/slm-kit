"""Create new prepared datasets while keeping the original file intact."""
import hashlib
import json
import random
import uuid
from pathlib import Path

from app.datasets.validate import (
    _extract_text,
    _iter_records,
    _validate_instruction_row,
    normalize_row,
    validate,
)
from app.domain import DatasetKind


def prepare(source, destination, kind, columns, test_fraction, shuffle, deduplicate, drop_invalid, seed):
    rows = []
    seen = set()
    dropped = 0
    corpus = kind in {DatasetKind.PRETRAIN_CORPUS, DatasetKind.DOMAIN_CORPUS}
    for index, row in _iter_records(Path(source), Path(source).suffix.lstrip(".")):
        if index > 100_000:
            raise ValueError("Preparation supports up to 100,000 rows per file. Split a larger source first.")
        if isinstance(row, dict):
            row = normalize_row(row, columns)
            if corpus:
                row = {"text": _extract_text(row)}
            valid = bool(row.get("text", "").strip()) if corpus else _validate_instruction_row(row) is None
        else:
            valid = False
        if not valid:
            if drop_invalid:
                dropped += 1
                continue
            raise ValueError(f"Row {index} is invalid after column mapping. Correct the mapping or enable Skip invalid rows.")
        fingerprint = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        if deduplicate and fingerprint in seen:
            dropped += 1
            continue
        seen.add(fingerprint)
        rows.append(row)
    if len(rows) < (2 if test_fraction else 1):
        raise ValueError("Not enough valid rows remain to create the requested split.")
    if shuffle:
        random.Random(seed).shuffle(rows)
    n_test = max(1, round(len(rows) * test_fraction)) if test_fraction else 0
    splits = [("train", rows[n_test:]), ("test", rows[:n_test])] if n_test else [("train", rows)]
    paths = []
    try:
        for name, items in splits:
            path = Path(destination) / f"prepared-{uuid.uuid4().hex}-{name}.{'txt' if corpus else 'jsonl'}"
            paths.append(path)
            with path.open("x", encoding="utf-8") as stream:
                for row in items:
                    stream.write((row["text"] if corpus else json.dumps(row, ensure_ascii=False)) + "\n")
        results = []
        for (name, _), path in zip(splits, paths, strict=True):
            output_kind = kind if corpus or name == "train" else DatasetKind.EVAL
            report, stats = validate(path, output_kind)
            results.append((name, path, output_kind, report, stats))
        return results, dropped
    except Exception:
        for path in paths:
            path.unlink(missing_ok=True)
        raise
