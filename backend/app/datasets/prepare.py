"""Stream input through canonical adapters into deterministic immutable splits."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path

from app.datasets.adapters import DatasetSchemaError, canonicalize
from app.datasets.validate import _iter_records, validate
from app.domain import DatasetKind


def _stable_key(seed: int, digest: str, sequence: int) -> str:
    return hashlib.sha256(f"{seed}:{digest}:{sequence}".encode()).hexdigest()


def prepare(source, destination, kind, columns, test_fraction, shuffle, deduplicate,
            drop_invalid, seed, validation_fraction=0.0):
    """Prepare using a disk spool rather than retaining the corpus in RAM."""
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if test_fraction + validation_fraction > 0.8:
        raise ValueError("Validation and test fractions must leave at least 20% for training.")
    token = uuid.uuid4().hex
    spool = destination / f".{token}.sqlite3"
    staged: list[Path] = []
    published: list[Path] = []
    dropped = 0
    corpus = kind in {DatasetKind.PRETRAIN_CORPUS, DatasetKind.DOMAIN_CORPUS}
    try:
        # sqlite3.Connection's context manager commits/rolls back but does not
        # close the handle. ``closing`` is required for reliable Windows cleanup.
        with closing(sqlite3.connect(spool)) as db:
            db.execute("PRAGMA journal_mode=OFF")
            db.execute("CREATE TABLE rows (sequence INTEGER PRIMARY KEY, digest TEXT, sort_key TEXT, payload TEXT)")
            if deduplicate:
                db.execute("CREATE UNIQUE INDEX uq_rows_digest ON rows(digest)")
            accepted = 0
            for index, row in _iter_records(source, source.suffix.lower().lstrip(".")):
                try:
                    record = canonicalize(row, columns) if isinstance(row, dict) else None
                    if record is None:
                        raise DatasetSchemaError("Row is not an object.")
                    payload = {"text": record.content_text()} if corpus else record.storage_row()
                    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    digest = hashlib.sha256(encoded.encode()).hexdigest()
                    cursor = db.execute(
                        "INSERT OR IGNORE INTO rows(sequence,digest,sort_key,payload) VALUES(?,?,?,?)",
                        (accepted, digest, _stable_key(seed, digest, accepted), encoded),
                    )
                    if cursor.rowcount == 0:
                        dropped += 1
                    else:
                        accepted += 1
                except (DatasetSchemaError, TypeError, ValueError) as exc:
                    if drop_invalid:
                        dropped += 1
                        continue
                    raise ValueError(f"Row {index} is invalid after column mapping: {exc}") from exc
            db.commit()
            count = int(db.execute("SELECT COUNT(*) FROM rows").fetchone()[0])
            n_test, n_validation = round(count * test_fraction), round(count * validation_fraction)
            requested = [("train", count - n_validation - n_test)]
            if validation_fraction:
                requested.append(("validation", n_validation))
            if test_fraction:
                requested.append(("test", n_test))
            if count < 1 or any(size < 1 for _, size in requested):
                raise ValueError("Not enough valid rows remain to create every requested split.")
            order = "sort_key, sequence" if shuffle else "sequence"
            cursor = db.execute(f"SELECT payload FROM rows ORDER BY {order}")  # noqa: S608
            suffix = "txt" if corpus else "jsonl"
            results = []
            try:
                for split, size in requested:
                    final = destination / f"prepared-{token}-{split}.{suffix}"
                    stage = final.with_suffix(final.suffix + ".part")
                    staged.append(stage)
                    with stage.open("x", encoding="utf-8", newline="\n") as stream:
                        for _ in range(size):
                            item = cursor.fetchone()
                            if item is None:
                                raise RuntimeError("Preparation spool ended before split output was complete.")
                            payload = json.loads(item[0])
                            stream.write((payload["text"] if corpus else json.dumps(payload, ensure_ascii=False)) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(stage, final)
                    published.append(final)
                    output_kind = kind if corpus or split == "train" else DatasetKind.EVAL
                    report, stats = validate(final, output_kind)
                    if not report.ok:
                        errors = "; ".join(i.message for i in report.issues if i.level == "error")
                        raise ValueError(f"Prepared {split} split failed validation: {errors}")
                    results.append((split, final, output_kind, report, stats))
            finally:
                # A live sqlite cursor retains a file handle on Windows even after
                # the connection context commits, preventing deterministic cleanup.
                cursor.close()
        return results, dropped
    except Exception:
        for path in [*staged, *published]:
            path.unlink(missing_ok=True)
        raise
    finally:
        spool.unlink(missing_ok=True)
