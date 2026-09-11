"""Shared JSONL log-file persistence for training/eval/generate/quantize jobs.

Every job type in this app streams live log lines over a WebSocket, but without
a durable copy those lines vanish the instant you navigate away or the job
finishes — reopening a past run showed an empty "Waiting for output…" panel.
This gives every job type one consistent way to persist + re-read its log
lines, independent of the live WS stream, so the Logs panel is comprehensive
both while something is running and after it's done.
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path


def append_log_line(path: Path, level: str, message: str, ts: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts": ts if ts is not None else time.time(), "level": level, "message": message}
    from app.core.observability import activity

    source = "training" if path.parent.name.isdigit() else "worker"
    activity.add(source, message, level, task_id=path.parent.name)
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass  # logging must never break the job it's logging


def read_log_tail(path: Path, lines: int = 2000) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open(encoding="utf-8", errors="replace") as stream:
        tail = deque(stream, maxlen=max(1, min(lines, 5000)))
    for raw in tail:
        raw = raw.strip()
        if not raw:
            continue
        try:
            out.append(json.loads(raw))
        except json.JSONDecodeError:
            out.append({"ts": None, "level": "info", "message": raw})
    return out[-max(1, lines):]
