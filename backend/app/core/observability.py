"""Bounded local activity stream and request spans, without external services."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from contextvars import ContextVar

correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)


class Activity:
    def __init__(self):
        self._events: deque[dict] = deque(maxlen=2000)
        self._lock = threading.Lock()
        self._seq = 0
        self.session = uuid.uuid4().hex[:12]

    def add(self, source: str, message: str, level="INFO", **fields):
        with self._lock:
            self._seq += 1
            event = {"seq": self._seq, "id": f"{self.session}:{self._seq}", "ts": time.time(),
                     "source": source, "level": level.upper(), "message": message[:8000],
                     "correlation_id": correlation_id.get(), **fields}
            self._events.append(event)
            return event

    def since(self, after=0, limit=500):
        with self._lock:
            return [e for e in self._events if e["seq"] > after][-min(2000, max(1, limit)):]


activity = Activity()


class ActivityHandler(logging.Handler):
    def emit(self, record):
        activity.add(record.name.removeprefix("slmkit."), record.getMessage(), record.levelname)
