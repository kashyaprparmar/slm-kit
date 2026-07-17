"""Training event stream schema.

A training subprocess emits newline-delimited JSON, one object per line, on
stdout. ``runner.py`` parses each line back into one of these events, persists
what matters (metrics, checkpoints, final status), and rebroadcasts it over the
WebSocket to the live-monitoring UI. Keeping this a small tagged union means any
backend (Unsloth, from-scratch, future Axolotl) speaks the same wire format.
"""

from __future__ import annotations

import json
import time
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, TypeAdapter


class LogEvent(BaseModel):
    type: Literal["log"] = "log"
    ts: float = Field(default_factory=time.time)
    level: str = "info"          # info | warning | error
    message: str


class MetricEvent(BaseModel):
    type: Literal["metric"] = "metric"
    ts: float = Field(default_factory=time.time)
    step: int
    total_steps: Optional[int] = None
    # Free-form so backends can report loss, lr, grad_norm, tokens_per_sec,
    # eta_seconds, epoch, etc. without schema changes.
    metrics: dict = Field(default_factory=dict)


class CheckpointEvent(BaseModel):
    type: Literal["checkpoint"] = "checkpoint"
    ts: float = Field(default_factory=time.time)
    step: int
    path: str
    is_final: bool = False


class SampleEvent(BaseModel):
    """A generated text sample (from-scratch pillar's sampling panel)."""

    type: Literal["sample"] = "sample"
    ts: float = Field(default_factory=time.time)
    step: int
    prompt: str = ""
    text: str


class StatusEvent(BaseModel):
    type: Literal["status"] = "status"
    ts: float = Field(default_factory=time.time)
    status: str                  # running | done | failed
    detail: Optional[str] = None


TrainingEvent = Union[LogEvent, MetricEvent, CheckpointEvent, SampleEvent, StatusEvent]

_ADAPTER: TypeAdapter[TrainingEvent] = TypeAdapter(TrainingEvent)


def dump_event(event: TrainingEvent) -> str:
    """Serialize an event to a single JSON line (no embedded newlines)."""
    return event.model_dump_json()


def parse_event(line: str) -> Optional[TrainingEvent]:
    """Parse one stdout line into an event, or None if it isn't one of ours.

    Subprocesses may print stray library logging to stdout; those lines are
    surfaced as generic LogEvents by the caller rather than failing.
    """
    line = line.strip()
    if not line or not line.startswith("{"):
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "type" not in obj:
        return None
    try:
        return _ADAPTER.validate_python(obj)
    except Exception:
        return None
