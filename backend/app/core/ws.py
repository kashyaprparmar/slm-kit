"""Tiny WebSocket broadcast hub.

Two channels: ``system`` (hardware telemetry) and ``run:{id}`` (per-run training
events + logs). Clients subscribe to a topic; publishers push JSON-able dicts.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import OrderedDict, defaultdict, deque

from fastapi import WebSocket


class WSHub:
    def __init__(self) -> None:
        self._topics: dict[str, set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()
        self._history = OrderedDict()

    async def connect(self, topic: str, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            history = list(self._history.get(topic, ()))
            if history:
                await asyncio.wait_for(ws.send_json({"type": "replay", "events": history}), 2)
            self._topics[topic].add(ws)

    async def disconnect(self, topic: str, ws: WebSocket) -> None:
        async with self._lock:
            self._topics[topic].discard(ws)

    async def publish(self, topic: str, message: dict) -> None:
        async with self._lock:
            message = {**message, "_event_id": uuid.uuid4().hex}
            if topic != "system":
                self._history.setdefault(topic, deque(maxlen=1000)).append(message)
                self._history.move_to_end(topic)
                while len(self._history) > 32:
                    self._history.popitem(last=False)
            targets = list(self._topics.get(topic, ()))
        dead: list[WebSocket] = []
        for ws in targets:
            try:
                await asyncio.wait_for(ws.send_json(message), timeout=2)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._topics[topic].discard(ws)


hub = WSHub()
