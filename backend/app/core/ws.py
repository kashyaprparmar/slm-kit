"""Tiny WebSocket broadcast hub.

Two channels: ``system`` (hardware telemetry) and ``run:{id}`` (per-run training
events + logs). Clients subscribe to a topic; publishers push JSON-able dicts.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict

from fastapi import WebSocket


class WSHub:
    def __init__(self) -> None:
        self._topics: dict[str, set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def connect(self, topic: str, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._topics[topic].add(ws)

    async def disconnect(self, topic: str, ws: WebSocket) -> None:
        async with self._lock:
            self._topics[topic].discard(ws)

    async def publish(self, topic: str, message: dict) -> None:
        async with self._lock:
            targets = list(self._topics.get(topic, ()))
        dead: list[WebSocket] = []
        for ws in targets:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._topics[topic].discard(ws)


hub = WSHub()
