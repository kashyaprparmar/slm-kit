"""Small bounded executor gate for disk/network/tokenizer work."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from functools import partial


class CPUJobs:
    def __init__(self, concurrency: int = 2):
        self._limit = max(1, concurrency)
        self._semaphore = asyncio.Semaphore(self._limit)
        self._active = 0

    async def run(self, function, /, *args, **kwargs):
        async with self._semaphore:
            self._active += 1
            try:
                return await asyncio.to_thread(partial(function, *args, **kwargs))
            finally:
                self._active -= 1

    @asynccontextmanager
    async def slot(self):
        async with self._semaphore:
            self._active += 1
            try:
                yield
            finally:
                self._active -= 1

    def snapshot(self) -> dict:
        return {"active": self._active, "limit": self._limit}


cpu_jobs = CPUJobs()
