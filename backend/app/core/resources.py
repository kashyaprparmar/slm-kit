"""Atomic GPU leases shared by every managed workload (no ML imports)."""

from __future__ import annotations

import threading
import time
import uuid


class ResourceBusy(RuntimeError):
    pass


class GPUResources:
    def __init__(self):
        self._lock = threading.Lock()
        self._owner: dict | None = None

    def acquire(self, kind: str, ident: str | int) -> str:
        with self._lock:
            if self._owner:
                raise ResourceBusy(
                    f"GPU is reserved for {self._owner['kind']} ({self._owner['id']}). "
                    "Wait for it to finish or stop it before starting this operation."
                )
            token = uuid.uuid4().hex
            self._owner = {"token": token, "kind": kind, "id": ident, "since": time.time()}
            return token

    def release(self, token: str | None):
        with self._lock:
            if self._owner and self._owner['token'] == token:
                self._owner = None

    def update(self, token: str, kind: str, ident: str | int):
        with self._lock:
            if self._owner and self._owner['token'] == token:
                self._owner.update(kind=kind, id=ident)

    def snapshot(self) -> dict:
        with self._lock:
            return {k: v for k, v in self._owner.items() if k != 'token'} if self._owner else {"kind": "idle"}


gpu = GPUResources()
