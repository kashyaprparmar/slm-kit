"""Live hardware telemetry via pynvml + psutil, streamed over WebSocket.

Polls faster while a GPU job is running (see ``set_active``) and slower when
idle. All GPU calls degrade gracefully on a machine with no NVIDIA GPU (e.g. a
dev laptop) so the API still boots and reports CPU/RAM/disk.
"""

from __future__ import annotations

import asyncio
import shutil

from app.config import get_settings
from app.core.ws import hub
from app.domain import HardwareProfile

_settings = get_settings()

_nvml_ready = False


def _init_nvml() -> bool:
    global _nvml_ready
    if _nvml_ready:
        return True
    try:
        import pynvml

        pynvml.nvmlInit()
        _nvml_ready = True
    except Exception:
        _nvml_ready = False
    return _nvml_ready


def read_hardware() -> HardwareProfile:
    hw = HardwareProfile(source="pynvml")

    # CPU / RAM
    try:
        import psutil

        vm = psutil.virtual_memory()
        hw.ram_total_mb = vm.total // (1024 * 1024)
        hw.ram_free_mb = vm.available // (1024 * 1024)
        hw.cpu_count = psutil.cpu_count(logical=True)
        hw.cpu_util_pct = psutil.cpu_percent(interval=None)
    except Exception:
        pass

    # Disk (where models/datasets live)
    try:
        usage = shutil.disk_usage(str(_settings.home))
        hw.disk_free_mb = usage.free // (1024 * 1024)
    except Exception:
        pass

    # GPU
    if _init_nvml():
        try:
            import pynvml

            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            hw.gpu_name = name.decode() if isinstance(name, bytes) else name
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            hw.vram_total_mb = mem.total // (1024 * 1024)
            hw.vram_free_mb = mem.free // (1024 * 1024)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            hw.gpu_util_pct = float(util.gpu)
        except Exception:
            hw.source = "fallback"
    else:
        hw.source = "fallback"
        # Assume the target box's budget so fit math still works headless.
        hw.vram_total_mb = hw.vram_total_mb or _settings.vram_budget_mb

    return hw


class HardwarePoller:
    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._active = False
        self._latest = HardwareProfile()

    @property
    def latest(self) -> HardwareProfile:
        return self._latest

    def set_active(self, active: bool) -> None:
        self._active = active

    async def _loop(self) -> None:
        while True:
            self._latest = await asyncio.to_thread(read_hardware)
            await hub.publish("system", self._latest.model_dump())
            delay = _settings.active_poll_seconds if self._active else _settings.idle_poll_seconds
            await asyncio.sleep(delay)

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            self._task = None


poller = HardwarePoller()
