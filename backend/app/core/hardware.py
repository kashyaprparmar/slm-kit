"""Live hardware telemetry via pynvml + psutil, streamed over WebSocket.

Polls faster while a GPU job is running (see ``set_active``) and slower when
idle. All GPU calls degrade gracefully on a machine with no NVIDIA GPU (e.g. a
dev laptop) so the API still boots and reports CPU/RAM/disk.
"""

from __future__ import annotations

import asyncio
import platform
import shutil
import threading

from app.config import get_settings
from app.core.ws import hub
from app.domain import GPUDeviceProfile, HardwareProfile

_settings = get_settings()

_nvml_ready = False
# The poller runs in a worker thread while API requests may also call
# ``read_hardware``.  Serialize NVML access: the driver can otherwise return a
# transient error part way through a snapshot, which used to discard an
# otherwise valid GPU detection.
_nvml_lock = threading.RLock()


def _init_nvml() -> bool:
    global _nvml_ready
    with _nvml_lock:
        if _nvml_ready:
            return True
        try:
            import pynvml

            pynvml.nvmlInit()
            _nvml_ready = True
        except Exception:
            _nvml_ready = False
        return _nvml_ready


def _complete_cuda_fallback(hw: HardwareProfile) -> None:
    """Fill essential display fields when optional NVML metrics fail.

    PyTorch is deliberately not imported during healthy NVML polling.  This
    path only runs after an incomplete NVIDIA response, so CUDA-enabled images
    do not present a misleading "No GPU detected" state to the dashboard.
    """
    if hw.gpu_name:
        return
    try:
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            return
        index = 0
        properties = torch.cuda.get_device_properties(index)
        total_mb = int(properties.total_memory // (1024 * 1024))
        free_mb: int | None = None
        try:
            free, _total = torch.cuda.mem_get_info(index)
            free_mb = int(free // (1024 * 1024))
        except Exception:
            pass
        name = torch.cuda.get_device_name(index)
        hw.gpu_name = name
        hw.vram_total_mb = total_mb
        hw.vram_free_mb = free_mb
        hw.gpu_count = max(hw.gpu_count, torch.cuda.device_count())
        hw.cuda_available = True
        hw.cuda_runtime_version = hw.cuda_runtime_version or torch.version.cuda
        hw.gpus.append(GPUDeviceProfile(
            id=index,
            name=name,
            vram_total_mb=total_mb,
            vram_free_mb=free_mb if free_mb is not None else total_mb,
        ))
        hw.source = "cuda-fallback"
    except Exception:
        pass


def read_hardware() -> HardwareProfile:
    hw = HardwareProfile(source="pynvml", platform=platform.platform())

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
        hw.disk_total_mb = usage.total // (1024 * 1024)
    except Exception:
        pass

    # GPU
    if _init_nvml():
        try:
            # See ``_nvml_lock`` above.  Keep the whole snapshot atomic so an
            # endpoint request cannot race the background telemetry poller.
            with _nvml_lock:
                import pynvml

                count = int(pynvml.nvmlDeviceGetCount())
                hw.gpu_count = count
                hw.cuda_available = count > 0
                driver = pynvml.nvmlSystemGetDriverVersion()
                hw.nvidia_driver_version = driver.decode() if isinstance(driver, bytes) else str(driver)
                try:
                    raw_cuda = int(pynvml.nvmlSystemGetCudaDriverVersion_v2())
                    hw.cuda_runtime_version = f"{raw_cuda // 1000}.{(raw_cuda % 1000) // 10}"
                except Exception:
                    pass
                for index in range(count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                    raw_name = pynvml.nvmlDeviceGetName(handle)
                    name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
                    raw_uuid = pynvml.nvmlDeviceGetUUID(handle)
                    uuid = raw_uuid.decode() if isinstance(raw_uuid, bytes) else str(raw_uuid)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    major = minor = None
                    try:
                        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                    except Exception:
                        pass
                    try:
                        temperature = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
                    except Exception:
                        temperature = None
                    try:
                        power = float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000
                    except Exception:
                        power = None
                    capability = f"{major}.{minor}" if major is not None else None
                    hw.gpus.append(GPUDeviceProfile(
                        id=index, uuid=uuid, name=name,
                        vram_total_mb=mem.total // (1024 * 1024),
                        vram_free_mb=mem.free // (1024 * 1024),
                        utilization_pct=float(util.gpu),
                        compute_capability=capability,
                        temperature_c=temperature, power_watts=power,
                        bf16_supported=bool(major is not None and major >= 8),
                        fp8_supported=bool(major is not None and (major, minor or 0) >= (8, 9)),
                        flash_attention_feasible=bool(major is not None and major >= 8),
                    ))
                if hw.gpus:
                    primary = hw.gpus[0]
                    hw.gpu_name = primary.name
                    hw.vram_total_mb = primary.vram_total_mb
                    hw.vram_free_mb = primary.vram_free_mb
                    hw.gpu_util_pct = primary.utilization_pct
        except Exception:
            hw.source = "fallback"
    else:
        hw.source = "fallback"
        # Planning defaults belong in estimates, never in measured telemetry.

    _complete_cuda_fallback(hw)

    # Avoid importing torch in the control plane merely to probe runtimes.
    # MPS is advertised only as a host capability; worker diagnostics performs
    # the authoritative library-level probe.
    hw.mps_available = platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}

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
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None


poller = HardwarePoller()
