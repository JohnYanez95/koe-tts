"""
GPU monitoring utilities for training.

Lightweight wrapper around pynvml for temperature/memory monitoring.
Used by ThermalWatchdog for emergency shutdown logic.
"""

from dataclasses import dataclass
from typing import Optional

# Lazy import to avoid hard dependency
_pynvml = None
_pynvml_initialized = False


def _init_pynvml() -> bool:
    """Initialize pynvml. Returns True if successful."""
    global _pynvml, _pynvml_initialized
    if _pynvml_initialized:
        return _pynvml is not None

    try:
        import pynvml
        pynvml.nvmlInit()
        _pynvml = pynvml
        _pynvml_initialized = True
        return True
    except Exception:
        _pynvml_initialized = True  # Mark as tried
        return False


@dataclass
class GpuStats:
    """GPU statistics snapshot."""
    temp_c: int
    util_pct: int
    mem_used_mb: int
    mem_total_mb: int
    power_w: Optional[int] = None

    @property
    def mem_used_pct(self) -> float:
        return (self.mem_used_mb / self.mem_total_mb * 100) if self.mem_total_mb > 0 else 0.0


def get_gpu_stats(device_idx: int = 0) -> Optional[GpuStats]:
    """
    Get GPU statistics.

    Args:
        device_idx: GPU index (default 0)

    Returns:
        GpuStats or None if unavailable
    """
    if not _init_pynvml():
        return None

    try:
        handle = _pynvml.nvmlDeviceGetHandleByIndex(device_idx)

        temp = _pynvml.nvmlDeviceGetTemperature(
            handle, _pynvml.NVML_TEMPERATURE_GPU
        )
        util = _pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = _pynvml.nvmlDeviceGetMemoryInfo(handle)

        try:
            power = _pynvml.nvmlDeviceGetPowerUsage(handle) // 1000  # mW to W
        except Exception:
            power = None

        return GpuStats(
            temp_c=temp,
            util_pct=util.gpu,
            mem_used_mb=mem.used // (1024 * 1024),
            mem_total_mb=mem.total // (1024 * 1024),
            power_w=power,
        )
    except Exception:
        return None


def get_gpu_temp(device_idx: int = 0) -> Optional[int]:
    """Get GPU temperature in Celsius. Returns None if unavailable."""
    stats = get_gpu_stats(device_idx)
    return stats.temp_c if stats else None
