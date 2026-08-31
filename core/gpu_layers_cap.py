"""
Heuristic caps for llama.cpp `n_gpu_layers` based on detected GPU memory.

Layer count is model-dependent; VRAM is the practical limiter. We use a conservative
MB-per-layer estimate so users are less likely to OOM when raising the slider.

AMD APUs (e.g. Phoenix / 7840HS) and Apple Silicon expose a small dedicated VRAM
carve-out in sysfs while the GPU can use a much larger unified memory pool. Those
platforms use a RAM-based proxy budget (same headroom fraction as Apple).
"""
from __future__ import annotations

import glob
import logging
import platform
import sys
from typing import Literal, Optional

logger = logging.getLogger("Qube.GPULayersCap")

# llama.cpp rarely needs >200; keep aligned with app_settings historical max
_ABS_CEILING = 200
# Reserve for OS, display, and other allocations (MB)
_VRAM_OVERHEAD_MB = 768.0
# Pessimistic effective MB per transformer layer for mixed offload (varies by model/quant)
_MB_PER_LAYER_ESTIMATE = 200.0
# When VRAM cannot be detected, allow a conservative range so CPU-only still works
_UNKNOWN_VRAM_MAX_LAYERS = 99
# Windows has no reliable VRAM probe yet (Vulkan/iGPU); cap manual slider when unknown.
_UNKNOWN_VRAM_WIN32_MAX_LAYERS = 32
# Fraction of the VRAM-derived layer cap to leave for OS / other GPU users (first-run default)
_HEADROOM_FRACTION = 0.25
# Unified-memory GPU budget as a fraction of system RAM (Apple Silicon + AMD APU)
_UNIFIED_RAM_FRACTION = 0.55
# Above this carve-out, treat Linux amdgpu mem_info_vram_total as discrete VRAM
_APU_CARVEOUT_MAX_BYTES = 6 * 1024 * 1024 * 1024
# Require a meaningful GTT pool before assuming an AMD APU unified-memory layout
_GTT_MIN_BYTES = 2 * 1024 * 1024 * 1024

GpuMemoryKind = Literal[
    "none",
    "nvidia",
    "amd_discrete",
    "amd_unified",
    "apple_unified",
]

_last_gpu_memory_kind: GpuMemoryKind = "none"
_vram_probe_done = False
_cached_vram_bytes = 0


def gpu_memory_kind() -> GpuMemoryKind:
    """Kind reported by the most recent ``detect_gpu_vram_bytes()`` call."""
    return _last_gpu_memory_kind


def _set_gpu_memory_kind(kind: GpuMemoryKind) -> None:
    global _last_gpu_memory_kind
    _last_gpu_memory_kind = kind


def _nvidia_vram_bytes() -> int:
    try:
        import pynvml

        try:
            pynvml.nvmlInit()
        except Exception as e:
            if "already" not in str(e).lower() and "initialized" not in str(e).lower():
                raise
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return int(pynvml.nvmlDeviceGetMemoryInfo(h).total)
    except Exception:
        return 0


def _read_linux_amdgpu_meminfo(filename: str) -> int:
    if not sys.platform.startswith("linux"):
        return 0
    best = 0
    for path in glob.glob(f"/sys/class/drm/card*/device/{filename}"):
        try:
            with open(path, encoding="utf-8") as f:
                raw = int(f.read().strip())
        except (OSError, ValueError):
            continue
        # Newer amdgpu reports bytes; very small values may be KiB on some kernels
        if raw > 64 * 1024 * 1024:
            best = max(best, raw)
        elif raw > 0:
            best = max(best, raw * 1024)
    return best


def _linux_amdgpu_carveout_bytes() -> int:
    return _read_linux_amdgpu_meminfo("mem_info_vram_total")


def _linux_amdgpu_gtt_bytes() -> int:
    return _read_linux_amdgpu_meminfo("mem_info_gtt_total")


def _is_amd_apu_unified_memory(*, carveout_bytes: int, gtt_bytes: int) -> bool:
    if carveout_bytes <= 0:
        return False
    if carveout_bytes > _APU_CARVEOUT_MAX_BYTES:
        return False
    return gtt_bytes >= _GTT_MIN_BYTES


def _unified_memory_proxy_bytes() -> int:
    try:
        import psutil

        total = int(psutil.virtual_memory().total)
        return int(total * _UNIFIED_RAM_FRACTION)
    except Exception as e:
        logger.debug("Unified memory proxy failed: %s", e)
        return 0


def _apple_unified_memory_proxy_bytes() -> int:
    """Apple Silicon uses unified memory; approximate a GPU budget for Metal offload."""
    if sys.platform != "darwin":
        return 0
    if platform.machine().lower() not in ("arm64", "aarch64"):
        return 0
    return _unified_memory_proxy_bytes()


def _probe_gpu_vram_bytes() -> int:
    n = _nvidia_vram_bytes()
    if n > 0:
        _set_gpu_memory_kind("nvidia")
        return n

    if sys.platform.startswith("linux"):
        carveout = _linux_amdgpu_carveout_bytes()
        gtt = _linux_amdgpu_gtt_bytes()
        if carveout > 0:
            if _is_amd_apu_unified_memory(carveout_bytes=carveout, gtt_bytes=gtt):
                proxy = _unified_memory_proxy_bytes()
                if proxy > 0:
                    _set_gpu_memory_kind("amd_unified")
                    logger.debug(
                        "AMD APU unified memory: carveout=%.1f GB gtt=%.1f GB proxy=%.1f GB",
                        carveout / (1024.0**3),
                        gtt / (1024.0**3),
                        proxy / (1024.0**3),
                    )
                    return proxy
            _set_gpu_memory_kind("amd_discrete")
            return carveout

    n = _apple_unified_memory_proxy_bytes()
    if n > 0:
        _set_gpu_memory_kind("apple_unified")
        return n

    _set_gpu_memory_kind("none")
    return 0


def detect_gpu_vram_bytes() -> int:
    """
    Best-effort total GPU-accessible memory in bytes for layer-cap heuristics.

    Returns 0 if unknown (no driver, VM without GPU passthrough, etc.).
    Also updates :func:`gpu_memory_kind` for hardware profile / telemetry.

    Result is cached for the process lifetime (hardware does not change at runtime).
    """
    global _vram_probe_done, _cached_vram_bytes
    if _vram_probe_done:
        return _cached_vram_bytes
    _cached_vram_bytes = _probe_gpu_vram_bytes()
    _vram_probe_done = True
    return _cached_vram_bytes


def reset_gpu_vram_cache_for_tests() -> None:
    """Clear cached VRAM probe (unit tests only)."""
    global _vram_probe_done, _cached_vram_bytes
    _vram_probe_done = False
    _cached_vram_bytes = 0
    _set_gpu_memory_kind("none")


def is_unified_gpu_memory() -> bool:
    """True when the active GPU memory budget uses a unified RAM pool."""
    detect_gpu_vram_bytes()
    return gpu_memory_kind() in ("amd_unified", "apple_unified")


def max_safe_n_gpu_layers(vram_bytes: Optional[int] = None) -> int:
    """
    Upper bound for the internal engine GPU layer slider and persisted setting.

    Uses available VRAM (or conservative default when unknown). Always in [0, 200].
    """
    if vram_bytes is None:
        vram_bytes = detect_gpu_vram_bytes()

    if vram_bytes <= 0:
        if sys.platform == "win32":
            return min(_ABS_CEILING, _UNKNOWN_VRAM_WIN32_MAX_LAYERS)
        return min(_ABS_CEILING, _UNKNOWN_VRAM_MAX_LAYERS)

    mb = float(vram_bytes) / (1024.0 * 1024.0)
    usable = max(0.0, mb - _VRAM_OVERHEAD_MB)
    est = int(usable / _MB_PER_LAYER_ESTIMATE)
    return max(0, min(_ABS_CEILING, est))


def default_internal_n_gpu_layers_suggested() -> int:
    """
    First-run default: 75% of the detected safe maximum layer count (0 if no GPU layers advised).

    On Windows without VRAM telemetry, default to CPU-only (0) so Vulkan/iGPU installs do not
    start with aggressive partial offload that fails llama_context creation.
    """
    if detect_gpu_vram_bytes() <= 0 and sys.platform == "win32":
        return 0
    cap = max_safe_n_gpu_layers()
    if cap <= 0:
        return 0
    return max(0, min(cap, int(round(cap * (1.0 - _HEADROOM_FRACTION)))))
