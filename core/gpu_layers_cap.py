"""
Heuristic caps for llama.cpp `n_gpu_layers` based on detected GPU memory.

Layer count is model-dependent; VRAM is the practical limiter. We use a conservative
MB-per-layer estimate so users are less likely to OOM when raising the slider.
"""
from __future__ import annotations

import glob
import logging
import platform
import sys
from typing import Optional

logger = logging.getLogger("Qube.GPULayersCap")

# llama.cpp rarely needs >200; keep aligned with app_settings historical max
_ABS_CEILING = 200
# Reserve for OS, display, and other allocations (MB)
_VRAM_OVERHEAD_MB = 768.0
# Pessimistic effective MB per transformer layer for mixed offload (varies by model/quant)
_MB_PER_LAYER_ESTIMATE = 200.0
# When VRAM cannot be detected, allow a conservative range so CPU-only still works
_UNKNOWN_VRAM_MAX_LAYERS = 99
# Fraction of the VRAM-derived layer cap to leave for OS / other GPU users (first-run default)
_HEADROOM_FRACTION = 0.25


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
    except Exception as e:
        logger.debug("NVIDIA VRAM probe failed: %s", e)
        return 0


def _linux_amdgpu_vram_bytes() -> int:
    if not sys.platform.startswith("linux"):
        return 0
    for path in glob.glob("/sys/class/drm/card*/device/mem_info_vram_total"):
        try:
            with open(path, encoding="utf-8") as f:
                raw = int(f.read().strip())
        except (OSError, ValueError):
            continue
        # Newer amdgpu reports bytes; very small values may be KiB on some kernels
        if raw > 64 * 1024 * 1024:
            return raw
        if raw > 0:
            return raw * 1024
    return 0


def _apple_unified_memory_proxy_bytes() -> int:
    """Apple Silicon uses unified memory; approximate a GPU budget for Metal offload."""
    if sys.platform != "darwin":
        return 0
    if platform.machine().lower() not in ("arm64", "aarch64"):
        return 0
    try:
        import psutil

        total = int(psutil.virtual_memory().total)
        return int(total * 0.55)
    except Exception as e:
        logger.debug("Apple unified memory proxy failed: %s", e)
        return 0


def detect_gpu_vram_bytes() -> int:
    """
    Best-effort total GPU-accessible memory in bytes.
    Returns 0 if unknown (no driver, VM without GPU passthrough, etc.).
    """
    n = _nvidia_vram_bytes()
    if n > 0:
        return n

    n = _linux_amdgpu_vram_bytes()
    if n > 0:
        return n

    n = _apple_unified_memory_proxy_bytes()
    if n > 0:
        return n

    return 0


def max_safe_n_gpu_layers(vram_bytes: Optional[int] = None) -> int:
    """
    Upper bound for the internal engine GPU layer slider and persisted setting.

    Uses available VRAM (or conservative default when unknown). Always in [0, 200].
    """
    if vram_bytes is None:
        vram_bytes = detect_gpu_vram_bytes()

    if vram_bytes <= 0:
        return min(_ABS_CEILING, _UNKNOWN_VRAM_MAX_LAYERS)

    mb = float(vram_bytes) / (1024.0 * 1024.0)
    usable = max(0.0, mb - _VRAM_OVERHEAD_MB)
    est = int(usable / _MB_PER_LAYER_ESTIMATE)
    return max(0, min(_ABS_CEILING, est))


def default_internal_n_gpu_layers_suggested() -> int:
    """
    First-run default: 75% of the detected safe maximum layer count (0 if no GPU layers advised).
    """
    cap = max_safe_n_gpu_layers()
    if cap <= 0:
        return 0
    return max(0, min(cap, int(round(cap * (1.0 - _HEADROOM_FRACTION)))))
