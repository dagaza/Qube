"""Pin the system libdrm before Python wheels can load their own bundled copies.

ROCm builds of torch ship ``libdrm.so.2`` / ``libdrm_amdgpu.so.1`` under
``torch/lib``. When torch is imported first, those copies own the sonames for the
rest of the process, and Mesa's RADV Vulkan driver (loaded lazily by llama.cpp)
binds to the older libdrm, fails ``amdgpu_query_sw_info``, and exposes no Vulkan
device, so ``llama_supports_gpu_offload()`` returns False.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger("Qube.GPUDriverPreload")

_SYSTEM_LIB_DIRS = (
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib/aarch64-linux-gnu",
    "/usr/lib64",
    "/usr/lib",
    "/lib/x86_64-linux-gnu",
    "/lib64",
)
# Load order matters: libdrm_amdgpu depends on libdrm.
_PRELOAD_SONAMES = ("libdrm.so.2", "libdrm_amdgpu.so.1")

_preloaded: list[str] | None = None


def _system_lib_path(soname: str) -> Path | None:
    for directory in _SYSTEM_LIB_DIRS:
        candidate = Path(directory) / soname
        if candidate.is_file():
            return candidate
    return None


def preload_system_gpu_driver_libs() -> list[str]:
    """Load system libdrm sonames with RTLD_GLOBAL (Linux only, idempotent)."""
    global _preloaded
    if _preloaded is not None:
        return list(_preloaded)
    _preloaded = []
    if not sys.platform.startswith("linux"):
        return []

    import ctypes

    for soname in _PRELOAD_SONAMES:
        path = _system_lib_path(soname)
        if path is None:
            continue
        try:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            logger.debug("preload %s failed: %s", path, exc)
            continue
        _preloaded.append(str(path))
    return list(_preloaded)


def shadowing_libdrm_amdgpu_path() -> str | None:
    """Bundled libdrm_amdgpu path when it is mapped and no system copy is."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        with open("/proc/self/maps", encoding="utf-8") as maps:
            paths = {line.split()[-1] for line in maps if "libdrm_amdgpu" in line}
    except OSError:
        return None
    system_dirs = tuple(f"{d}/" for d in _SYSTEM_LIB_DIRS)
    if any(p.startswith(system_dirs) for p in paths):
        return None
    foreign = sorted(p for p in paths if p.startswith("/"))
    return foreign[0] if foreign else None


def reset_preload_state_for_tests() -> None:
    global _preloaded
    _preloaded = None
