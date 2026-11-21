"""Linux release smoke policy (GitHub Actions vs local decode)."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path


def _proc_cpu_flags() -> set[str]:
    flags_path = Path("/proc/cpuinfo")
    if not flags_path.is_file():
        return set()
    for line in flags_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lower().startswith("flags"):
            return set(line.split(":", 1)[-1].split())
    return set()


def linux_host_has_avx512_vnni() -> bool:
    """True when the host can execute our Zen4-targeted EVEX VNNI ggml builds."""
    return "avx512vnni" in _proc_cpu_flags()


def libcuda_driver_loadable() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    try:
        ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_LAZY)
        return True
    except OSError:
        return False


def bundled_decode_skip_reason(variant: str) -> str | None:
    """
    Return a message when GGUF decode must not run (CI hardware limits).

    ``QUBE_SMOKE_FORCE_DECODE=1`` overrides skips for manual debugging.
    """
    if os.environ.get("QUBE_SMOKE_FORCE_DECODE") == "1":
        return None
    key = (variant or "cpu").strip().lower()
    on_actions = os.environ.get("GITHUB_ACTIONS") == "true"
    if key == "cuda":
        if on_actions or not libcuda_driver_loadable():
            return "CUDA decode needs libcuda.so.1 (NVIDIA driver); skipped on driverless CI"
        return None
    if not on_actions:
        return None
    if key in ("cpu", "vulkan"):
        if not linux_host_has_avx512_vnni():
            return (
                "bundled libggml uses AVX512-VNNI (Zen 4 class); "
                "runner lacks avx512vnni in /proc/cpuinfo"
            )
    return None


def vulkan_gpu_layers_skip_reason() -> str | None:
    """Vulkan GPU offload is not available on GitHub-hosted runners."""
    if os.environ.get("QUBE_SMOKE_FORCE_DECODE") == "1":
        return None
    if os.environ.get("GITHUB_ACTIONS") == "true":
        return "Vulkan n_gpu_layers>0 smoke skipped (no GPU on GitHub-hosted runners)"
    return None