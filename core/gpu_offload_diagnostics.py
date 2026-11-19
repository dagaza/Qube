"""Runtime checks for why llama.cpp GPU offload may be unavailable."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from core.llama_cpp_import import llama_cpp_lib_dir


def llama_cpp_package_has_vulkan_libs() -> bool:
    """True when the installed llama-cpp-python wheel includes Vulkan ggml libs."""
    lib_dir = llama_cpp_lib_dir()
    if lib_dir is None:
        return False
    return any(lib_dir.glob("libggml-vulkan.so*")) or any(
        lib_dir.glob("ggml-vulkan.dll")
    )


def dri_render_node_accessible() -> bool:
    """True when at least one /dev/dri/renderD* node can be opened read/write."""
    dri = Path("/dev/dri")
    if not dri.is_dir():
        return False
    for node in sorted(dri.glob("renderD*")):
        try:
            fd = os.open(str(node), os.O_RDWR)
            os.close(fd)
            return True
        except OSError:
            continue
    return False


def log_startup_gpu_offload_probe(logger: logging.Logger) -> None:
    """One-shot boot log for Vulkan/llama offload (Linux Vulkan builds)."""
    if not sys.platform.startswith("linux"):
        return
    if not llama_cpp_package_has_vulkan_libs():
        return

    from core.gpu_driver_preload import (
        radv_icd_mapped,
        shadowing_libdrm_amdgpu_path,
        shadowing_libvulkan_loader_path,
    )
    from core.llama_cpp_import import get_llama_class, winget_validation_blocks_llama_import

    supports: bool | None = None
    backend: str | None = None
    if not winget_validation_blocks_llama_import() and get_llama_class() is not None:
        from core.inference_transparency import get_build_snapshot

        build = get_build_snapshot(refresh=True)
        supports = build.get("supports_gpu_offload")
        backend = build.get("backend_hint")

    logger.info(
        "[LlamaCpp] startup probe supports_gpu_offload=%s dri=%s vulkan_libs=%s "
        "backend_hint=%s libvulkan_shadow=%s libdrm_shadow=%s radv_icd=%s",
        supports,
        dri_render_node_accessible(),
        True,
        backend,
        shadowing_libvulkan_loader_path(),
        shadowing_libdrm_amdgpu_path(),
        radv_icd_mapped(),
    )
