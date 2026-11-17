"""Runtime checks for why llama.cpp GPU offload may be unavailable."""

from __future__ import annotations

import os
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
