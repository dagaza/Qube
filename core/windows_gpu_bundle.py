"""Windows PyInstaller GPU bundle requirements shared by packaging and CI."""

from __future__ import annotations

from pathlib import Path

from core.cuda_wheel_bundle import (
    LLAMA_LIB_CANDIDATES_WINDOWS,
    REQUIRED_CUDA_WHEEL_LIBS_WINDOWS,
)

LLAMA_LIB_CANDIDATES: tuple[str, ...] = LLAMA_LIB_CANDIDATES_WINDOWS
REQUIRED_CUDA_WHEEL_LIBS: tuple[str, ...] = REQUIRED_CUDA_WHEEL_LIBS_WINDOWS

# Copied next to llama_cpp libs during Windows Vulkan builds.
REQUIRED_VULKAN_LOADER_LIBS: tuple[str, ...] = ("vulkan-1.dll",)

# GPU backend DLLs produced by llama-cpp-python source/wheel builds.
VULKAN_GGML_LIB_NAMES: tuple[str, ...] = ("ggml-vulkan.dll",)
CUDA_GGML_LIB_NAMES: tuple[str, ...] = ("ggml-cuda.dll",)
SHARED_GGML_LIB_NAMES: tuple[str, ...] = ("ggml.dll", "ggml-base.dll")


def llama_lib_dir(dist_dir: Path) -> Path:
    return dist_dir / "_internal" / "llama_cpp" / "lib"


def _first_present(lib_dir: Path, names: tuple[str, ...]) -> str | None:
    for name in names:
        if (lib_dir / name).is_file():
            return name
    return None


def missing_llama_lib(dist_dir: Path) -> str | None:
    lib_dir = llama_lib_dir(dist_dir)
    if _first_present(lib_dir, LLAMA_LIB_CANDIDATES):
        return None
    return f"none of {LLAMA_LIB_CANDIDATES} found under {lib_dir}"


def missing_cuda_wheel_libs(dist_dir: Path) -> list[str]:
    lib_dir = llama_lib_dir(dist_dir)
    return [name for name in REQUIRED_CUDA_WHEEL_LIBS if not (lib_dir / name).is_file()]


def missing_vulkan_loader_libs(dist_dir: Path) -> list[str]:
    lib_dir = llama_lib_dir(dist_dir)
    return [name for name in REQUIRED_VULKAN_LOADER_LIBS if not (lib_dir / name).is_file()]


def missing_vulkan_ggml_lib(dist_dir: Path) -> str | None:
    lib_dir = llama_lib_dir(dist_dir)
    if _first_present(lib_dir, VULKAN_GGML_LIB_NAMES):
        return None
    return f"none of {VULKAN_GGML_LIB_NAMES} found under {lib_dir}"


def missing_cuda_ggml_lib(dist_dir: Path) -> str | None:
    lib_dir = llama_lib_dir(dist_dir)
    if _first_present(lib_dir, CUDA_GGML_LIB_NAMES):
        return None
    return f"none of {CUDA_GGML_LIB_NAMES} found under {lib_dir}"
