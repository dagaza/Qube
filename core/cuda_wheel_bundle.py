"""CUDA wheel libraries required in packaged GPU bundles (Linux and Windows)."""

from __future__ import annotations

import sys

# Linux: shared with scripts/release/verify_linux_cuda_bundle.sh
REQUIRED_CUDA_WHEEL_LIBS_LINUX: tuple[str, ...] = (
    "libcudart.so.12",
    "libcublas.so.12",
    "libcublasLt.so.12",
)

# Windows: NVIDIA pip wheels ship DLLs under nvidia/*/lib or bin
REQUIRED_CUDA_WHEEL_LIBS_WINDOWS: tuple[str, ...] = (
    "cudart64_12.dll",
    "cublas64_12.dll",
    "cublasLt64_12.dll",
)

LLAMA_LIB_CANDIDATES_LINUX: tuple[str, ...] = ("libllama.so", "libllama.so.0")
LLAMA_LIB_CANDIDATES_WINDOWS: tuple[str, ...] = ("llama.dll",)

# GitHub Releases rejects individual assets >= 2 GiB.
GITHUB_RELEASE_ASSET_LIMIT_BYTES: int = 2 * 1024 * 1024 * 1024


def required_cuda_wheel_lib_names() -> tuple[str, ...]:
    if sys.platform == "win32":
        return REQUIRED_CUDA_WHEEL_LIBS_WINDOWS
    return REQUIRED_CUDA_WHEEL_LIBS_LINUX


def llama_lib_candidates() -> tuple[str, ...]:
    if sys.platform == "win32":
        return LLAMA_LIB_CANDIDATES_WINDOWS
    return LLAMA_LIB_CANDIDATES_LINUX
