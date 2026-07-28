"""Windows CUDA PyInstaller bundle layout checks shared by packaging and CI."""

from __future__ import annotations

from pathlib import Path

from core.cuda_wheel_bundle import (
    GITHUB_RELEASE_ASSET_LIMIT_BYTES,
    LLAMA_LIB_CANDIDATES_WINDOWS,
    REQUIRED_CUDA_WHEEL_LIBS_WINDOWS,
)


def cuda_lib_dir(dist_dir: Path) -> Path:
    return dist_dir / "_internal" / "llama_cpp" / "lib"


def missing_cuda_wheel_libs(dist_dir: Path) -> list[str]:
    lib_dir = cuda_lib_dir(dist_dir)
    return [name for name in REQUIRED_CUDA_WHEEL_LIBS_WINDOWS if not (lib_dir / name).is_file()]


def missing_llama_lib(dist_dir: Path) -> str | None:
    lib_dir = cuda_lib_dir(dist_dir)
    for name in LLAMA_LIB_CANDIDATES_WINDOWS:
        if (lib_dir / name).is_file():
            return None
    return f"none of {LLAMA_LIB_CANDIDATES_WINDOWS} found under {lib_dir}"
