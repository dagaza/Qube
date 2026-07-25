"""Linux CUDA PyInstaller bundle requirements shared by packaging and CI."""

from __future__ import annotations

from pathlib import Path

# Shared libraries llama.cpp CUDA wheels need at runtime (driver libcuda comes from
# the end-user NVIDIA driver). Keep this list in sync with
# scripts/release/verify_linux_cuda_bundle.sh.
REQUIRED_CUDA_WHEEL_LIBS: tuple[str, ...] = (
    "libcudart.so.12",
    "libcublas.so.12",
    "libcublasLt.so.12",
)

LLAMA_LIB_CANDIDATES: tuple[str, ...] = ("libllama.so", "libllama.so.0")

# GitHub Releases rejects individual assets >= 2 GiB.
GITHUB_RELEASE_ASSET_LIMIT_BYTES: int = 2 * 1024 * 1024 * 1024


def cuda_lib_dir(dist_dir: Path) -> Path:
    return dist_dir / "_internal" / "llama_cpp" / "lib"


def missing_cuda_wheel_libs(dist_dir: Path) -> list[str]:
    lib_dir = cuda_lib_dir(dist_dir)
    return [name for name in REQUIRED_CUDA_WHEEL_LIBS if not (lib_dir / name).is_file()]


def missing_llama_lib(dist_dir: Path) -> str | None:
    lib_dir = cuda_lib_dir(dist_dir)
    for name in LLAMA_LIB_CANDIDATES:
        if (lib_dir / name).is_file():
            return None
    return f"none of {LLAMA_LIB_CANDIDATES} found under {lib_dir}"
