"""Locate shared libraries shipped by NVIDIA pip wheels (namespace packages)."""

from __future__ import annotations

import importlib
from pathlib import Path

CUDA_WHEEL_PACKAGES: tuple[str, ...] = (
    "nvidia.cuda_runtime",
    "nvidia.cublas",
)


def nvidia_package_lib_dir(import_name: str) -> Path:
    mod = importlib.import_module(import_name)
    if getattr(mod, "__path__", None):
        return Path(mod.__path__[0]) / "lib"
    if getattr(mod, "__file__", None):
        return Path(mod.__file__).resolve().parent / "lib"
    raise RuntimeError(f"Cannot locate lib directory for {import_name}")


def iter_nvidia_wheel_libs(*import_names: str) -> list[Path]:
    libs: list[Path] = []
    seen: set[str] = set()
    for import_name in import_names:
        lib_dir = nvidia_package_lib_dir(import_name)
        if not lib_dir.is_dir():
            raise RuntimeError(f"NVIDIA wheel lib dir missing: {lib_dir}")
        for path in sorted(lib_dir.iterdir()):
            if path.is_file() and ".so" in path.name and path.name not in seen:
                seen.add(path.name)
                libs.append(path)
    return libs
