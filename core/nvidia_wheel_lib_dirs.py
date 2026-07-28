"""Locate shared libraries shipped by NVIDIA pip wheels (namespace packages)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from core.cuda_wheel_bundle import required_cuda_wheel_lib_names

CUDA_WHEEL_PACKAGES: tuple[str, ...] = (
    "nvidia.cuda_runtime",
    "nvidia.cublas",
)


def nvidia_package_lib_dir(import_name: str) -> Path:
    mod = importlib.import_module(import_name)
    if getattr(mod, "__path__", None):
        base = Path(mod.__path__[0])
        for sub in ("lib", "bin"):
            candidate = base / sub
            if candidate.is_dir():
                return candidate
        return base / "lib"
    if getattr(mod, "__file__", None):
        parent = Path(mod.__file__).resolve().parent
        for sub in ("lib", "bin"):
            candidate = parent / sub
            if candidate.is_dir():
                return candidate
    raise RuntimeError(f"Cannot locate lib directory for {import_name}")


def _is_cuda_lib_filename(name: str) -> bool:
    if sys.platform == "win32":
        return name.lower().endswith(".dll")
    return ".so" in name


def iter_nvidia_wheel_libs(*import_names: str) -> list[Path]:
    libs: list[Path] = []
    seen: set[str] = set()
    allowed = set(required_cuda_wheel_lib_names())
    for import_name in import_names:
        lib_dir = nvidia_package_lib_dir(import_name)
        if not lib_dir.is_dir():
            raise RuntimeError(f"NVIDIA wheel lib dir missing: {lib_dir}")
        for path in sorted(lib_dir.iterdir()):
            if not path.is_file():
                continue
            if not _is_cuda_lib_filename(path.name):
                continue
            if path.name not in allowed:
                continue
            if path.name not in seen:
                seen.add(path.name)
                libs.append(path)
    missing = allowed - seen
    if missing:
        raise RuntimeError(f"Required NVIDIA wheel libraries missing: {', '.join(sorted(missing))}")
    return libs
