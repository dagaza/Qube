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


def _package_base_dir(import_name: str) -> Path:
    mod = importlib.import_module(import_name)
    if getattr(mod, "__path__", None):
        return Path(mod.__path__[0])
    if getattr(mod, "__file__", None):
        return Path(mod.__file__).resolve().parent
    raise RuntimeError(f"Cannot locate package directory for {import_name}")


def nvidia_package_lib_dirs(import_name: str) -> list[Path]:
    """Return lib/bin directories that may contain CUDA shared libraries."""
    base = _package_base_dir(import_name)
    # Windows wheels often ship DLLs under bin/ but also include lib/ (import libs only).
    subs = ("bin", "lib") if sys.platform == "win32" else ("lib", "bin")
    dirs = [base / sub for sub in subs if (base / sub).is_dir()]
    if dirs:
        return dirs
    raise RuntimeError(f"Cannot locate lib directory for {import_name}")


def nvidia_package_lib_dir(import_name: str) -> Path:
    return nvidia_package_lib_dirs(import_name)[0]


def _is_cuda_lib_filename(name: str) -> bool:
    if sys.platform == "win32":
        return name.lower().endswith(".dll")
    return ".so" in name


def iter_nvidia_wheel_libs(*import_names: str) -> list[Path]:
    libs: list[Path] = []
    seen: set[str] = set()
    allowed = set(required_cuda_wheel_lib_names())
    for import_name in import_names:
        lib_dirs = nvidia_package_lib_dirs(import_name)
        for lib_dir in lib_dirs:
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
