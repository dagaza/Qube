"""Locate shared libraries shipped by NVIDIA pip wheels (namespace packages)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.linux_cuda_bundle import REQUIRED_CUDA_WHEEL_LIBS

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
    allowed = set(REQUIRED_CUDA_WHEEL_LIBS)
    for import_name in import_names:
        lib_dir = nvidia_package_lib_dir(import_name)
        if not lib_dir.is_dir():
            raise RuntimeError(f"NVIDIA wheel lib dir missing: {lib_dir}")
        for path in sorted(lib_dir.iterdir()):
            if not path.is_file() or ".so" not in path.name:
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
