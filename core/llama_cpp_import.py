"""Prepare DLL search paths and safely import llama-cpp-python."""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("Qube.LlamaCpp")

_Llama: Any | None = None
_attempted = False
_error: BaseException | None = None
_prepared = False


def llama_cpp_package_root() -> Path | None:
    """Resolve the llama-cpp-python install tree without importing ``llama_cpp``."""
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        meipass = getattr(sys, "_MEIPASS", None)
        candidates = [
            exe_dir / "_internal" / "llama_cpp",
            Path(meipass) / "llama_cpp" if meipass else None,
            Path(meipass) / "_internal" / "llama_cpp" if meipass else None,
        ]
        for path in candidates:
            if path is not None and path.is_dir():
                return path
        return None

    try:
        spec = importlib.util.find_spec("llama_cpp")
        if spec is None or not spec.origin:
            return None
        return Path(spec.origin).resolve().parent
    except Exception:
        return None


def llama_cpp_lib_dir() -> Path | None:
    if getattr(sys, "frozen", False):
        root = llama_cpp_package_root()
        if root is None:
            return None
        lib = root / "lib"
        return lib if lib.is_dir() else root

    root = llama_cpp_package_root()
    if root is None:
        return None
    lib = root / "lib"
    return lib if lib.is_dir() else root


def prepare_llama_cpp_runtime() -> None:
    global _prepared
    if _prepared:
        return
    _prepared = True

    lib_dir = llama_cpp_lib_dir()
    if lib_dir is None:
        return

    lib_prefix = str(lib_dir)
    if sys.platform == "win32":
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(lib_dir)
            except OSError as exc:
                logger.debug("add_dll_directory failed for %s: %s", lib_dir, exc)
        current = os.environ.get("PATH", "")
        if lib_prefix not in current.split(os.pathsep):
            os.environ["PATH"] = lib_prefix + os.pathsep + current
    elif sys.platform.startswith("linux"):
        from core.gpu_driver_preload import preload_system_gpu_driver_libs

        preload_system_gpu_driver_libs()
        current = os.environ.get("LD_LIBRARY_PATH", "")
        parts = [p for p in current.split(os.pathsep) if p]
        if lib_prefix not in parts:
            os.environ["LD_LIBRARY_PATH"] = (
                lib_prefix + (os.pathsep + current if current else "")
            )


def get_llama_class() -> Any | None:
    global _Llama, _attempted, _error
    if winget_validation_blocks_llama_import():
        logger.info("llama_cpp import skipped: WinGet validation mode active")
        return None
    if _attempted:
        return _Llama
    _attempted = True
    prepare_llama_cpp_runtime()
    try:
        from llama_cpp import Llama

        _Llama = Llama
    except Exception as exc:
        _error = exc
        logger.warning("llama_cpp import failed: %s", exc)
        _Llama = None
    return _Llama


def llama_import_was_attempted() -> bool:
    """True after the first ``get_llama_class()`` call in this process."""
    return _attempted


def llama_import_error() -> BaseException | None:
    get_llama_class()
    return _error


def reset_llama_import_state_for_tests() -> None:
    global _Llama, _attempted, _error, _prepared
    _Llama = None
    _attempted = False
    _error = None
    _prepared = False


def winget_validation_blocks_llama_import() -> bool:
    from core.winget_validation import is_winget_validation_mode

    return is_winget_validation_mode()
