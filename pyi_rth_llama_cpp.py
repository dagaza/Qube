"""PyInstaller runtime hook: register llama_cpp/lib before extension imports."""

from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        from core.llama_cpp_import import prepare_llama_cpp_runtime

        prepare_llama_cpp_runtime()
    except Exception:
        exe_dir = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
        for lib_dir in (
            exe_dir / "llama_cpp" / "lib",
            Path(sys.executable).resolve().parent / "_internal" / "llama_cpp" / "lib",
        ):
            if not lib_dir.is_dir():
                continue
            try:
                import os

                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(str(lib_dir))
                os.environ["PATH"] = str(lib_dir) + os.pathsep + os.environ.get("PATH", "")
            except Exception:
                pass
            break
