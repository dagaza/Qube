#!/usr/bin/env python3
"""Copy NVIDIA CUDA runtime shared libraries next to libllama in a Linux bundle."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def stage(dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    try:
        import nvidia.cuda_runtime as cuda_runtime
    except ImportError as exc:
        print(f"ERROR: nvidia.cuda_runtime is not installed: {exc}", file=sys.stderr)
        return 1

    lib_dir = Path(cuda_runtime.__file__).resolve().parent / "lib"
    if not lib_dir.is_dir():
        print(f"ERROR: CUDA runtime lib dir missing: {lib_dir}", file=sys.stderr)
        return 1

    for src in sorted(lib_dir.iterdir()):
        if src.is_file() and ".so" in src.name:
            target = dest / src.name
            shutil.copy2(src, target)
            print(f"Copied {src.name} -> {target}")
            copied += 1

    if copied == 0:
        print(f"ERROR: no CUDA runtime libraries found under {lib_dir}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("Usage: stage_cuda_runtime_libs.py <dest-dir>", file=sys.stderr)
        return 2
    return stage(Path(args[0]))


if __name__ == "__main__":
    raise SystemExit(main())
