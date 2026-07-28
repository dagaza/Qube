#!/usr/bin/env python3
"""Copy NVIDIA CUDA wheel shared libraries next to llama_cpp libs in a bundle."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.nvidia_wheel_lib_dirs import CUDA_WHEEL_PACKAGES, iter_nvidia_wheel_libs


def stage(dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    try:
        libs = iter_nvidia_wheel_libs(*CUDA_WHEEL_PACKAGES)
    except (ImportError, RuntimeError) as exc:
        print(f"ERROR: could not locate NVIDIA CUDA wheel libraries: {exc}", file=sys.stderr)
        return 1

    if not libs:
        print("ERROR: no NVIDIA CUDA wheel libraries found to stage", file=sys.stderr)
        return 1

    for src in libs:
        target = dest / src.name
        shutil.copy2(src, target)
        print(f"Copied {src.name} -> {target}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("Usage: stage_cuda_runtime_libs.py <dest-dir>", file=sys.stderr)
        return 2
    return stage(Path(args[0]))


if __name__ == "__main__":
    raise SystemExit(main())
