#!/usr/bin/env python3
"""Copy the Vulkan loader next to llama_cpp libs in a Windows bundle."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.vulkan_runtime import find_vulkan_loader


def stage(dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    try:
        loader = find_vulkan_loader()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    target = dest / loader.name
    shutil.copy2(loader, target)
    print(f"Copied {loader.name} -> {target}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("Usage: stage_vulkan_runtime_libs.py <dest-dir>", file=sys.stderr)
        return 2
    return stage(Path(args[0]))


if __name__ == "__main__":
    raise SystemExit(main())
