#!/usr/bin/env python3
"""Scan the repo for lazy main-stage access that eager-builds hidden pages.

Examples::

    python tools/audit_lazy_stage_footguns.py
    python tools/audit_lazy_stage_footguns.py --paths main.py core/composer_commands.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.lazy_main_stage_footguns import scan_lazy_stage_footguns  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detect getattr/hasattr/property access to lazy main-stage views",
    )
    parser.add_argument(
        "--paths",
        nargs="*",
        type=Path,
        help="Optional files or directories to scan (default: entire repo)",
    )
    args = parser.parse_args(argv)

    include: list[Path] | None = None
    if args.paths:
        include = []
        for raw in args.paths:
            path = raw if raw.is_absolute() else _ROOT / raw
            if path.is_dir():
                include.extend(sorted(path.rglob("*.py")))
            elif path.is_file():
                include.append(path)

    findings = scan_lazy_stage_footguns(_ROOT, include_paths=include)
    if not findings:
        print("No lazy-stage footguns found.")
        return 0

    print("Lazy-stage footguns (use _view peek or ensure_*_view() instead):\n")
    for item in findings:
        print(f"{item.path}:{item.line_no} [{item.kind} {item.view_name}]")
        print(f"  {item.line}")
    print(f"\nTotal: {len(findings)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
