#!/usr/bin/env python3
"""
View logs/routing_debug.log (Qube.RoutingDebug) without terminal spam.

Examples:
  python tools/view_routing_logs.py --last 200
  python tools/view_routing_logs.py --follow
  python tools/view_routing_logs.py --filter HYBRID --last 500
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.routing_debug_sink import default_routing_debug_log_path  # noqa: E402


def _read_tail_lines(path: Path, n: int) -> list[str]:
    if not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"Error reading {path}: {e}", file=sys.stderr)
        return []
    lines = text.splitlines()
    if n <= 0 or len(lines) <= n:
        return lines
    return lines[-n:]


def _filter_lines(lines: list[str], needle: str | None) -> list[str]:
    if not needle:
        return lines
    n = needle.lower()
    return [ln for ln in lines if n in ln.lower()]


def _print_lines(lines: list[str]) -> None:
    for ln in lines:
        print(ln)


def main() -> int:
    ap = argparse.ArgumentParser(description="View Qube routing debug log file.")
    ap.add_argument(
        "--last",
        type=int,
        default=200,
        metavar="N",
        help="Show last N lines (default 200). Use 0 for all lines.",
    )
    ap.add_argument(
        "-f",
        "--follow",
        action="store_true",
        help="Keep reading new lines (like tail -f). Ctrl+C to stop.",
    )
    ap.add_argument(
        "--filter",
        type=str,
        default=None,
        metavar="SUBSTR",
        help="Only lines containing substring (case-insensitive).",
    )
    ap.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Override log file path (default: logs/routing_debug.log under repo root).",
    )
    args = ap.parse_args()

    path = args.path or default_routing_debug_log_path()
    if not path.is_file() and not args.follow:
        print(f"No log file yet: {path}", file=sys.stderr)
        return 1

    last_n = max(0, int(args.last))

    if not args.follow:
        lines = _read_tail_lines(path, last_n)
        lines = _filter_lines(lines, args.filter)
        _print_lines(lines)
        return 0

    pos = 0
    if path.is_file():
        lines = _read_tail_lines(path, last_n)
        lines = _filter_lines(lines, args.filter)
        _print_lines(lines)
        try:
            pos = path.stat().st_size
        except OSError:
            pos = 0
    else:
        pos = 0

    try:
        while True:
            time.sleep(0.5)
            if not path.is_file():
                continue
            try:
                st = path.stat()
            except OSError:
                continue
            if st.st_size < pos:
                pos = 0
            if st.st_size <= pos:
                continue
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
            for raw in chunk.splitlines():
                if args.filter and args.filter.lower() not in raw.lower():
                    continue
                print(raw)
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("", file=sys.stderr)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
