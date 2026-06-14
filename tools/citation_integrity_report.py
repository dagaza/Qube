#!/usr/bin/env python3
"""Summarize citation_integrity events from ~/.qube/logs/llm_debug.log."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _default_log_path() -> Path:
    return Path.home() / ".qube" / "logs" / "llm_debug.log"


def main() -> int:
    parser = argparse.ArgumentParser(description="Citation integrity log summary")
    parser.add_argument(
        "--log",
        type=Path,
        default=_default_log_path(),
        help="Path to llm_debug.log",
    )
    args = parser.parse_args()
    path = args.log
    if not path.is_file():
        print(f"No log file at {path}", file=sys.stderr)
        return 1

    total = 0
    violations = 0
    missing = 0
    citation_issues = 0
    orphan_ids: Counter[str] = Counter()
    routes: Counter[str] = Counter()

    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if '"event": "citation_integrity"' not in line:
                continue
            try:
                payload = json.loads(line.split("|", 3)[-1].strip())
            except (json.JSONDecodeError, IndexError):
                continue
            if payload.get("event") != "citation_integrity":
                continue
            total += 1
            if payload.get("integrity_violation"):
                violations += 1
            if payload.get("missing_citation_when_sources_present"):
                missing += 1
            if payload.get("citation_issue"):
                citation_issues += 1
            route = str(payload.get("execution_route") or "unknown")
            routes[route] += 1
            for oid in payload.get("citation_orphan_ids") or []:
                orphan_ids[str(oid)] += 1

    print(f"Turns logged: {total}")
    print(f"Orphan violations: {violations}")
    print(f"Missing citations (web sources present): {missing}")
    print(f"Any citation issue (orphan or missing): {citation_issues}")
    if total:
        print(f"Orphan violation rate: {violations / total:.2%}")
        print(f"Missing citation rate: {missing / total:.2%}")
        print(f"Citation issue rate: {citation_issues / total:.2%}")
    if routes:
        print("By route:")
        for route, count in routes.most_common():
            print(f"  {route}: {count}")
    if orphan_ids:
        print("Orphan id histogram:")
        for oid, count in orphan_ids.most_common():
            print(f"  [{oid}]: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
