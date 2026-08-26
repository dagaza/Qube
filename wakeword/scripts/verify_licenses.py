#!/usr/bin/env python3
"""Fail-closed license gate for the Qube wake word pipeline.

Walks every ``*.license.json`` manifest under the datasets root and, with
``--require-commercial``, asserts that each asset is on the commercial allowlist
(see ``docs/licensing.md``). Exits non-zero on the first policy violation so it can
gate CI and ``train.py``.

Usage:
    python scripts/verify_licenses.py --datasets datasets --require-commercial
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import licenses  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify wake word dataset licenses.")
    parser.add_argument(
        "--datasets",
        default=str(Path(__file__).resolve().parent.parent / "datasets"),
        help="Path to the datasets/ root containing *.license.json manifests.",
    )
    parser.add_argument(
        "--require-commercial",
        action="store_true",
        help="Require every asset to be commercially licensed (production gate).",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Do not fail when no manifests are found (useful for a fresh scaffold).",
    )
    args = parser.parse_args(argv)

    datasets_root = Path(args.datasets)
    if not datasets_root.exists():
        print(f"ERROR: datasets root does not exist: {datasets_root}", file=sys.stderr)
        return 2

    result = licenses.run_gate(datasets_root, require_commercial=args.require_commercial)

    for warning in result.warnings:
        print(f"WARN: {warning}")

    if result.checked == 0:
        message = "No license manifests found (datasets not yet downloaded)."
        if args.allow_empty:
            print(f"OK: {message} (--allow-empty)")
            return 0
        print(f"ERROR: {message} Run download_datasets.py first.", file=sys.stderr)
        return 1

    if not result.ok:
        print(
            f"\nFAILED: {len(result.errors)} license violation(s) "
            f"across {result.checked} asset(s):",
            file=sys.stderr,
        )
        for error in result.errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    mode = "commercial" if args.require_commercial else "presence-only"
    print(f"OK: {result.checked} asset manifest(s) passed the {mode} license gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
