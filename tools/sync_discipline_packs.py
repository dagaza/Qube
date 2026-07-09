#!/usr/bin/env python3
"""Check discipline pack registry against live adapter readiness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.knowledge.discipline_pack_sync import sync_report, validate_discipline_packs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate scientific discipline packs vs adapter registry/readiness",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when validation errors exist (default)",
    )
    args = parser.parse_args()

    report = sync_report()
    print(report)
    errors = validate_discipline_packs()
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
