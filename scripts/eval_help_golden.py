#!/usr/bin/env python3
"""Run help golden retrieval eval and exit non-zero on Phase 6 misses."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.help_golden_eval import (  # noqa: E402
    assert_v1_targets,
    evaluate_golden_questions,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--locale",
        default="en",
        help="Help locale to evaluate (default: en)",
    )
    args = parser.parse_args()

    summary = evaluate_golden_questions(locale=args.locale)
    positive = summary.total - summary.negative_total
    print(
        f"Help golden eval ({summary.total} cases): "
        f"top-1 {summary.top1_hits}/{positive} ({summary.top1_rate:.1%}), "
        f"top-3 {summary.top3_hits}/{positive} ({summary.top3_rate:.1%}), "
        f"canonical {summary.canonical_hits}/{summary.canonical_total} "
        f"({summary.canonical_rate:.1%}), "
        f"settings paths {summary.settings_path_hits}/{summary.settings_path_total} "
        f"({summary.settings_path_rate:.1%}), "
        f"negative {summary.negative_hits}/{summary.negative_total} "
        f"({summary.negative_rate:.1%})"
    )
    try:
        assert_v1_targets(summary)
    except AssertionError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("All Phase 6 golden eval targets met.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
