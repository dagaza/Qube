#!/usr/bin/env python3
"""Export aggregated @help query telemetry from Qube.Help log files (§13.4)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.help_query_export import export_help_query_report, load_help_log_events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_files",
        nargs="*",
        help="Log files to scan (default: stdin)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON report",
    )
    args = parser.parse_args()

    if args.log_files:
        events = []
        for path_str in args.log_files:
            events.extend(load_help_log_events(Path(path_str)))
    else:
        from core.help_query_export import iter_help_log_events

        events = iter_help_log_events(sys.stdin)

    report = export_help_query_report(events)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    print(f"@help events: {report['total_events']} ({report['unique_queries']} unique queries)")
    print("\nTop queries:")
    for row in report["top_queries"][:10]:
        empty = row["empty_retrieval_count"]
        print(
            f"  {row['count']:3d}x  empty={empty:2d}  {row['query'][:120]}"
        )
    print("\nDoc backlog (priority):")
    for item in report["doc_backlog"][:10]:
        print(
            f"  score={item['priority_score']:.2f}  "
            f"{item['count']}x  {item['suggested_action']}  "
            f"{item['query'][:100]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
