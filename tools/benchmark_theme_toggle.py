#!/usr/bin/env python3
"""Parse ThemeProfile lines from qube.log and check regression guardrails.

Examples::

    python tools/benchmark_theme_toggle.py --last 6
    python tools/benchmark_theme_toggle.py --check-regression
    python tools/benchmark_theme_toggle.py --log ~/.qube/logs/qube.log --last 4

Enable logging in the app first::

    QUBE_THEME_PROFILE=1 python main.py

Then toggle theme (stay on Conversations for apples-to-apples checks) and run
this tool against ``~/.qube/logs/qube.log``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.app_log_sink import default_app_log_path  # noqa: E402
from core.theme_toggle_profile import (  # noqa: E402
    ThemeProfileRegressionThresholds,
    check_theme_profile_regression,
    filter_hot_path_toggle_entries,
    format_theme_profile_table,
    parse_theme_profile_log,
)


def _read_log(path: Path) -> str:
    if not path.is_file():
        print(f"Log file not found: {path}", file=sys.stderr)
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"Error reading {path}: {exc}", file=sys.stderr)
        return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize ThemeProfile entries from qube.log",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help=f"Path to qube.log (default: {default_app_log_path()})",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=6,
        help="Show the last N ThemeProfile entries (default: 6)",
    )
    parser.add_argument(
        "--check-regression",
        action="store_true",
        help="Exit 1 if recent hot-path toggles exceed regression thresholds",
    )
    parser.add_argument(
        "--regression-last",
        type=int,
        default=2,
        help="When checking regression, evaluate the last N hot-path toggles (default: 2)",
    )
    parser.add_argument(
        "--max-total-ms",
        type=int,
        default=None,
        help="Override QUBE_THEME_REGRESS_MAX_TOTAL_MS for this run",
    )
    parser.add_argument(
        "--max-qss-apply-ms",
        type=int,
        default=None,
        help="Override QUBE_THEME_REGRESS_MAX_QSS_APPLY_MS for this run",
    )
    parser.add_argument(
        "--max-widget-count",
        type=int,
        default=None,
        help="Override QUBE_THEME_REGRESS_MAX_WIDGET_COUNT for this run",
    )
    parser.add_argument(
        "--max-built-stages",
        type=int,
        default=None,
        help="Override QUBE_THEME_REGRESS_MAX_BUILT_STAGES for this run",
    )
    args = parser.parse_args(argv)

    log_path = args.log or default_app_log_path()
    all_entries = parse_theme_profile_log(_read_log(log_path))
    if not all_entries:
        print(f"No ThemeProfile entries found in {log_path}", file=sys.stderr)
        print("Run with QUBE_THEME_PROFILE=1 and toggle the theme first.", file=sys.stderr)
        return 2

    entries = all_entries
    if args.last > 0:
        entries = entries[-args.last :]

    print(f"Log: {log_path}")
    print(format_theme_profile_table(entries))
    print()

    hot_path = filter_hot_path_toggle_entries(entries)
    if hot_path:
        totals = [entry.total_ms for entry in hot_path]
        qss_values = [entry.qss_apply_ms for entry in hot_path if entry.qss_apply_ms is not None]
        widget_values = [entry.widget_count for entry in hot_path if entry.widget_count is not None]
        print(
            f"Hot-path toggles in view: {len(hot_path)} | "
            f"total ms min/avg/max: "
            f"{min(totals)}/{sum(totals) // len(totals)}/{max(totals)}"
        )
        if qss_values:
            print(
                f"qss_apply ms min/avg/max: "
                f"{min(qss_values)}/{sum(qss_values) // len(qss_values)}/{max(qss_values)}"
            )
        if widget_values:
            print(
                f"widget_count min/avg/max: "
                f"{min(widget_values)}/{sum(widget_values) // len(widget_values)}/{max(widget_values)}"
            )
    else:
        print("No hot-path theme toggles in the selected window.")

    if not args.check_regression:
        return 0

    regression_entries = filter_hot_path_toggle_entries(all_entries)
    if args.regression_last > 0:
        regression_entries = regression_entries[-args.regression_last :]
    if not regression_entries:
        print("Regression check: no hot-path toggles in log", file=sys.stderr)
        return 2

    thresholds = ThemeProfileRegressionThresholds.from_env()
    if args.max_total_ms is not None:
        thresholds = ThemeProfileRegressionThresholds(
            max_total_ms=args.max_total_ms,
            max_qss_apply_ms=thresholds.max_qss_apply_ms,
            max_widget_count=thresholds.max_widget_count,
            max_built_stages=thresholds.max_built_stages,
        )
    if args.max_qss_apply_ms is not None:
        thresholds = ThemeProfileRegressionThresholds(
            max_total_ms=thresholds.max_total_ms,
            max_qss_apply_ms=args.max_qss_apply_ms,
            max_widget_count=thresholds.max_widget_count,
            max_built_stages=thresholds.max_built_stages,
        )
    if args.max_widget_count is not None:
        thresholds = ThemeProfileRegressionThresholds(
            max_total_ms=thresholds.max_total_ms,
            max_qss_apply_ms=thresholds.max_qss_apply_ms,
            max_widget_count=args.max_widget_count,
            max_built_stages=thresholds.max_built_stages,
        )
    if args.max_built_stages is not None:
        thresholds = ThemeProfileRegressionThresholds(
            max_total_ms=thresholds.max_total_ms,
            max_qss_apply_ms=thresholds.max_qss_apply_ms,
            max_widget_count=thresholds.max_widget_count,
            max_built_stages=args.max_built_stages,
        )

    violations = check_theme_profile_regression(regression_entries, thresholds)
    if not violations:
        print(
            f"Regression check: OK ({len(regression_entries)} hot-path toggle(s) reviewed)"
        )
        return 0

    print("Regression check: FAILED", file=sys.stderr)
    for violation in violations:
        print(f"  - {violation.message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
