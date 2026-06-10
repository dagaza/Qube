"""CLI argument parsing for app startup."""

from __future__ import annotations

import argparse


def parse_boot_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Qube desktop assistant.")
    parser.add_argument(
        "--routing-debug",
        action="store_true",
        help="Open the routing debug view as a detached side tool window.",
    )
    parser.add_argument(
        "--trace-diff-debug",
        action="store_true",
        help="Open the canonical trace diff debugger as a detached tool window.",
    )
    parser.add_argument(
        "--run-scenario",
        default="",
        metavar="PATH",
        help=(
            "After startup, open the guided scenario comparison workflow "
            "(Qube pathway with model gate, then external pathway after LM Studio is ready)."
        ),
    )
    parser.add_argument(
        "--scenario-single-phase",
        action="store_true",
        help="With --run-scenario, run only the Qube pathway phase (still requires model loaded).",
    )
    parser.add_argument(
        "--scenario-backend",
        choices=("qube", "external"),
        default="qube",
        help="Legacy single-backend hint; prefer the guided workflow from --run-scenario.",
    )
    parser.add_argument(
        "--compare-sessions",
        nargs=2,
        metavar=("SESSION_A", "SESSION_B"),
        help="After startup, compare two saved session JSON files offline.",
    )
    return parser.parse_args(argv)
