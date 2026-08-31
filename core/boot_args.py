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
    parser.add_argument(
        "--mock-bootstrap-download",
        action="store_true",
        help="Simulate bootstrap downloads on a timer (no files fetched).",
    )
    parser.add_argument(
        "--winget-validation",
        action="store_true",
        help=(
            "Defer llama.cpp / CUDA DLL loads for WinGet-style post-install validation "
            "(also sets QUBE_WINGET_VALIDATION=1)."
        ),
    )
    parser.add_argument(
        "--bootstrap-trace",
        action="store_true",
        help=(
            "Write granular bootstrap/launch steps to bootstrap-trace.jsonl and stderr "
            "(also sets QUBE_BOOTSTRAP_TRACE=1; run from a terminal on Windows)."
        ),
    )
    return parser.parse_args(argv)
