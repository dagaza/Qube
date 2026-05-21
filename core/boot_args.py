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
    return parser.parse_args(argv)
