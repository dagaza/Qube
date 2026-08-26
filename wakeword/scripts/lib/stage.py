"""Shared CLI scaffolding for pipeline-stage scripts.

The data/feature/train/eval stages are structured stubs at the M0/M1 scaffold
phase: they parse args, load + validate the config, and then raise a clear
``PipelineNotImplemented`` describing exactly what milestone implements them. This
keeps the contract (CLI + config keys) stable while the heavy ML work lands later.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class PipelineNotImplemented(NotImplementedError):
    """Raised by stub stages so the gap is explicit, not a silent no-op."""


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a wake word config YAML (e.g. configs/hey_qube.yaml).",
    )


def load_stage_config(args: argparse.Namespace) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from lib.config import load_config  # local import keeps gate stdlib-only

    return load_config(args.config)


def not_implemented(milestone: str, summary: str, steps: list[str]) -> "PipelineNotImplemented":
    lines = [f"[{milestone}] {summary}", "", "Implementation outline:"]
    lines.extend(f"  {i}. {step}" for i, step in enumerate(steps, 1))
    lines.append("")
    lines.append("See docs/roadmap.md for milestone context.")
    return PipelineNotImplemented("\n".join(lines))
