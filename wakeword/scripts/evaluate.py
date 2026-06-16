#!/usr/bin/env python3
"""Stage 8 — evaluate a trained model on the held-out, real-voice corpus.

Produces results/<id>/<version>/eval.json + a markdown report with recall, false
positives/hour, precision, latency, and quiet-vs-noisy robustness across a threshold
sweep (0.3-0.7). The recommended threshold feeds Qube's set_wakeword_threshold_override.

See docs/evaluation.md. Final human sign-off goes through the existing Wakeword Test Lab.

Status: structured stub (milestone M5).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import stage  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    stage.add_config_arg(parser)
    parser.add_argument("--model", help="Path to the .onnx model to evaluate.")
    parser.add_argument(
        "--corpus",
        default="evaluation/corpus.json",
        help="Held-out evaluation corpus index.",
    )
    args = parser.parse_args(argv)
    stage.load_stage_config(args)

    raise stage.not_implemented(
        "M5",
        "Held-out evaluation + report generation is not implemented yet.",
        [
            "Load the corpus index (positives, negatives, adversarial) from --corpus.",
            "Run the model over each clip at thresholds 0.3..0.7.",
            "Compute recall, FP/hour, precision, latency, quiet-vs-noisy robustness.",
            "Pick recommended_threshold (max recall s.t. FP/hr <= target).",
            "Write results/<id>/<version>/eval.json + markdown; confirm in the Test Lab.",
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main())
