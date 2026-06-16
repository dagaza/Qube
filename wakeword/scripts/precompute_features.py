#!/usr/bin/env python3
"""Stage 4 — precompute openWakeWord embedding features (.npy) for negatives + validation.

THIS IS THE CRITICAL MILESTONE (M2). It is the FOSS replacement for the notebook's
non-commercial ACAV100M / validation_set feature files: instead of downloading NC
features, we generate our own from clean, commercially-licensed speech (LibriSpeech)
and noise (MUSAN) by running them through the Apache-2.0 openWakeWord embedding model.

Outputs:
  - data.negative_features        (bulk negative training features)
  - data.fp_validation_features   (held-out false-positive validation set)

Status: structured stub (milestone M2).
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
    parser.add_argument(
        "--set",
        choices=["negatives", "validation", "all"],
        default="all",
        help="Which feature set to compute.",
    )
    args = parser.parse_args(argv)
    stage.load_stage_config(args)

    raise stage.not_implemented(
        "M2",
        "Self-computed negative/validation features are not implemented yet "
        "(the core FOSS-licensing fix).",
        [
            "Load the openWakeWord melspectrogram + embedding models (Apache-2.0, "
            "16 kHz) matching the runtime in Qube's requirements.txt (openwakeword==0.4.0).",
            "Stream LibriSpeech (CC-BY-4.0) clips -> melspec -> embedding features.",
            "Append to a memory-mapped .npy at data.negative_features (shape (-1,16,96)).",
            "Build the FP validation set from MUSAN + LibriSpeech dev -> "
            "data.fp_validation_features (~11h equivalent).",
            "Write provenance manifests for the generated .npy (dataset='LibriSpeech/"
            "MUSAN', commercial_use=true) so verify_licenses.py stays green.",
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main())
