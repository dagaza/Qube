#!/usr/bin/env python3
"""Stage 5 — augment positive clips with room reverb + background noise/music.

Mixes synthetic positive clips with RIRs (MIT IR Survey 16kHz) and background
noise/music (MUSAN/FSD50K/FMA) across a signal-to-noise sweep, for robustness in
real rooms. All augmentation sources must be commercially licensed.

Consumes the M3 synthetic positives (scripts/generate_positives.py) and is the #2
data-quality lever (far-field / noisy augmentation).

Status: structured stub (milestone M4).
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
    args = parser.parse_args(argv)
    stage.load_stage_config(args)

    raise stage.not_implemented(
        "M4",
        "Audio augmentation (RIR reverb + noise/music mixing) is not implemented yet.",
        [
            "Load RIRs from data.rir_paths and background audio from data.background_paths.",
            "For training.augmentation_rounds, convolve each positive clip with a "
            "random RIR and mix noise/music at a sampled SNR (use audiomentations).",
            "Keep everything 16 kHz mono.",
            "Feed augmented positives into the feature/embedding step for training.",
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main())
