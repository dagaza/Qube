#!/usr/bin/env python3
"""Stage 3 — generate synthetic positive + adversarial speech with Piper TTS.

Reads ``wakeword.phrase`` and ``wakeword.adversarial_phrases`` from the config and
synthesizes many pronunciations (varying voice, speed, pitch). Adversarial sound-alike
phrases are generated as hard negatives to lower the false-accept rate.

Status: structured stub (milestone M2/M3). Piper is Apache/MIT licensed (commercial OK).
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
    parser.add_argument("--pilot", action="store_true", help="Use the small pilot example count.")
    args = parser.parse_args(argv)
    stage.load_stage_config(args)

    raise stage.not_implemented(
        "M2/M3",
        "Piper TTS positive/adversarial clip generation is not implemented yet.",
        [
            "Load the Piper voice (en_US-libritts_r-medium) — MIT/Apache.",
            "Synthesize `training.examples` clips of `wakeword.phrase` with voice/"
            "speed/pitch variation.",
            "Synthesize adversarial clips for each `wakeword.adversarial_phrases`.",
            "Write 16 kHz mono WAVs to datasets/speech/positive/ and /adversarial/.",
            "Emit one example WAV to listen-check pronunciation (Section 1 of notebook).",
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main())
