#!/usr/bin/env python3
"""Stage 7 — export a trained checkpoint to .onnx and .tflite at 16 kHz.

The exported model must be loadable by Qube's runtime (openwakeword==0.4.0,
16 kHz mono, 80 ms frames) and land at the install_hint path so it is auto-discovered
by core/wakeword_manager.py.

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
    parser.add_argument("--checkpoint", help="Path to the trained .pt checkpoint.")
    args = parser.parse_args(argv)
    stage.load_stage_config(args)

    raise stage.not_implemented(
        "M4",
        "Checkpoint -> ONNX/TFLite export is not implemented yet.",
        [
            "Convert the checkpoint to ONNX (onnx/onnxscript) and TFLite per export.formats.",
            "Verify input shape/sample-rate matches Qube's audio worker (16 kHz, CHUNK 1280).",
            "Write to models/<id>/<version>/<id>.onnx (+ .tflite).",
            "Print the install_hint so the user can copy it into ~/.qube/models/wakeword/.",
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main())
