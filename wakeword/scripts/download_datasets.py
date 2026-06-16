#!/usr/bin/env python3
"""Stage 1 — download commercially-licensed datasets and write provenance manifests.

For every asset fetched, this stage MUST write a sidecar ``<asset>.license.json``
(see datasets/licenses/manifest.schema.json) and update
``datasets/licenses/manifest.lock.json`` with versions + sha256 checksums.

Status: structured stub (milestone M1). The CLI + config contract are stable; the
download/manifest logic is the implementation task.
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
        "--only",
        choices=["background-noise", "music", "room-impulse", "speech"],
        help="Download a single dataset category instead of all.",
    )
    args = parser.parse_args(argv)
    stage.load_stage_config(args)

    raise stage.not_implemented(
        "M1",
        "Dataset download + provenance manifest writing is not implemented yet.",
        [
            "Resolve sources from docs/replacements.md (LibriSpeech, MUSAN, FSD50K "
            "BY/CC0 subset, FMA commercial cut, MIT IR Survey 16kHz).",
            "Download into datasets/<category>/ at 16 kHz mono.",
            "Write a <asset>.license.json next to every file (schema: "
            "datasets/licenses/manifest.schema.json).",
            "Record sha256 + dataset_version into datasets/licenses/manifest.lock.json.",
            "Run verify_licenses.py --require-commercial and ensure it passes.",
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main())
