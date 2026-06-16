#!/usr/bin/env python3
"""Stage 6 — train an openWakeWord model from a config.

Entry point for the whole pipeline:
    python scripts/train.py --config configs/hey_qube.yaml

Before any training starts, this runs the fail-closed license gate when the config
requests the commercial tier — production models refuse to train on non-commercial
data. After training it writes models/<phrase>/<version>/ + model_card.json.

Status: training core is a structured stub (milestone M4); the license gate is live.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import licenses, stage  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    stage.add_config_arg(parser)
    parser.add_argument("--pilot", action="store_true", help="Use pilot examples/steps for a quick run.")
    parser.add_argument(
        "--skip-license-gate",
        action="store_true",
        help="DANGEROUS: skip the commercial license gate (personal-use models only).",
    )
    args = parser.parse_args(argv)
    config = stage.load_stage_config(args)

    require_commercial = bool(
        config.get("provenance", {}).get("require_commercial_license", False)
    )
    if require_commercial and not args.skip_license_gate:
        datasets_root = stage.REPO_ROOT / "wakeword" / "datasets"
        if not datasets_root.exists():
            datasets_root = Path(__file__).resolve().parent.parent / "datasets"
        result = licenses.run_gate(datasets_root, require_commercial=True)
        for warning in result.warnings:
            print(f"WARN: {warning}")
        if result.checked == 0:
            print(
                "\nREFUSING TO TRAIN: commercial tier requires licensed datasets, but "
                "no license manifests were found. Run download_datasets.py first, or "
                "use --skip-license-gate for a personal-use model.",
                file=sys.stderr,
            )
            return 1
        if not result.ok:
            print(
                "\nREFUSING TO TRAIN: commercial license gate failed. "
                "Fix dataset provenance or use --skip-license-gate for a personal-use "
                "model (which can never be promoted to 'recommended').",
                file=sys.stderr,
            )
            for error in result.errors:
                print(f"  - {error}", file=sys.stderr)
            return 1
        print(f"License gate passed ({result.checked} asset manifest(s)).")

    raise stage.not_implemented(
        "M4",
        "openWakeWord training loop is not implemented yet.",
        [
            "Build the openWakeWord YAML training config from this file's "
            "training.* + data.* keys (examples/steps/false_penalty/layer_dim).",
            "Run the openWakeWord auto-train against the precomputed features.",
            "Checkpoint + early-stop on the FP validation set.",
            "Call export.py to emit .onnx + .tflite at 16 kHz.",
            "Write models/<id>/<version>/model_card.json (datasets, checksums, params, "
            "seed, oww_commit, hardware, duration, eval metrics, license tier).",
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main())
