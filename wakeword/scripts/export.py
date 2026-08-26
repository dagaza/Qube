#!/usr/bin/env python3
"""Stage 8 (milestone M4) — export a trained checkpoint to ONNX (+ optional TFLite).

Loads models/<id>/<version>/checkpoint.pt, rebuilds the classifier, and writes
<id>.onnx (the load-bearing artifact for Qube's runtime) plus, if onnx2tf is available,
<id>.tflite. The exported ONNX is verified against the runtime contract ((batch, 16, 96)
-> (batch, 1)) with a real onnxruntime inference before it's blessed.

Usage:
    python scripts/export.py --config configs/hey_qube.yaml --version v0.1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import config as cfglib  # noqa: E402
from lib import export as export_lib  # noqa: E402
from lib import licenses, model as model_lib  # noqa: E402

WAKEWORD_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = WAKEWORD_ROOT / "datasets"
MODELS_ROOT = WAKEWORD_ROOT / "models"

log = logging.getLogger("export")


def load_checkpoint(path: Path):
    """Load a checkpoint and rebuild the net with its trained weights (lazy torch)."""
    import torch  # lazy

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    net = model_lib.build_classifier(layer_dim=int(ckpt.get("layer_dim", 32)))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt


def _update_model_card(card_path: Path, export_info: dict) -> None:
    if not card_path.is_file():
        return
    card = json.loads(card_path.read_text(encoding="utf-8"))
    card["export"] = export_info
    card_path.write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to the wake word config YAML.")
    parser.add_argument("--version", default="v0.1", help="Model version to export.")
    parser.add_argument("--no-tflite", action="store_true", help="Skip the optional TFLite export.")
    args = parser.parse_args(argv)

    config = cfglib.load_config(args.config)
    phrase_id = str(config.get("wakeword", {}).get("id", "wakeword"))
    formats = [f.lower() for f in config.get("export", {}).get("formats", ["onnx"])]

    model_dir = MODELS_ROOT / phrase_id / args.version
    ckpt_path = model_dir / "checkpoint.pt"
    if not ckpt_path.is_file():
        log.error("No checkpoint at %s. Run train.py first.", ckpt_path)
        return 1

    net, _ = load_checkpoint(ckpt_path)

    onnx_path = model_dir / f"{phrase_id}.onnx"
    export_lib.export_onnx(net, onnx_path)
    in_shape, out_shape = export_lib.verify_onnx(onnx_path)
    log.info("Exported + verified %s  in=%s out=%s", onnx_path.name, in_shape, out_shape)

    export_info = {"onnx": onnx_path.name, "input_shape": in_shape, "output_shape": out_shape}

    if "tflite" in formats and not args.no_tflite:
        tflite_path = export_lib.export_tflite(onnx_path, model_dir / f"{phrase_id}.tflite")
        if tflite_path:
            log.info("Exported %s", tflite_path.name)
            export_info["tflite"] = tflite_path.name

    licenses.write_manifest(
        onnx_path,
        datasets_root=MODELS_ROOT,
        dataset=f"trained-model/{phrase_id}",
        source_url="https://github.com/dscripka/openWakeWord",
        license_id="Apache-2.0",
        commercial_use=True,
        attribution="openWakeWord architecture (Apache-2.0); trained on commercial-allowlisted data.",
        dataset_version=args.version,
        notes="Exported wake word model for Qube's openWakeWord runtime.",
    )
    _update_model_card(model_dir / "model_card.json", export_info)

    install_hint = config.get("export", {}).get("install_hint", f"~/.qube/models/wakeword/en/{phrase_id}/")
    log.info("Done. Install into Qube by copying %s to %s", onnx_path.name, install_hint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
