#!/usr/bin/env python3
"""Stage 7 (milestone M4) — train an openWakeWord-compatible model from a config.

Entry point for the training half of the pipeline:
    python scripts/train.py --config configs/hey_qube.yaml

Flow:
  1. Fail-closed license gate (commercial tier refuses to train on non-commercial data).
  2. Assemble features: positive + augmented-positive embeddings (from the M3 synthetic
     clips) as label 1; the precomputed negative features + hard-negative embeddings as
     label 0; the FP-validation features for early-stopping.
  3. Train the classifier (lib/training.run_training) — weighted BCE, Adam, early-stop.
  4. Save models/<id>/<version>/checkpoint.pt + model_card.json (full provenance).

Then run ``scripts/export.py`` to emit the ``.onnx`` (+ optional ``.tflite``) Qube loads.

The heavy pieces (openWakeWord embedding model, torch) are imported lazily and run only in
the pinned training environment; the license gate, provenance collection, and model-card
assembly run anywhere.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import config as cfglib  # noqa: E402
from lib import licenses, model_card, stage, training  # noqa: E402

WAKEWORD_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = WAKEWORD_ROOT / "datasets"
MODELS_ROOT = WAKEWORD_ROOT / "models"

log = logging.getLogger("train")


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(WAKEWORD_ROOT), timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - env dependent
        return "unknown"


def collect_training_datasets(datasets_root: Path) -> list[dict]:
    """Gather dataset provenance (lock + license manifests) for the model card."""
    entries: list[dict] = []
    seen: set[str] = set()

    lock = licenses.read_lock(datasets_root)
    for key, rec in sorted(lock.get("datasets", {}).items()):
        archives = rec.get("archives", {})
        entries.append({
            "name": key,
            "version": rec.get("version", ""),
            "sha256": next(iter(archives.values()), "") if archives else "",
        })
        seen.add(key)

    for path in licenses.iter_manifest_paths(datasets_root):
        try:
            man = licenses.load_manifest(path)
        except (OSError, ValueError):
            continue
        name = man.dataset or man.path.stem
        if name in seen:
            continue
        seen.add(name)
        entries.append({
            "name": name,
            "version": str(man.data.get("dataset_version", "")),
            "license": man.license_id,
            "sha256": str(man.data.get("sha256", "")),
        })
    return entries


def _run_license_gate(config: dict, skip: bool) -> str:
    """Return the license-audit outcome: 'passed' | 'skipped'. Exits on failure."""
    require_commercial = bool(config.get("provenance", {}).get("require_commercial_license", False))
    if skip or not require_commercial:
        return "skipped"
    result = licenses.run_gate(DATASETS_ROOT, require_commercial=True)
    for warning in result.warnings:
        log.warning("WARN: %s", warning)
    if result.checked == 0:
        log.error("REFUSING TO TRAIN: commercial tier but no license manifests. "
                  "Run download_datasets.py + generate_positives.py first.")
        raise SystemExit(1)
    if not result.ok:
        for err in result.errors:
            log.error("FAIL: %s", err)
        log.error("REFUSING TO TRAIN: commercial license gate failed.")
        raise SystemExit(1)
    log.info("License gate passed (%d manifest(s)).", result.checked)
    return "passed"


def _assemble_features(config: dict, phrase_id: str):
    """Build (positives, negatives, validation) feature arrays (lazy, env-only)."""
    import numpy as np

    from lib import features  # lazy: needs openwakeword embedding model

    data = config.get("data", {})
    speech = DATASETS_ROOT / "speech"
    extractor = features.FeatureExtractor()

    def embed_dirs(dirs: list[Path]) -> "np.ndarray":
        files = [f for d in dirs for f in sorted(d.glob("*.wav"))]
        if not files:
            return np.empty((0, 16, 96), dtype=np.float32)
        tmp = DATASETS_ROOT / "features" / f"_tmp_{phrase_id}_{abs(hash(tuple(dirs)))}.npy"
        features.compute_features(files, tmp, extractor=extractor)
        arr = np.load(tmp)
        tmp.unlink(missing_ok=True)
        return arr

    positives = embed_dirs([speech / "positive" / phrase_id, speech / "positive-augmented" / phrase_id])

    neg_path = WAKEWORD_ROOT / data.get("negative_features", "")
    base_neg = np.load(neg_path, mmap_mode="r") if neg_path.is_file() else np.empty((0, 16, 96), np.float32)
    hard_neg = embed_dirs([speech / "hard-negative" / phrase_id])
    negatives = np.concatenate([np.asarray(base_neg), hard_neg], axis=0) if hard_neg.shape[0] else np.asarray(base_neg)

    val_path = WAKEWORD_ROOT / data.get("fp_validation_features", "")
    validation = np.load(val_path) if val_path.is_file() else None
    return positives, negatives, validation


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stage.add_config_arg(parser)
    parser.add_argument("--pilot", action="store_true", help="Use pilot examples/steps for a quick run.")
    parser.add_argument("--version", default="v0.1", help="Model version tag for the output dir.")
    parser.add_argument("--hardware", default="", help="Free-text hardware note for the model card.")
    parser.add_argument("--skip-license-gate", action="store_true",
                        help="DANGEROUS: skip the commercial gate (personal-use models only).")
    args = parser.parse_args(argv)

    config = cfglib.load_config(args.config)
    phrase_id = str(config.get("wakeword", {}).get("id", "wakeword"))
    spec = training.build_training_spec(config, pilot=args.pilot)
    audit = _run_license_gate(config, args.skip_license_gate)

    log.info("Assembling features for '%s' ...", phrase_id)
    positives, negatives, validation = _assemble_features(config, phrase_id)
    log.info("positives=%d negatives=%d validation=%s",
             positives.shape[0], negatives.shape[0],
             "none" if validation is None else validation.shape[0])
    if positives.shape[0] == 0 or negatives.shape[0] == 0:
        log.error("Need both positive and negative features. Run generate_positives.py / "
                  "hard_negative_mining.py / precompute_features.py first.")
        return 1

    log.info("Training: %d steps, layer_dim=%d, false_penalty=%d, seed=%d",
             spec.steps, spec.layer_dim, spec.false_penalty, spec.seed)
    started = time.time()
    state, metrics = training.run_training(spec, positives, negatives, validation)
    duration = time.time() - started

    out_dir = MODELS_ROOT / phrase_id / args.version
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch  # lazy
    ckpt_path = out_dir / "checkpoint.pt"
    torch.save({"state_dict": state, "layer_dim": spec.layer_dim, "config": config,
                "metrics": metrics, "spec": spec.__dict__}, ckpt_path)

    card = model_card.build_model_card(
        config, version=args.version, git_commit=git_commit(),
        training_datasets=collect_training_datasets(DATASETS_ROOT),
        metrics=metrics, hardware=args.hardware, duration_seconds=duration,
        license_audit=audit,
    )
    model_card.write_model_card(card, out_dir / "model_card.json")
    log.info("Saved %s + model_card.json", ckpt_path.name)
    log.info("Next: python scripts/export.py --config %s --version %s", args.config, args.version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
