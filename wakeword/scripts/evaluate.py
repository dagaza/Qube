#!/usr/bin/env python3
"""Stage 9 (milestone M5) — evaluate a trained model on the held-out, real-voice corpus.

Runs the exported ``<id>.onnx`` over every clip in ``evaluation/corpus.json`` (real voices,
never TTS), then computes the operating-point metrics that decide shippability — recall/FRR,
false-accepts per hour, precision, adversarial false-accept rate, DET/ROC, latency — across
a threshold sweep, and picks the recommended threshold (max recall s.t. FP/hr <= target).

Outputs ``results/<id>/<version>/eval.json`` + ``eval.md``. The recommended threshold seeds
Qube's ``set_wakeword_threshold_override``; final human sign-off goes through the existing
Wakeword Test Lab (Settings -> Wakeword). Can also emit a sweep-compatible metrics entry so
``run_pilot_sweep.py --stage rank`` ranks variants on real numbers.

Model inference (openWakeWord) is lazy; the metric aggregation is pure and unit-tested.

Usage:
    python scripts/evaluate.py --config configs/hey_qube.yaml --version v0.1
    python scripts/evaluate.py --config configs/qube.yaml --corpus evaluation/corpus.json \
        --emit-sweep-metrics results/pilot_metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import config as cfglib  # noqa: E402
from lib import corpus as corpus_lib  # noqa: E402
from lib import metrics as metrics_lib  # noqa: E402

WAKEWORD_ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = WAKEWORD_ROOT / "models"
RESULTS_ROOT = WAKEWORD_ROOT / "results"

log = logging.getLogger("evaluate")

ClipScorer = Callable[[corpus_lib.ClipEntry], "float | tuple[float, float]"]
LongformScorer = Callable[[corpus_lib.LongformEntry], "list[float]"]


def _normalize(result) -> tuple[float, float | None]:
    if isinstance(result, tuple):
        return float(result[0]), float(result[1])
    return float(result), None


def evaluate_corpus(
    config: dict,
    corpus: corpus_lib.Corpus,
    *,
    clip_scorer: ClipScorer,
    longform_scorer: LongformScorer,
    version: str = "",
    thresholds: list[float] | None = None,
    selection: dict | None = None,
) -> dict:
    """Core (test-injectable): score the corpus + assemble the eval report dict."""
    thresholds = thresholds or metrics_lib.default_thresholds()
    sel = selection or {}
    max_fp = float(sel.get("max_false_positives_per_hour", 1.0))
    min_recall = float(sel.get("min_recall", 0.85))

    positives: list[dict] = []
    for entry in corpus.positives:
        score, latency = _normalize(clip_scorer(entry))
        row = {"score": score, "environment": entry.environment}
        if latency is not None:
            row["latency_ms"] = latency
        positives.append(row)
    adversarial = [{"score": _normalize(clip_scorer(e))[0]} for e in corpus.adversarial]
    longform = [
        {"fire_scores": list(longform_scorer(e)), "duration_seconds": e.duration_seconds}
        for e in corpus.negatives_longform
    ]

    swept = metrics_lib.sweep(positives, adversarial, longform, thresholds)
    verdict = metrics_lib.verdict(swept, max_fp_per_hour=max_fp, min_recall=min_recall)
    at = next((m for m in swept if m.threshold == verdict.recommended_threshold), None)

    return {
        "wakeword_id": str(config.get("wakeword", {}).get("id", "")),
        "model_version": version,
        "corpus": corpus.summary(),
        "selection": {"max_false_positives_per_hour": max_fp, "min_recall": min_recall},
        "thresholds": {
            str(m.threshold): {
                "recall": round(m.recall, 4),
                "frr": round(m.frr, 4),
                "precision": round(m.precision, 4),
                "fp_per_hour": round(m.fp_per_hour, 4),
                "adversarial_far": round(m.adversarial_far, 4),
                "latency_ms_p50": round(m.latency_ms_p50, 1),
                "recall_quiet": round(m.recall_quiet, 4),
                "recall_noisy": round(m.recall_noisy, 4),
            }
            for m in swept
        },
        "recommended_threshold": verdict.recommended_threshold,
        "verdict": "pass" if verdict.passed else "fail",
        "verdict_reasons": verdict.reasons,
        "robustness": {
            "quiet_recall": round(at.recall_quiet, 4) if at else None,
            "noisy_recall": round(at.recall_noisy, 4) if at else None,
        },
        "latency_ms": {
            "p50": round(at.latency_ms_p50, 1) if at else None,
            "p95": round(at.latency_ms_p95, 1) if at else None,
        },
        "roc": metrics_lib.roc_points(swept),
        "det": metrics_lib.det_points(swept),
    }


def sweep_metric_entry(eval_dict: dict) -> dict:
    """Reduce a full eval to the compact row run_pilot_sweep.py --stage rank consumes."""
    rec = eval_dict.get("recommended_threshold")
    at = eval_dict["thresholds"].get(str(rec), {}) if rec is not None else {}
    return {
        "variant_id": eval_dict["wakeword_id"],
        "recall": at.get("recall", 0.0),
        "false_positives_per_hour": at.get("fp_per_hour", 999.0),
        "noisy_room_robustness": at.get("recall_noisy", 0.0),
        "latency_ms": at.get("latency_ms_p50", 0.0),
    }


def write_report(eval_dict: dict, path: Path) -> Path:
    lines = [
        f"# Wake word evaluation — {eval_dict['wakeword_id']} {eval_dict.get('model_version', '')}",
        "",
        f"- Corpus: {eval_dict['corpus']}",
        f"- Verdict: **{eval_dict['verdict'].upper()}**  "
        f"(recommended threshold: {eval_dict['recommended_threshold']})",
        f"- Robustness: quiet recall {eval_dict['robustness']['quiet_recall']} / "
        f"noisy recall {eval_dict['robustness']['noisy_recall']}",
        f"- Latency: p50 {eval_dict['latency_ms']['p50']} ms / p95 {eval_dict['latency_ms']['p95']} ms",
        "",
        "| Threshold | Recall | FRR | Precision | FP/hr | Adv-FAR |",
        "|---|---|---|---|---|---|",
    ]
    for t, m in eval_dict["thresholds"].items():
        lines.append(
            f"| {t} | {m['recall']} | {m['frr']} | {m['precision']} | "
            f"{m['fp_per_hour']} | {m['adversarial_far']} |"
        )
    if eval_dict["verdict_reasons"]:
        lines += ["", "## Notes", *[f"- {r}" for r in eval_dict["verdict_reasons"]]]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --- real (lazy) model scorers ---------------------------------------------------

def _build_model(model_path: Path):
    from openwakeword.model import Model  # lazy: needs onnxruntime + oww

    return Model(wakeword_model_paths=[str(model_path)])


def _predict_scores(model, wav_path: Path) -> list[float]:
    """Stream a WAV through the model, returning the per-frame score sequence."""
    from lib import audio  # lazy: soundfile

    model.reset()
    signal = audio.read_mono_16k(wav_path)
    pcm = (signal * 32767.0).astype("int16")
    name = next(iter(model.models.keys()))
    scores: list[float] = []
    for start in range(0, len(pcm) - 1280 + 1, 1280):
        result = model.predict(pcm[start : start + 1280])
        scores.append(float(result[name]))
    return scores or [0.0]


def _peak_fires(scores: list[float], *, floor: float = 0.1, refractory: int = 10) -> list[float]:
    """De-duplicated activation peaks: local maxima above ``floor``, refractory-gated."""
    fires: list[float] = []
    last = -refractory
    for i, s in enumerate(scores):
        if s >= floor and i - last >= refractory:
            fires.append(s)
            last = i
    return fires


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to the wake word config YAML.")
    parser.add_argument("--version", default="v0.1", help="Model version to evaluate.")
    parser.add_argument("--model", default=None, help="Path to the .onnx model (else models/<id>/<version>/<id>.onnx).")
    parser.add_argument("--corpus", default="evaluation/corpus.json", help="Held-out corpus index.")
    parser.add_argument("--emit-sweep-metrics", default=None, help="Append a sweep-rank metrics row to this JSON.")
    args = parser.parse_args(argv)

    config = cfglib.load_config(args.config)
    phrase_id = str(config.get("wakeword", {}).get("id", "wakeword"))
    corpus = corpus_lib.load_corpus(args.corpus)
    log.info("Corpus: %s", corpus.summary())

    model_path = Path(args.model) if args.model else MODELS_ROOT / phrase_id / args.version / f"{phrase_id}.onnx"
    if not model_path.is_file():
        log.error("No model at %s. Run train.py + export.py first.", model_path)
        return 1
    model = _build_model(model_path)

    def clip_scorer(entry: corpus_lib.ClipEntry) -> float:
        return max(_predict_scores(model, entry.path))

    def longform_scorer(entry: corpus_lib.LongformEntry) -> list[float]:
        return _peak_fires(_predict_scores(model, entry.path))

    started = time.time()
    eval_dict = evaluate_corpus(
        config, corpus, clip_scorer=clip_scorer, longform_scorer=longform_scorer,
        version=args.version, selection=_selection_from_config(config),
    )
    eval_dict["wall_time_seconds"] = round(time.time() - started, 2)

    out_dir = RESULTS_ROOT / phrase_id / args.version
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval.json").write_text(json.dumps(eval_dict, indent=2) + "\n", encoding="utf-8")
    write_report(eval_dict, out_dir / "eval.md")
    log.info("Verdict: %s (threshold %s). Wrote %s",
             eval_dict["verdict"], eval_dict["recommended_threshold"], out_dir / "eval.json")

    if args.emit_sweep_metrics:
        _append_sweep_metric(Path(args.emit_sweep_metrics), sweep_metric_entry(eval_dict))
    return 0 if eval_dict["verdict"] == "pass" else 2


def _selection_from_config(config: dict) -> dict:
    prov = config.get("evaluation", {})
    return {
        "max_false_positives_per_hour": prov.get("max_false_positives_per_hour", 1.0),
        "min_recall": prov.get("min_recall", 0.85),
    }


def _append_sweep_metric(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    existing = [e for e in existing if e.get("variant_id") != entry["variant_id"]]
    existing.append(entry)
    path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
