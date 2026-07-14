#!/usr/bin/env python3
"""Stage 3-6 orchestrator (milestone M3) — run the phonetic pilot sweep.

Drives the cheap M3 experiment: for each candidate spelling in
``configs/experiments.yaml`` it generates diverse positives + hard negatives at the
*pilot* budget, then (once M4/M5 land) trains + evaluates each and ranks them by the
operating-point rule in ``docs/evaluation.md`` to pick a winner per word-class.

Stages:
  * ``plan``  (default) — print the variants, pilot budget, and selection rule.
  * ``data``  — synthesize pilot positives + hard negatives per variant and write a
                run plan (``results/pilot_sweep_plan.json``) with the exact train command
                for each variant. This is fully runnable today.
  * ``rank``  — read per-variant eval metrics (``--results``) and select winners. The
                metrics themselves come from evaluate.py (M5); the ranking is live.

Usage:
    python scripts/run_pilot_sweep.py --experiments configs/experiments.yaml
    python scripts/run_pilot_sweep.py --experiments configs/experiments.yaml \
        --config configs/qube.yaml --stage data --pilot
    python scripts/run_pilot_sweep.py --experiments configs/experiments.yaml \
        --stage rank --results results/pilot_metrics.json
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_positives as gp  # noqa: E402
import hard_negative_mining as hnm  # noqa: E402
from lib import config as cfglib  # noqa: E402
from lib import experiments, tts  # noqa: E402

WAKEWORD_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = WAKEWORD_ROOT / "datasets"
RESULTS_ROOT = WAKEWORD_ROOT / "results"

log = logging.getLogger("run_pilot_sweep")


def variant_config(base: dict, variant: experiments.PilotVariant, pilot: experiments.PilotParams) -> dict:
    """Derive a per-variant config from a base config at the pilot budget."""
    cfg = copy.deepcopy(base)
    cfg.setdefault("wakeword", {})
    cfg["wakeword"]["id"] = variant.id
    cfg["wakeword"]["phrase"] = variant.phrase
    cfg.setdefault("training", {})
    cfg["training"]["examples"] = pilot.examples
    cfg["training"]["steps"] = pilot.steps
    cfg["training"]["false_penalty"] = pilot.false_penalty
    cfg["training"]["seed"] = pilot.seed
    return cfg


def print_plan(variants: list[experiments.PilotVariant], pilot: experiments.PilotParams, rule: experiments.SelectionRule) -> None:
    print("Pilot sweep plan")
    print(f"  budget: {pilot.examples} examples / {pilot.steps} steps / "
          f"false_penalty={pilot.false_penalty} / seed={pilot.seed}")
    print(f"  selection: recall >= {rule.min_recall}, FP/hr <= {rule.max_false_positives_per_hour}, "
          f"tie-break {rule.tie_breaker}")
    print(f"  variants ({len(variants)}):")
    for v in variants:
        print(f"    [{v.word_class:11s}] {v.id:12s} phrase='{v.phrase}'")


def run_data_stage(
    base: dict,
    variants: list[experiments.PilotVariant],
    pilot: experiments.PilotParams,
    *,
    datasets_root: Path,
    synth_fn: tts.SynthFn,
    num_speakers: int = tts.DEFAULT_NUM_SPEAKERS,
) -> list[dict]:
    """Synthesize pilot positives + hard negatives for each variant; return a run plan."""
    plan_entries: list[dict] = []
    for variant in variants:
        cfg = variant_config(base, variant, pilot)
        log.info("== variant %s (%s) ==", variant.id, variant.word_class)
        positives, _ = gp.generate_positives(
            cfg, datasets_root=datasets_root, count=pilot.examples,
            synth_fn=synth_fn, num_speakers=num_speakers,
        )
        negatives, _ = hnm.mine_hard_negatives(
            cfg, datasets_root=datasets_root, count=pilot.examples,
            synth_fn=synth_fn, num_speakers=num_speakers,
        )
        plan_entries.append({
            "variant_id": variant.id,
            "word_class": variant.word_class,
            "phrase": variant.phrase,
            "positives": len(positives),
            "hard_negatives": len(negatives),
            "train_command": (
                f"python scripts/train.py --config configs/{variant.id}.yaml --pilot"
            ),
        })
    return plan_entries


def load_results(path: Path, variants: list[experiments.PilotVariant]) -> list[experiments.VariantResult]:
    """Parse an eval-metrics JSON into :class:`VariantResult` objects."""
    word_class = {v.id: v.word_class for v in variants}
    raw = json.loads(path.read_text(encoding="utf-8"))
    results: list[experiments.VariantResult] = []
    for entry in raw:
        vid = str(entry["variant_id"])
        results.append(
            experiments.VariantResult(
                variant_id=vid,
                word_class=str(entry.get("word_class") or word_class.get(vid, "")),
                recall=float(entry["recall"]),
                false_positives_per_hour=float(entry["false_positives_per_hour"]),
                noisy_room_robustness=float(entry.get("noisy_room_robustness", 0.0)),
                latency_ms=float(entry.get("latency_ms", 0.0)),
            )
        )
    return results


def run_rank_stage(
    results: list[experiments.VariantResult],
    rule: experiments.SelectionRule,
    *,
    results_root: Path,
) -> dict:
    ranked = experiments.rank_variants(results, rule)
    winners = experiments.select_winners(results, rule)

    print("Ranked pilot results:")
    for r in ranked:
        flag = "OK " if r.meets(rule) else "-- "
        print(f"  {flag}[{r.word_class:11s}] {r.variant_id:12s} "
              f"recall={r.recall:.3f} FP/hr={r.false_positives_per_hour:.3f} "
              f"noisy={r.noisy_room_robustness:.3f}")
    print("Winners:")
    for word_class, winner in winners.items():
        print(f"  {word_class}: {winner.variant_id if winner else '(none met ship criteria)'}")

    summary = {
        "ranked": [r.__dict__ for r in ranked],
        "winners": {k: (v.variant_id if v else None) for k, v in winners.items()},
    }
    results_root.mkdir(parents=True, exist_ok=True)
    out = results_root / "pilot_sweep_results.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log.info("Wrote %s", out)
    return summary


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", default="configs/experiments.yaml", help="Experiment sweep config.")
    parser.add_argument("--config", default="configs/qube.yaml", help="Base config (data paths / provenance).")
    parser.add_argument("--stage", choices=["plan", "data", "rank"], default="plan")
    parser.add_argument("--pilot", action="store_true", help="(data) Use the pilot budget (recommended).")
    parser.add_argument("--voice", default=None, help="Path to a Piper voice .onnx (else auto-download).")
    parser.add_argument("--num-speakers", type=int, default=tts.DEFAULT_NUM_SPEAKERS)
    parser.add_argument("--results", default=None, help="(rank) JSON of per-variant eval metrics.")
    args = parser.parse_args(argv)

    exp = cfglib.load_config(args.experiments)
    variants = experiments.expand_variants(exp)
    pilot = experiments.parse_pilot_params(exp)
    rule = experiments.parse_selection_rule(exp)

    if args.stage == "plan":
        print_plan(variants, pilot, rule)
        return 0

    if args.stage == "rank":
        if not args.results:
            parser.error("--stage rank requires --results <metrics.json>")
        results = load_results(Path(args.results), variants)
        run_rank_stage(results, rule, results_root=RESULTS_ROOT)
        return 0

    # data stage
    base = cfglib.load_config(args.config)
    voice_path = tts.resolve_voice(args.voice)
    backend = tts.PiperBackend(voice_path)
    plan_entries = run_data_stage(
        base, variants, pilot, datasets_root=DATASETS_ROOT,
        synth_fn=backend, num_speakers=args.num_speakers,
    )
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    plan_path = RESULTS_ROOT / "pilot_sweep_plan.json"
    plan_path.write_text(json.dumps(plan_entries, indent=2) + "\n", encoding="utf-8")
    log.info("Wrote run plan -> %s", plan_path)
    log.info("Next: train each variant (M4) then re-run with --stage rank --results <metrics>.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
