"""Tests for the M3 pilot-sweep orchestrator (run_pilot_sweep.py).

Synthesis is injected (no Piper); training/eval are out of scope (M4/M5). We exercise
per-variant config derivation, the runnable data stage, and the ranking stage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_pilot_sweep as sweep  # noqa: E402
from lib import experiments, tts  # noqa: E402

EXP = {
    "pilot": {"examples": 6, "steps": 10000, "false_penalty": 2500, "seed": 1337},
    "variants": {
        "single_word": [
            {"id": "keube", "phrase": "keube"},
            {"id": "cube", "phrase": "cube"},
        ],
        "two_word": [{"id": "hey_keube", "phrase": "hey_keube"}],
    },
    "selection": {"max_false_positives_per_hour": 1.0, "min_recall": 0.85,
                  "tie_breaker": "noisy_room_robustness"},
}

BASE = {
    "wakeword": {"id": "keube", "phrase": "keube", "adversarial_phrases": ["cube", "tube"]},
    "training": {"examples": 50000},
    "provenance": {"tier": "commercial"},
}


def _fake_synth(phrase: str, params: tts.SynthesisParams, out_path: Path) -> None:
    out_path.write_bytes(b"fake-wav")


def test_variant_config_overrides_identity_and_budget() -> None:
    variants = experiments.expand_variants(EXP)
    pilot = experiments.parse_pilot_params(EXP)
    cfg = sweep.variant_config(BASE, variants[1], pilot)  # "cube"
    assert cfg["wakeword"]["id"] == "cube"
    assert cfg["wakeword"]["phrase"] == "cube"
    assert cfg["training"]["examples"] == 6
    # base config is not mutated
    assert BASE["wakeword"]["id"] == "keube"
    assert BASE["training"]["examples"] == 50000


def test_run_data_stage_generates_all_variants(tmp_path: Path) -> None:
    variants = experiments.expand_variants(EXP)
    pilot = experiments.parse_pilot_params(EXP)
    entries = sweep.run_data_stage(
        BASE, variants, pilot, datasets_root=tmp_path, synth_fn=_fake_synth, num_speakers=10
    )
    assert len(entries) == 3
    for entry in entries:
        assert entry["positives"] == 6
        assert entry["hard_negatives"] == 6
        assert entry["train_command"].startswith("python scripts/train.py")
    assert (tmp_path / "speech" / "positive" / "keube").is_dir()
    assert (tmp_path / "speech" / "hard-negative" / "cube").is_dir()


def test_load_results_and_rank(tmp_path: Path) -> None:
    variants = experiments.expand_variants(EXP)
    metrics = [
        {"variant_id": "keube", "recall": 0.95, "false_positives_per_hour": 0.2, "noisy_room_robustness": 0.7},
        {"variant_id": "cube", "recall": 0.99, "false_positives_per_hour": 5.0},
        {"variant_id": "hey_keube", "recall": 0.97, "false_positives_per_hour": 0.1},
    ]
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    results = sweep.load_results(metrics_path, variants)
    assert {r.variant_id for r in results} == {"keube", "cube", "hey_keube"}
    # word_class backfilled from the variant table
    assert next(r for r in results if r.variant_id == "keube").word_class == "single_word"

    rule = experiments.parse_selection_rule(EXP)
    summary = sweep.run_rank_stage(results, rule, results_root=tmp_path)
    # cube fails the FP gate, so keube wins single_word
    assert summary["winners"]["single_word"] == "keube"
    assert summary["winners"]["two_word"] == "hey_keube"
    assert (tmp_path / "pilot_sweep_results.json").is_file()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
