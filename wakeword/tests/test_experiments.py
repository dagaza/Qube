"""Tests for pilot-sweep planning + winner selection (lib/experiments.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from lib import experiments  # noqa: E402


SAMPLE = {
    "pilot": {"examples": 5000, "steps": 10000, "false_penalty": 2500, "seed": 1337},
    "variants": {
        "single_word": [
            {"id": "keube", "phrase": "keube"},
            {"id": "cube", "phrase": "cube"},
        ],
        "two_word": [
            {"id": "hey_keube", "phrase": "hey_keube"},
        ],
    },
    "selection": {
        "max_false_positives_per_hour": 1.0,
        "min_recall": 0.85,
        "tie_breaker": "noisy_room_robustness",
    },
}


def test_parse_pilot_params() -> None:
    p = experiments.parse_pilot_params(SAMPLE)
    assert (p.examples, p.steps, p.false_penalty, p.seed) == (5000, 10000, 2500, 1337)


def test_parse_pilot_params_defaults_on_empty() -> None:
    p = experiments.parse_pilot_params({})
    assert p.examples == 5000 and p.steps == 10000


def test_parse_selection_rule() -> None:
    rule = experiments.parse_selection_rule(SAMPLE)
    assert rule.min_recall == 0.85
    assert rule.max_false_positives_per_hour == 1.0
    assert rule.tie_breaker == "noisy_room_robustness"


def test_expand_variants_flattens_with_word_class() -> None:
    variants = experiments.expand_variants(SAMPLE)
    assert len(variants) == 3
    assert {v.word_class for v in variants} == {"single_word", "two_word"}
    keube = next(v for v in variants if v.id == "keube")
    assert keube.phrase == "keube"


def test_expand_variants_rejects_duplicate_ids() -> None:
    bad = {"variants": {"a": [{"id": "x", "phrase": "x"}], "b": [{"id": "x", "phrase": "y"}]}}
    with pytest.raises(ValueError, match="Duplicate"):
        experiments.expand_variants(bad)


def test_expand_variants_rejects_empty() -> None:
    with pytest.raises(ValueError, match="No variants"):
        experiments.expand_variants({"variants": {}})


def _r(vid, recall, fp, noisy=0.0, latency=0.0, wc="single_word") -> experiments.VariantResult:
    return experiments.VariantResult(
        variant_id=vid, word_class=wc, recall=recall,
        false_positives_per_hour=fp, noisy_room_robustness=noisy, latency_ms=latency,
    )


def test_meets_gate() -> None:
    rule = experiments.SelectionRule(max_false_positives_per_hour=1.0, min_recall=0.85)
    assert _r("a", 0.90, 0.5).meets(rule)
    assert not _r("b", 0.80, 0.5).meets(rule)  # low recall
    assert not _r("c", 0.90, 2.0).meets(rule)  # too many FPs


def test_rank_variants_passing_beat_failing() -> None:
    rule = experiments.parse_selection_rule(SAMPLE)
    results = [
        _r("fail_high_recall", 0.99, 5.0),  # fails FP gate despite best recall
        _r("pass_low", 0.86, 0.2),
        _r("pass_high", 0.95, 0.2),
    ]
    ranked = experiments.rank_variants(results, rule)
    assert [r.variant_id for r in ranked] == ["pass_high", "pass_low", "fail_high_recall"]


def test_rank_variants_tie_breaks_on_robustness() -> None:
    rule = experiments.parse_selection_rule(SAMPLE)
    results = [
        _r("a", 0.90, 0.2, noisy=0.5),
        _r("b", 0.90, 0.2, noisy=0.9),
    ]
    ranked = experiments.rank_variants(results, rule)
    assert ranked[0].variant_id == "b"


def test_select_winners_per_word_class() -> None:
    rule = experiments.parse_selection_rule(SAMPLE)
    results = [
        _r("keube", 0.95, 0.2, wc="single_word"),
        _r("cube", 0.90, 0.2, wc="single_word"),
        _r("hey_keube", 0.97, 0.1, wc="two_word"),
    ]
    winners = experiments.select_winners(results, rule)
    assert winners["single_word"].variant_id == "keube"
    assert winners["two_word"].variant_id == "hey_keube"


def test_select_winners_none_when_all_fail() -> None:
    rule = experiments.parse_selection_rule(SAMPLE)
    results = [_r("keube", 0.50, 9.0, wc="single_word")]
    winners = experiments.select_winners(results, rule)
    assert winners["single_word"] is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
