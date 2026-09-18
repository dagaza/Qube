"""Tests for evaluation metrics (lib/metrics.py) — pure, no audio/model."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from lib import metrics  # noqa: E402

POSITIVES = [
    {"score": 0.9, "environment": "quiet", "latency_ms": 100.0},
    {"score": 0.4, "environment": "noisy", "latency_ms": 200.0},
]
ADVERSARIAL = [{"score": 0.6}, {"score": 0.2}]
LONGFORM = [{"fire_scores": [0.7, 0.55, 0.3], "duration_seconds": 3600.0}]  # 1 hour


def test_percentile() -> None:
    assert metrics.percentile([], 50) == 0.0
    assert metrics.percentile([5.0], 95) == 5.0
    assert metrics.percentile([0.0, 10.0], 50) == 5.0
    assert metrics.percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)


def test_compute_threshold_metrics_at_half() -> None:
    m = metrics.compute_threshold_metrics(POSITIVES, ADVERSARIAL, LONGFORM, 0.5)
    assert m.recall == 0.5
    assert m.frr == 0.5
    assert m.adversarial_far == 0.5
    assert m.fp_per_hour == 2.0            # 2 fires over 1 hour
    assert m.precision == pytest.approx(0.25)  # tp=1 / (tp1 + adv1 + longform2)
    assert m.recall_quiet == 1.0
    assert m.recall_noisy == 0.0
    assert m.latency_ms_p50 == 100.0       # only the detected positive counts


def test_default_thresholds_span() -> None:
    ts = metrics.default_thresholds(0.3, 0.7, 0.05)
    assert ts[0] == 0.3
    assert ts[-1] == 0.7
    assert len(ts) == 9


def test_select_threshold_max_recall_under_fp_cap() -> None:
    swept = metrics.sweep(POSITIVES, ADVERSARIAL, LONGFORM, [0.5, 0.6, 0.72])
    # 0.5: fp/hr=2 (fails cap); 0.6: 1 fire -> fp/hr=1 recall .5; 0.72: 0 fires recall .5
    chosen = metrics.select_threshold(swept, max_fp_per_hour=1.0, min_recall=0.4)
    assert chosen in (0.6, 0.72)


def test_select_threshold_none_when_recall_floor_unmet() -> None:
    swept = metrics.sweep(POSITIVES, ADVERSARIAL, LONGFORM, [0.5, 0.6])
    assert metrics.select_threshold(swept, max_fp_per_hour=1.0, min_recall=0.99) is None


def test_select_threshold_ties_break_to_higher_threshold() -> None:
    # Two thresholds, identical passing recall + fp -> pick the higher (fewer false accepts).
    pos = [{"score": 0.95, "environment": "quiet"}]
    swept = metrics.sweep(pos, [], [], [0.5, 0.6])
    assert metrics.select_threshold(swept, max_fp_per_hour=1.0, min_recall=0.5) == 0.6


def test_roc_and_det_points_sorted_by_far() -> None:
    swept = metrics.sweep(POSITIVES, ADVERSARIAL, LONGFORM, [0.3, 0.5, 0.7])
    roc = metrics.roc_points(swept)
    det = metrics.det_points(swept)
    assert roc == sorted(roc)
    assert det == sorted(det)


def test_verdict_pass_and_fail() -> None:
    swept = metrics.sweep(POSITIVES, ADVERSARIAL, LONGFORM, metrics.default_thresholds())
    ok = metrics.verdict(swept, max_fp_per_hour=1.0, min_recall=0.4)
    assert ok.passed and ok.recommended_threshold is not None
    bad = metrics.verdict(swept, max_fp_per_hour=1.0, min_recall=0.99)
    assert not bad.passed and bad.reasons


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
