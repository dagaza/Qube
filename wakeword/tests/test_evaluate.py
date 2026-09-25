"""Tests for evaluate.py orchestration (injected scorers; no audio/model)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import evaluate as ev  # noqa: E402
from lib import corpus as corpus_lib  # noqa: E402

CONFIG = {"wakeword": {"id": "hey_keube", "phrase": "hey_keube"}}


def _corpus() -> corpus_lib.Corpus:
    return corpus_lib.Corpus(
        corpus_version="test",
        root=Path("."),
        positives=[
            corpus_lib.ClipEntry(path=Path("p1.wav"), environment="quiet"),
            corpus_lib.ClipEntry(path=Path("p2.wav"), environment="noisy"),
        ],
        adversarial=[corpus_lib.ClipEntry(path=Path("a1.wav"))],
        negatives_longform=[corpus_lib.LongformEntry(path=Path("lf.wav"), duration_seconds=3600.0)],
    )


def test_normalize_handles_scalar_and_tuple() -> None:
    assert ev._normalize(0.7) == (0.7, None)
    assert ev._normalize((0.7, 120.0)) == (0.7, 120.0)


def test_peak_fires_refractory_dedup() -> None:
    scores = [0.0, 0.9, 0.85, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.95]
    fires = ev._peak_fires(scores, floor=0.5, refractory=5)
    assert fires == [0.9, 0.95]  # the 0.85 is inside the refractory window


def test_evaluate_corpus_pass(tmp_path: Path) -> None:
    scores = {"p1.wav": 0.95, "p2.wav": 0.9, "a1.wav": 0.1}

    def clip_scorer(entry: corpus_lib.ClipEntry):
        return (scores[entry.path.name], 150.0)

    def longform_scorer(entry: corpus_lib.LongformEntry):
        return [0.1, 0.2]  # no fires above typical thresholds

    result = ev.evaluate_corpus(
        CONFIG, _corpus(), clip_scorer=clip_scorer, longform_scorer=longform_scorer,
        version="v0.1", selection={"max_false_positives_per_hour": 1.0, "min_recall": 0.85},
    )
    assert result["verdict"] == "pass"
    assert result["recommended_threshold"] is not None
    assert result["robustness"]["quiet_recall"] == 1.0
    assert result["robustness"]["noisy_recall"] == 1.0
    assert result["latency_ms"]["p50"] == 150.0


def test_evaluate_corpus_fail_high_fp(tmp_path: Path) -> None:
    def clip_scorer(entry: corpus_lib.ClipEntry):
        return 0.95  # everything fires

    def longform_scorer(entry: corpus_lib.LongformEntry):
        return [0.99] * 50  # 50 fires / hour -> far above cap at every threshold

    result = ev.evaluate_corpus(
        CONFIG, _corpus(), clip_scorer=clip_scorer, longform_scorer=longform_scorer,
        selection={"max_false_positives_per_hour": 1.0, "min_recall": 0.85},
    )
    assert result["verdict"] == "fail"
    assert result["recommended_threshold"] is None
    assert result["verdict_reasons"]


def test_sweep_metric_entry_shape() -> None:
    def clip_scorer(entry):
        return (0.95, 150.0)

    result = ev.evaluate_corpus(
        CONFIG, _corpus(), clip_scorer=clip_scorer, longform_scorer=lambda e: [],
        selection={"max_false_positives_per_hour": 1.0, "min_recall": 0.85},
    )
    entry = ev.sweep_metric_entry(result)
    assert entry["variant_id"] == "hey_keube"
    assert set(entry) == {"variant_id", "recall", "false_positives_per_hour",
                          "noisy_room_robustness", "latency_ms"}


def test_write_report_and_append_sweep_metric(tmp_path: Path) -> None:
    result = ev.evaluate_corpus(
        CONFIG, _corpus(), clip_scorer=lambda e: 0.95, longform_scorer=lambda e: [],
        selection={"max_false_positives_per_hour": 1.0, "min_recall": 0.85},
    )
    report = ev.write_report(result, tmp_path / "eval.md")
    assert report.is_file()
    assert "Verdict" in report.read_text()

    metrics_path = tmp_path / "pilot_metrics.json"
    ev._append_sweep_metric(metrics_path, ev.sweep_metric_entry(result))
    ev._append_sweep_metric(metrics_path, ev.sweep_metric_entry(result))  # dedup by variant_id
    data = json.loads(metrics_path.read_text())
    assert len(data) == 1
    assert data[0]["variant_id"] == "hey_keube"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
