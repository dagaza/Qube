"""Memory v7.1 consolidation scoring."""
from core.memory_consolidation import (
    build_consolidation_hints,
    compute_consolidation_score,
    should_stage_for_consolidation,
)


def test_consolidation_hints_multi_day():
    payload = {
        "retrieval_days": ["2026-05-01", "2026-05-02"],
        "times_retrieved": 4,
        "times_cited_positively": 2,
        "provenance_quote": "quoted",
    }
    hints = build_consolidation_hints(payload)
    assert "multi_day_retrieval" in hints
    assert "provenance_present" in hints


def test_should_stage_for_consolidation_context_row():
    payload = {
        "retrieval_days": ["2026-05-01", "2026-05-03"],
        "times_retrieved": 5,
        "times_cited_positively": 3,
        "provenance_quote": "user said this",
        "retrieval_score_sum": 3.0,
        "retrieval_score_count": 5,
    }
    ok, score, hints = should_stage_for_consolidation(
        payload, "qube_memory::context::preference"
    )
    assert ok is True
    assert score >= 0.55
    assert hints


def test_episode_rows_not_staged():
    ok, score, hints = should_stage_for_consolidation(
        {"category": "episode", "retrieval_days": ["2026-05-01", "2026-05-02"]},
        "qube_memory::episode::sess",
    )
    assert ok is False
    assert score == 0.0
    assert hints == []


def test_compute_consolidation_score_bounded():
    score = compute_consolidation_score({"retrieval_days": ["2026-05-01"]})
    assert 0.0 <= score <= 1.0
