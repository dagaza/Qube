"""Memory v7.1 promotion scoring."""
import time

from core.memory_promotion import (
    compute_promotion_score,
    consolidation_from_retrieval_days,
    get_promotion_thresholds,
    is_almost_promoted,
    passes_promotion_gates,
    passes_promotion_gates_with_reason,
    promotion_score_breakdown,
)


def _payload(**overrides):
    base = {
        "content": "The user prefers metric units for all measurements.",
        "provenance_quote": "I prefer metric units.",
        "times_retrieved": 5,
        "times_cited_positively": 5,
        "unique_query_count": 4,
        "retrieval_query_fps": ["a", "b", "c", "d"],
        "retrieval_days": ["2026-05-01", "2026-05-03", "2026-05-05"],
        "retrieval_score_sum": 3.6,
        "retrieval_score_count": 5,
        "timestamp": int(time.time()) - 86400,
        "last_used_at": int(time.time()) - 3600,
        "first_seen_at": int(time.time()) - 86400 * 3,
    }
    base.update(overrides)
    return base


def test_promotion_score_breakdown_has_weighted_rows():
    rows = promotion_score_breakdown(_payload())
    signals = {r["signal"] for r in rows}
    assert "relevance" in signals
    assert "consolidation" in signals
    assert "total" in signals


def test_promotion_gates_reject_low_recall():
    ok, reason = passes_promotion_gates(
        _payload(times_retrieved=1),
        "qube_memory::context::preference",
    )
    assert ok is False
    assert "times_retrieved" in reason


def test_promotion_gates_accept_strong_row():
    ok, _ = passes_promotion_gates(
        _payload(),
        "qube_memory::knowledge::knowledge",
    )
    assert ok is True
    assert compute_promotion_score(_payload()) >= 0.65


def test_consolidation_from_retrieval_days_multi_day():
    score_one = consolidation_from_retrieval_days(["2026-05-01"])
    score_multi = consolidation_from_retrieval_days(["2026-05-01", "2026-05-10"])
    assert score_one == 0.2
    assert score_multi > score_one


def test_passes_promotion_gates_with_reason_returns_components():
    ok, reason, components = passes_promotion_gates_with_reason(
        _payload(times_retrieved=1),
        "qube_memory::context::preference",
    )
    assert ok is False
    assert "times_retrieved" in reason
    assert "relevance" in components


def test_is_almost_promoted_when_gate_fails_but_score_high():
    payload = _payload(times_retrieved=1)
    assert is_almost_promoted(payload, "qube_memory::context::preference") is True


def test_presets_change_thresholds():
    conservative = get_promotion_thresholds("conservative")
    aggressive = get_promotion_thresholds("aggressive")
    assert conservative["min_score"] > aggressive["min_score"]
    assert conservative["min_retrieved"] > aggressive["min_retrieved"]
