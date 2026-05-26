"""Memory v7.1 usage drain merge helpers."""
import time

from core.memory_usage_drain import (
    apply_usage_deltas_to_payload,
    iso_day_from_timestamp,
    merge_retrieval_day,
)


def test_merge_retrieval_day_dedupes_and_caps():
    days = merge_retrieval_day(["2026-05-01"], "2026-05-01")
    assert days == ["2026-05-01"]
    days = merge_retrieval_day(days, "2026-05-02")
    assert days == ["2026-05-01", "2026-05-02"]


def test_apply_usage_deltas_tracks_scores_and_days():
    ts = time.time()
    payload = apply_usage_deltas_to_payload(
        {},
        retrieved=2,
        cited=1,
        query_fps=["abc", "def"],
        retrieval_scores=[0.5, 0.7],
        now_ts=ts,
    )
    assert payload["times_retrieved"] == 2
    assert payload["times_cited_positively"] == 1
    assert payload["unique_query_count"] == 2
    assert payload["retrieval_score_count"] == 2
    assert abs(payload["retrieval_score_sum"] - 1.2) < 1e-6
    assert payload["retrieval_days"] == [iso_day_from_timestamp(ts)]
