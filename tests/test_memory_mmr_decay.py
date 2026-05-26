"""Memory v7 MMR + temporal decay."""
import time

from core.memory_retrieval_policy import apply_mmr, temporal_decay_multiplier


def test_mmr_reduces_near_duplicate_top_results():
    items = [
        {"score": 0.9, "content": "User prefers dark roast coffee every morning."},
        {"score": 0.88, "content": "User prefers dark roast coffee each morning."},
        {"score": 0.5, "content": "User commutes by bicycle to work."},
    ]
    out = apply_mmr(items, top_k=2)
    assert len(out) == 2
    texts = [i["content"] for i in out]
    assert any("bicycle" in t for t in texts)


def test_mmr_normalizes_different_score_scales():
    items = [
        {"score": 100.0, "content": "Alpha topic one."},
        {"score": 99.0, "content": "Alpha topic one duplicate."},
        {"score": 1.0, "content": "Beta topic two."},
    ]
    out = apply_mmr(items, top_k=2)
    assert len(out) == 2
    assert any("Beta" in i["content"] for i in out)


def test_preference_tier_skips_temporal_decay():
    mult = temporal_decay_multiplier("preference", {"last_used_at": int(time.time()) - 86400 * 100})
    assert mult == 1.0


def test_context_tier_decays_with_age():
    mult = temporal_decay_multiplier(
        "context",
        {"last_used_at": int(time.time()) - 86400 * 30},
    )
    assert mult < 1.0
