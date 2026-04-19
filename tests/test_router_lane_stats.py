"""Tier 3 data-layer unit tests for ``mcp/router_lane_stats.py``.

Pure unit tests on the LaneStats / LaneStatsRegistry primitives;
no router import, no Qt, no embedder. These run BEFORE any
router integration so the data layer has its own strong baseline
and any later integration regression can be triaged locally.
"""
from __future__ import annotations

import os
import random
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mcp.router_lane_stats import (
    BETA_PRIOR,
    HIGH_RELIABILITY_BAND,
    LANE_BIAS_DAMPING,
    LANE_BIAS_RAW_CLAMP,
    LOW_RELIABILITY_BAND,
    MAX_ABS_OFFSET,
    MIN_OBSERVATIONS,
    RECENT_WINDOW_SIZE,
    LaneStats,
    LaneStatsRegistry,
    RouteFeedbackEvent,
)


# ----------------------------------------------------------------
# Event factory — central so test setup is uniform and intent
# stays readable.
# ----------------------------------------------------------------
def _evt(
    route: str = "memory",
    success: bool = True,
    *,
    top_intent: str | None = None,
    top_source: str = "substring",
    confidence_margin: float = 0.20,
    latency_ms: float = 50.0,
    drift: bool = False,
    per_lane_hits: dict | None = None,
) -> RouteFeedbackEvent:
    return RouteFeedbackEvent(
        route=route,
        top_intent=top_intent or (route if route != "hybrid" else "rag"),
        top_source=top_source,
        confidence_margin=confidence_margin,
        latency_ms=latency_ms,
        success=success,
        drift=drift,
        per_lane_hits=per_lane_hits or {},
    )


# ============================================================
# 1. Functional correctness
# ============================================================
class LaneStatsFunctionalTests(unittest.TestCase):

    def test_lane_stats_initial_state(self) -> None:
        reg = LaneStatsRegistry()
        for lane in ("memory", "rag", "web"):
            self.assertEqual(reg.adaptive_offset(lane), 0.0)
            self.assertIsNone(reg.recent_success_rate(lane))
            self.assertEqual(reg.embedding_trust(lane), 0.5)
            self.assertEqual(reg.substring_trust(lane), 0.5)
        self.assertFalse(reg.tier3_active())

    def test_update_credits_correct_lane_for_single_route(self) -> None:
        reg = LaneStatsRegistry()
        reg.update(_evt(route="memory", success=True))
        snap = reg.snapshot()
        self.assertEqual(snap["memory"]["total"], 1)
        self.assertEqual(snap["memory"]["success"], 1)
        self.assertEqual(snap["rag"]["total"], 0)
        self.assertEqual(snap["web"]["total"], 0)

    def test_update_credits_both_retrieval_lanes_for_hybrid(self) -> None:
        """Hybrid attribution: each retrieval lane gets credit /
        blame from its own per-lane hit count, not the route-level
        success blob. Web is never part of hybrid.
        """
        reg = LaneStatsRegistry()
        # Hybrid where memory returned data but rag did not.
        reg.update(_evt(
            route="hybrid",
            success=True,  # route-level: at least one lane fired
            per_lane_hits={"memory": 3, "rag": 0, "web": 0},
        ))
        snap = reg.snapshot()
        self.assertEqual(snap["memory"]["total"], 1)
        self.assertEqual(snap["memory"]["success"], 1)
        self.assertEqual(snap["rag"]["total"], 1)
        self.assertEqual(snap["rag"]["success"], 0)
        self.assertEqual(snap["web"]["total"], 0)

    def test_drift_event_is_skipped(self) -> None:
        reg = LaneStatsRegistry()
        reg.update(_evt(route="memory", success=True, drift=True))
        self.assertEqual(reg.snapshot()["memory"]["total"], 0)

    def test_none_route_is_skipped(self) -> None:
        reg = LaneStatsRegistry()
        reg.update(_evt(route="none", success=True))
        for lane in ("memory", "rag", "web"):
            self.assertEqual(reg.snapshot()[lane]["total"], 0)

    def test_unknown_route_is_skipped(self) -> None:
        """Defensive: an unknown route string must not crash and must
        not credit any lane."""
        reg = LaneStatsRegistry()
        reg.update(_evt(route="banana", success=True))
        for lane in ("memory", "rag", "web"):
            self.assertEqual(reg.snapshot()[lane]["total"], 0)

    def test_top_source_attribution_for_trust_counts(self) -> None:
        reg = LaneStatsRegistry()
        for _ in range(5):
            reg.update(_evt(route="memory", success=True, top_source="embedding"))
        for _ in range(5):
            reg.update(_evt(route="memory", success=True, top_source="substring"))
        # Look at the underlying LaneStats for the per-source totals.
        mem = reg._lanes["memory"]
        self.assertEqual(mem.embedding_total, 5)
        self.assertEqual(mem.embedding_success, 5)
        self.assertEqual(mem.substring_total, 5)
        self.assertEqual(mem.substring_success, 5)

    def test_recent_window_is_bounded(self) -> None:
        reg = LaneStatsRegistry()
        for i in range(200):
            reg.update(_evt(route="memory", success=(i % 2 == 0)))
        mem = reg._lanes["memory"]
        self.assertEqual(len(mem._recent), RECENT_WINDOW_SIZE)
        # Lifetime totals reflect all 200, not just the window.
        self.assertEqual(mem.total, 200)

    def test_tier3_active_flips_at_min_observations(self) -> None:
        reg = LaneStatsRegistry()
        for _ in range(MIN_OBSERVATIONS - 1):
            reg.update(_evt(route="memory", success=True))
        self.assertFalse(reg.tier3_active())
        reg.update(_evt(route="memory", success=True))
        self.assertTrue(reg.tier3_active())

    def test_to_dict_from_dict_round_trip(self) -> None:
        reg = LaneStatsRegistry()
        for _ in range(15):
            reg.update(_evt(route="memory", success=True, top_source="embedding"))
        for _ in range(7):
            reg.update(_evt(route="rag", success=False, top_source="substring"))
        reg.update(_evt(
            route="hybrid", success=True,
            per_lane_hits={"memory": 1, "rag": 1},
            top_source="embedding",
        ))

        round_tripped = LaneStatsRegistry.from_dict(reg.to_dict())

        for lane in ("memory", "rag", "web"):
            a = reg._lanes[lane]
            b = round_tripped._lanes[lane]
            self.assertEqual(a.total, b.total, f"lane={lane} total mismatch")
            self.assertEqual(a.success, b.success)
            self.assertEqual(a.embedding_total, b.embedding_total)
            self.assertEqual(a.embedding_success, b.embedding_success)
            self.assertEqual(a.substring_total, b.substring_total)
            self.assertEqual(a.substring_success, b.substring_success)
            self.assertAlmostEqual(a.confidence_sum, b.confidence_sum, places=6)
            self.assertEqual(list(a._recent), list(b._recent))
        # And the offsets / trust scores are identical too.
        for lane in ("memory", "rag", "web"):
            self.assertAlmostEqual(
                reg.adaptive_offset(lane),
                round_tripped.adaptive_offset(lane),
                places=9,
            )
            self.assertAlmostEqual(
                reg.embedding_trust(lane),
                round_tripped.embedding_trust(lane),
                places=9,
            )


# ============================================================
# 2. Stability — the bedrock guarantees
# ============================================================
class LaneStatsStabilityTests(unittest.TestCase):

    def test_adaptive_offset_strictly_bounded(self) -> None:
        """Random walk: 1000 events with random success / source.
        ``adaptive_offset`` must NEVER escape ``[-MAX_ABS_OFFSET,
        +MAX_ABS_OFFSET]`` even momentarily."""
        rng = random.Random(0xC0FFEE)
        reg = LaneStatsRegistry()
        for _ in range(1000):
            reg.update(_evt(
                route="memory",
                success=rng.random() < 0.5,
                top_source="embedding" if rng.random() < 0.5 else "substring",
            ))
            off = reg.adaptive_offset("memory")
            self.assertGreaterEqual(off, -MAX_ABS_OFFSET)
            self.assertLessEqual(off, +MAX_ABS_OFFSET)

    def test_repeated_identical_failures_converge(self) -> None:
        reg = LaneStatsRegistry()
        for _ in range(100):
            reg.update(_evt(route="memory", success=False))
        # success_rate = 0.0 → max positive offset.
        self.assertAlmostEqual(reg.adaptive_offset("memory"), +MAX_ABS_OFFSET, places=9)
        # Doing more failures must NOT push it any higher.
        for _ in range(100):
            reg.update(_evt(route="memory", success=False))
            self.assertLessEqual(reg.adaptive_offset("memory"), +MAX_ABS_OFFSET)

    def test_repeated_identical_successes_converge(self) -> None:
        reg = LaneStatsRegistry()
        for _ in range(100):
            reg.update(_evt(route="memory", success=True))
        self.assertAlmostEqual(reg.adaptive_offset("memory"), -MAX_ABS_OFFSET, places=9)
        for _ in range(100):
            reg.update(_evt(route="memory", success=True))
            self.assertGreaterEqual(reg.adaptive_offset("memory"), -MAX_ABS_OFFSET)

    def test_offset_returns_to_zero_after_window_reset(self) -> None:
        """50 failures then 50 successes — the bounded window now
        contains only successes, so the offset must be at the
        success extreme (no memory of the prior failures)."""
        reg = LaneStatsRegistry()
        for _ in range(RECENT_WINDOW_SIZE):
            reg.update(_evt(route="memory", success=False))
        self.assertAlmostEqual(reg.adaptive_offset("memory"), +MAX_ABS_OFFSET, places=9)
        for _ in range(RECENT_WINDOW_SIZE):
            reg.update(_evt(route="memory", success=True))
        self.assertAlmostEqual(reg.adaptive_offset("memory"), -MAX_ABS_OFFSET, places=9)

    def test_neutral_band_zero_offset(self) -> None:
        """Success rate strictly between 0.4 and 0.7 must yield 0.0."""
        reg = LaneStatsRegistry()
        # 25 successes + 25 failures = exactly 0.5 → neutral.
        for _ in range(25):
            reg.update(_evt(route="memory", success=True))
        for _ in range(25):
            reg.update(_evt(route="memory", success=False))
        sr = reg.recent_success_rate("memory")
        self.assertIsNotNone(sr)
        self.assertTrue(LOW_RELIABILITY_BAND < sr < HIGH_RELIABILITY_BAND, sr)
        self.assertEqual(reg.adaptive_offset("memory"), 0.0)

    def test_below_min_observations_offset_is_zero(self) -> None:
        reg = LaneStatsRegistry()
        for _ in range(MIN_OBSERVATIONS - 1):
            reg.update(_evt(route="memory", success=False))
        self.assertEqual(reg.adaptive_offset("memory"), 0.0)
        self.assertIsNone(reg.recent_success_rate("memory"))

    def test_offset_is_pure_read(self) -> None:
        """Calling ``adaptive_offset`` 100 times with no intervening
        ``update`` MUST return the exact same value every call —
        proves it does not mutate state."""
        reg = LaneStatsRegistry()
        for _ in range(20):
            reg.update(_evt(route="memory", success=False))
        first = reg.adaptive_offset("memory")
        for _ in range(100):
            self.assertEqual(reg.adaptive_offset("memory"), first)

    def test_unknown_lane_offset_is_zero(self) -> None:
        """Defensive — asking about a lane that doesn't exist must
        not crash and must yield 0.0."""
        reg = LaneStatsRegistry()
        self.assertEqual(reg.adaptive_offset("recall"), 0.0)
        self.assertEqual(reg.adaptive_offset("banana"), 0.0)

    def test_constants_match_design(self) -> None:
        """Pin the design constants — a future refactor that
        silently loosens any of them must fail CI."""
        self.assertEqual(RECENT_WINDOW_SIZE, 50)
        self.assertEqual(MIN_OBSERVATIONS, 10)
        self.assertAlmostEqual(MAX_ABS_OFFSET, 0.05, places=6)
        self.assertAlmostEqual(LOW_RELIABILITY_BAND, 0.4, places=6)
        self.assertAlmostEqual(HIGH_RELIABILITY_BAND, 0.7, places=6)
        self.assertAlmostEqual(BETA_PRIOR, 1.0, places=6)
        # Damped lane-bias constants consumed by the router. The
        # raw-clamp must always equal MAX_ABS_OFFSET so the data
        # layer and the router agree on the bound; damping is the
        # single tunable that controls the *applied* magnitude.
        self.assertAlmostEqual(LANE_BIAS_RAW_CLAMP, 0.05, places=6)
        self.assertAlmostEqual(LANE_BIAS_DAMPING, 0.6, places=6)
        self.assertAlmostEqual(
            LANE_BIAS_RAW_CLAMP, MAX_ABS_OFFSET, places=9,
            msg="LANE_BIAS_RAW_CLAMP must mirror MAX_ABS_OFFSET — "
                "they are the same physical bound.",
        )


# ============================================================
# 3. Trust scores
# ============================================================
class LaneTrustTests(unittest.TestCase):

    def test_trust_starts_at_half_with_beta_prior(self) -> None:
        reg = LaneStatsRegistry()
        for lane in ("memory", "rag", "web"):
            self.assertAlmostEqual(reg.embedding_trust(lane), 0.5, places=6)
            self.assertAlmostEqual(reg.substring_trust(lane), 0.5, places=6)

    def test_embedding_trust_reflects_per_source_success_rate(self) -> None:
        reg = LaneStatsRegistry()
        for _ in range(10):
            reg.update(_evt(route="memory", success=True, top_source="embedding"))
        for _ in range(10):
            reg.update(_evt(route="memory", success=False, top_source="substring"))
        emb = reg.embedding_trust("memory")
        sub = reg.substring_trust("memory")
        self.assertGreater(emb, sub)
        # With BETA_PRIOR=1.0: emb = (10+1)/(10+2) ≈ 0.917
        # sub = (0+1)/(10+2) ≈ 0.083
        self.assertAlmostEqual(emb, 11.0 / 12.0, places=6)
        self.assertAlmostEqual(sub, 1.0 / 12.0, places=6)

    def test_trust_handles_zero_observations_safely(self) -> None:
        """Beta-prior must protect against divide-by-zero on lanes
        that have only ever seen the OTHER source."""
        reg = LaneStatsRegistry()
        for _ in range(5):
            reg.update(_evt(route="memory", success=True, top_source="embedding"))
        # zero substring observations, but call must still return 0.5.
        self.assertAlmostEqual(reg.substring_trust("memory"), 0.5, places=6)


# ============================================================
# 4. LaneStats record() direct unit tests
# ============================================================
class LaneStatsRecordTests(unittest.TestCase):
    """Direct tests of LaneStats.record so we don't conflate any
    bugs with the registry-level attribution logic."""

    def test_record_increments_total_and_success(self) -> None:
        s = LaneStats()
        s.record(success=True, source="embedding", confidence_margin=0.1)
        self.assertEqual(s.total, 1)
        self.assertEqual(s.success, 1)
        self.assertEqual(s.embedding_total, 1)
        self.assertEqual(s.embedding_success, 1)
        self.assertEqual(s.substring_total, 0)

    def test_record_failure_does_not_increment_success(self) -> None:
        s = LaneStats()
        s.record(success=False, source="substring", confidence_margin=0.0)
        self.assertEqual(s.total, 1)
        self.assertEqual(s.success, 0)
        self.assertEqual(s.substring_total, 1)
        self.assertEqual(s.substring_success, 0)

    def test_record_unknown_source_still_updates_global_counters(self) -> None:
        s = LaneStats()
        s.record(success=True, source="meow", confidence_margin=0.0)
        self.assertEqual(s.total, 1)
        self.assertEqual(s.success, 1)
        self.assertEqual(s.embedding_total, 0)
        self.assertEqual(s.substring_total, 0)
        self.assertEqual(s.n_recent(), 1)


if __name__ == "__main__":
    unittest.main()
