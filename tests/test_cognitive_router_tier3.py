"""Tier 3 integration tests for ``CognitiveRouterV4`` —
feedback-driven adaptive calibration layer.

Covers:
  * functional integration (observe_feedback → adaptive offset → applied lane bias),
  * the damped, band-gated application semantics
    (``MIN_CONFIDENCE_FLOOR <= top_score <= HIGH_CONFIDENCE_CEILING``),
  * stability under stacked offsets (sensitivity + latency + Tier 3),
  * the activation guard (zero feedback → bit-identical Tier 1+2 behavior),
  * decision-dict additive contract,
  * regression sample for Tier 1 + Tier 2 paths,
  * reversibility within the damped envelope.

We follow the same convention as ``tests/test_cognitive_router_margin.py``
and ``tests/test_cognitive_router_tier2.py`` — pure numpy synthetic
centroids + a feedback-event factory, no Qt / audio / embedder.
"""
from __future__ import annotations

import math
import os
import random
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mcp.cognitive_router import (
    HIGH_CONFIDENCE_CEILING,
    MIN_CONFIDENCE_FLOOR,
    CognitiveRouterV4,
)
from mcp.router_lane_stats import (
    LANE_BIAS_DAMPING,
    LANE_BIAS_RAW_CLAMP,
    MAX_ABS_OFFSET,
    MIN_OBSERVATIONS,
    RECENT_WINDOW_SIZE,
    RouteFeedbackEvent,
)


_NEUTRAL_QUERY = "the quick brown fox jumps over the lazy dog"
_DAMPED_BIAS = LANE_BIAS_RAW_CLAMP * LANE_BIAS_DAMPING  # == 0.03 by design


# Synthetic centroids in R^6 (mirror of test_cognitive_router_tier2.py
# helper but inlined to keep this file self-contained).
_MEM_AXIS = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
_RAG_AXIS = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
_WEB_AXIS = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _intent_vec(m: float = 0.0, r: float = 0.0, w: float = 0.0) -> np.ndarray:
    sq = m * m + r * r + w * w
    if sq > 1.0:
        raise ValueError(f"cosines exceed unit length: {(m, r, w)}")
    residual = math.sqrt(max(0.0, 1.0 - sq))
    v = np.array([m, r, w, 0.0, 0.0, residual], dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _install_tier2_centroids(router: CognitiveRouterV4) -> None:
    router.set_memory_centroid(_MEM_AXIS.copy())
    router.set_rag_centroid(_RAG_AXIS.copy())
    router.set_web_centroid(_WEB_AXIS.copy())


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


def _feed(router: CognitiveRouterV4, n: int, **kwargs) -> None:
    for _ in range(n):
        router.observe_feedback(_evt(**kwargs))


def _in_band_vec(amp: float = 0.5) -> np.ndarray:
    """Returns an intent vector whose cosine against ``_MEM_AXIS`` is
    ``amp``. With ``amp`` chosen inside ``[0.30, 0.75]`` the resulting
    ``top_score`` falls inside the Tier 3 decision-boundary band, so
    the damped lane bias is applied."""
    return _intent_vec(m=amp)


# ============================================================
# 1. Functional integration
# ============================================================
class Tier3FeedbackIntegrationTests(unittest.TestCase):

    def test_observe_feedback_is_no_op_before_min_observations(self) -> None:
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        _feed(router, MIN_OBSERVATIONS - 1, route="memory", success=True)
        d = router.route(_NEUTRAL_QUERY, intent_vector=_in_band_vec(0.5))
        # Underlying signal is dormant.
        self.assertEqual(router.lane_stats.adaptive_offset("memory"), 0.0)
        self.assertEqual(router.lane_stats.adaptive_offset("rag"), 0.0)
        self.assertEqual(router.lane_stats.adaptive_offset("web"), 0.0)
        # Decision-dict applied bias is therefore zero too,
        # regardless of band membership.
        self.assertEqual(d["memory_lane_bias"], 0.0)
        self.assertEqual(d["rag_lane_bias"], 0.0)
        self.assertEqual(d["web_lane_bias"], 0.0)
        self.assertIsNone(d["memory_lane_success_rate"])
        self.assertIsNone(d["rag_lane_success_rate"])
        self.assertIsNone(d["web_lane_success_rate"])
        self.assertFalse(d["tier3_active"])
        # ``tier3_band_active`` is a pure function of ``top_score``,
        # independent of feedback availability.
        self.assertTrue(d["tier3_band_active"])
        # And the threshold is still the base (no latency, no
        # sensitivity, and lane_bias is zero from undersample).
        self.assertAlmostEqual(d["memory_threshold"], router.base_memory_threshold, places=9)

    def test_lane_bias_in_band_lowers_threshold_by_damped_amount(self) -> None:
        """In-band, reliable lane: applied bias is
        ``-LANE_BIAS_RAW_CLAMP * LANE_BIAS_DAMPING`` (== ``-0.03``)."""
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        in_band = _in_band_vec(0.5)
        baseline = router.route(_NEUTRAL_QUERY, intent_vector=in_band)["memory_threshold"]
        _feed(router, 20, route="memory", success=True)
        d = router.route(_NEUTRAL_QUERY, intent_vector=in_band)
        self.assertAlmostEqual(d["memory_lane_bias"], -_DAMPED_BIAS, places=9)
        self.assertAlmostEqual(d["memory_threshold"], baseline - _DAMPED_BIAS, places=9)
        self.assertTrue(d["tier3_band_active"])
        self.assertTrue(d["tier3_active"])

    def test_lane_bias_in_band_raises_threshold_by_damped_amount(self) -> None:
        """In-band, unreliable lane: applied bias is ``+0.03``."""
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        in_band = _in_band_vec(0.5)
        baseline = router.route(_NEUTRAL_QUERY, intent_vector=in_band)["memory_threshold"]
        _feed(router, 20, route="memory", success=False)
        d = router.route(_NEUTRAL_QUERY, intent_vector=in_band)
        self.assertAlmostEqual(d["memory_lane_bias"], +_DAMPED_BIAS, places=9)
        self.assertAlmostEqual(d["memory_threshold"], baseline + _DAMPED_BIAS, places=9)
        self.assertTrue(d["tier3_band_active"])

    def test_other_lanes_unaffected_by_single_lane_feedback(self) -> None:
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        in_band = _in_band_vec(0.5)
        rag_baseline = router.route(_NEUTRAL_QUERY, intent_vector=in_band)["rag_threshold"]
        web_baseline = router.route(_NEUTRAL_QUERY, intent_vector=in_band)["internet_threshold"]
        _feed(router, 20, route="memory", success=True)
        d = router.route(_NEUTRAL_QUERY, intent_vector=in_band)
        self.assertAlmostEqual(d["rag_lane_bias"], 0.0)
        self.assertAlmostEqual(d["web_lane_bias"], 0.0)
        self.assertAlmostEqual(d["rag_threshold"], rag_baseline, places=9)
        self.assertAlmostEqual(d["internet_threshold"], web_baseline, places=9)

    def test_hybrid_route_credits_both_retrieval_lanes(self) -> None:
        """Hybrid where ONLY memory returned data: memory should get
        20 successes, rag should get 20 failures. Both lanes should
        therefore reach the bounded extreme in the underlying registry."""
        router = CognitiveRouterV4()
        for _ in range(20):
            router.observe_feedback(_evt(
                route="hybrid",
                success=True,
                per_lane_hits={"memory": 3, "rag": 0},
            ))
        # Underlying raw signal hits the bound in both directions.
        self.assertAlmostEqual(router.lane_stats.adaptive_offset("memory"), -MAX_ABS_OFFSET, places=9)
        self.assertAlmostEqual(router.lane_stats.adaptive_offset("rag"),    +MAX_ABS_OFFSET, places=9)

    def test_drift_feedback_is_no_op(self) -> None:
        router = CognitiveRouterV4()
        _feed(router, 100, route="memory", success=False, drift=True)
        d = router.route(_NEUTRAL_QUERY)
        self.assertEqual(router.lane_stats.adaptive_offset("memory"), 0.0)
        self.assertEqual(d["memory_lane_bias"], 0.0)
        self.assertFalse(d["tier3_active"])

    def test_none_route_feedback_is_no_op(self) -> None:
        router = CognitiveRouterV4()
        _feed(router, 100, route="none", success=True)
        d = router.route(_NEUTRAL_QUERY)
        self.assertEqual(router.lane_stats.adaptive_offset("memory"), 0.0)
        self.assertEqual(router.lane_stats.adaptive_offset("rag"), 0.0)
        self.assertEqual(router.lane_stats.adaptive_offset("web"), 0.0)
        self.assertEqual(d["memory_lane_bias"], 0.0)
        self.assertEqual(d["rag_lane_bias"], 0.0)
        self.assertEqual(d["web_lane_bias"], 0.0)
        self.assertFalse(d["tier3_active"])

    def test_observe_feedback_does_not_crash_on_none_event(self) -> None:
        """Defensive: ``observe_feedback(None)`` must NOT raise so a
        malformed worker call can't kill a user-facing turn."""
        router = CognitiveRouterV4()
        router.observe_feedback(None)  # type: ignore[arg-type]
        d = router.route(_NEUTRAL_QUERY)
        self.assertEqual(d["memory_lane_bias"], 0.0)


# ============================================================
# 2. Band-gate semantics — Tier 3 only nudges borderline decisions
# ============================================================
class Tier3BandGateTests(unittest.TestCase):

    def test_lane_bias_zero_when_top_score_below_floor(self) -> None:
        """Below-floor query: the underlying lane is unreliable, but
        Tier 3 must NOT nudge the threshold up — that would be a
        rescue of a clearly-weak signal."""
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        _feed(router, 20, route="memory", success=False)
        # NEUTRAL query with no intent_vector → top_score = 0.0 < 0.30.
        d = router.route(_NEUTRAL_QUERY)
        self.assertEqual(d["memory_lane_bias"], 0.0)
        self.assertEqual(d["rag_lane_bias"], 0.0)
        self.assertEqual(d["web_lane_bias"], 0.0)
        self.assertFalse(d["tier3_band_active"])
        # And memory_threshold is the base — the +0.05 raw offset is
        # NOT applied because the query is out of band.
        self.assertAlmostEqual(d["memory_threshold"], router.base_memory_threshold, places=9)

    def test_lane_bias_zero_when_top_score_above_ceiling(self) -> None:
        """Above-ceiling query: this is the critical 'Tier 3 cannot
        overpower a strong Tier 2 signal' invariant. A reliable lane
        with a strong embedding hit must not have its threshold
        nudged down (which would be redundant) or up."""
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        _feed(router, 20, route="memory", success=True)
        above = _intent_vec(m=0.85)  # cosine 0.85 > 0.75 ceiling
        d = router.route(_NEUTRAL_QUERY, intent_vector=above)
        self.assertEqual(d["memory_lane_bias"], 0.0)
        self.assertFalse(d["tier3_band_active"])
        self.assertAlmostEqual(d["memory_threshold"], router.base_memory_threshold, places=9)
        # Strong route still fires.
        self.assertEqual(d["route"], "memory")

    def test_lane_bias_zero_at_band_boundaries_inclusive(self) -> None:
        """Both boundaries are ``<=``-inclusive: exactly at
        ``MIN_CONFIDENCE_FLOOR`` and exactly at ``HIGH_CONFIDENCE_CEILING``
        the bias still applies. Pins the inclusivity choice against
        accidental ``<`` vs ``<=`` regressions."""
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        _feed(router, 20, route="memory", success=True)

        at_floor = _intent_vec(m=MIN_CONFIDENCE_FLOOR)
        d_lo = router.route(_NEUTRAL_QUERY, intent_vector=at_floor)
        self.assertTrue(d_lo["tier3_band_active"])
        self.assertAlmostEqual(d_lo["memory_lane_bias"], -_DAMPED_BIAS, places=9)

        at_ceiling = _intent_vec(m=HIGH_CONFIDENCE_CEILING)
        d_hi = router.route(_NEUTRAL_QUERY, intent_vector=at_ceiling)
        self.assertTrue(d_hi["tier3_band_active"])
        self.assertAlmostEqual(d_hi["memory_lane_bias"], -_DAMPED_BIAS, places=9)

    def test_tier3_band_active_field_reflects_top_score(self) -> None:
        """Round-trip three queries (below band / in band / above band)
        and assert ``tier3_band_active`` matches each region."""
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        _feed(router, 20, route="memory", success=True)

        below = _intent_vec(m=0.10)
        d_below = router.route(_NEUTRAL_QUERY, intent_vector=below)
        self.assertFalse(d_below["tier3_band_active"])
        self.assertEqual(d_below["memory_lane_bias"], 0.0)

        in_b = _in_band_vec(0.5)
        d_in = router.route(_NEUTRAL_QUERY, intent_vector=in_b)
        self.assertTrue(d_in["tier3_band_active"])
        self.assertAlmostEqual(d_in["memory_lane_bias"], -_DAMPED_BIAS, places=9)

        above = _intent_vec(m=0.90)
        d_above = router.route(_NEUTRAL_QUERY, intent_vector=above)
        self.assertFalse(d_above["tier3_band_active"])
        self.assertEqual(d_above["memory_lane_bias"], 0.0)


# ============================================================
# 3. Stability — clamps, base threshold, pure read, damped envelope
# ============================================================
class Tier3StabilityTests(unittest.TestCase):

    def test_threshold_clamped_within_lo_hi_after_offset_stack(self) -> None:
        """Most adverse possible setup: maximally adverse adaptive
        offset (lane bias = +0.03 when in band) + low sensitivity
        (raises threshold) + slow latency (+0.05). The router's
        existing per-lane ``hi`` clamp must STILL bound the result.
        """
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        # 6 slow turns engage +0.05 latency offset.
        for _ in range(6):
            router.record_latency(3.0)
        # 20 failures → +0.03 damped lane bias on memory in-band.
        _feed(router, 20, route="memory", success=False)
        # Low sensitivity raises threshold further.
        d = router.route(
            _NEUTRAL_QUERY,
            intent_vector=_in_band_vec(0.5),
            weights={"memory_sensitivity": 0.5},
        )
        # memory `hi` clamp = 0.75 in cognitive_router source.
        self.assertLessEqual(d["memory_threshold"], 0.75 + 1e-9)
        # And the bias really is the damped envelope, not the raw clamp.
        self.assertAlmostEqual(d["memory_lane_bias"], +_DAMPED_BIAS, places=9)

    def test_lane_bias_never_exceeds_damped_envelope(self) -> None:
        """1000 random feedback events + random in-band/out-of-band
        queries: applied bias never exceeds
        ``LANE_BIAS_RAW_CLAMP * LANE_BIAS_DAMPING``."""
        rng = random.Random(0xC0DE)
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        bound = _DAMPED_BIAS + 1e-9

        for _ in range(1000):
            lane = rng.choice(("memory", "rag", "web"))
            success = rng.random() < 0.5
            router.observe_feedback(_evt(route=lane, success=success))
            amp = rng.uniform(0.0, 0.99)
            which = rng.choice(("m", "r", "w"))
            v = _intent_vec(**{which: amp})
            d = router.route(_NEUTRAL_QUERY, intent_vector=v)
            self.assertLessEqual(abs(d["memory_lane_bias"]), bound)
            self.assertLessEqual(abs(d["rag_lane_bias"]),    bound)
            self.assertLessEqual(abs(d["web_lane_bias"]),    bound)

    def test_base_thresholds_never_mutated_by_observe_feedback(self) -> None:
        router = CognitiveRouterV4()
        snap = (
            router.base_rag_threshold,
            router.base_memory_threshold,
            router.base_internet_threshold,
        )
        # Run a varied sequence of feedbacks; some success, some fail,
        # mixed lanes, hybrid attribution, drift, none.
        seq = [
            _evt(route="memory", success=True),
            _evt(route="memory", success=False),
            _evt(route="rag",    success=True,  top_source="embedding"),
            _evt(route="rag",    success=False, top_source="embedding"),
            _evt(route="web",    success=True),
            _evt(route="hybrid", success=True, per_lane_hits={"memory": 1, "rag": 0}),
            _evt(route="hybrid", success=True, per_lane_hits={"memory": 0, "rag": 2}),
            _evt(route="memory", success=False, drift=True),
            _evt(route="none",   success=True),
        ]
        for _ in range(20):
            for e in seq:
                router.observe_feedback(e)
        self.assertEqual(
            (
                router.base_rag_threshold,
                router.base_memory_threshold,
                router.base_internet_threshold,
            ),
            snap,
        )

    def test_threshold_does_not_compound_across_turns_with_tier3(self) -> None:
        """Tier-1 stability invariant must still hold with Tier 3 in
        the mix: identical inputs → identical thresholds, no drift."""
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        _feed(router, 20, route="memory", success=True)
        in_band = _in_band_vec(0.5)
        weights = {"memory_sensitivity": 1.5}
        first = router.route(
            _NEUTRAL_QUERY, intent_vector=in_band, weights=weights,
        )["memory_threshold"]
        for _ in range(10):
            d = router.route(
                _NEUTRAL_QUERY, intent_vector=in_band, weights=weights,
            )
            self.assertAlmostEqual(d["memory_threshold"], first, places=9)

    def test_lazy_threshold_delta_when_no_feedback_calls(self) -> None:
        """Calling ``route(...)`` 100 times without intervening
        ``observe_feedback`` must yield identical thresholds and
        identical applied bias values — the Tier 3 layer is a pure
        per-turn read with no mutable state on the router itself."""
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        _feed(router, 30, route="memory", success=False)
        in_band = _in_band_vec(0.5)
        first = router.route(_NEUTRAL_QUERY, intent_vector=in_band)
        for _ in range(99):
            d = router.route(_NEUTRAL_QUERY, intent_vector=in_band)
            self.assertAlmostEqual(
                d["memory_lane_bias"], first["memory_lane_bias"], places=9,
            )
            self.assertAlmostEqual(
                d["memory_threshold"], first["memory_threshold"], places=9,
            )


# ============================================================
# 4. Decision-dict additive contract
# ============================================================
class DecisionDictTier3ContractTests(unittest.TestCase):

    _TIER1_KEYS = (
        "route", "drift",
        "memory_score", "rag_score", "web_score",
        "recall_score", "recall_active", "chat_score",
        "recall_margin_over_chat",
        "rag_threshold", "memory_threshold", "internet_threshold",
        "recall_threshold",
        "strategy",
    )
    _TIER2_KEYS = (
        "memory_score_embedding", "rag_score_embedding", "web_score_embedding",
        "memory_score_final", "rag_score_final", "web_score_final",
        "memory_score_source", "rag_score_source", "web_score_source",
        "top_intent", "top_intent_source",
        "top_score", "second_best_score", "confidence_margin",
        "min_confidence_floor", "ambiguity_margin",
        "tier2_active",
    )
    _TIER3_KEYS = (
        "memory_lane_bias", "rag_lane_bias", "web_lane_bias",
        "memory_lane_success_rate", "rag_lane_success_rate", "web_lane_success_rate",
        "memory_embedding_trust", "rag_embedding_trust", "web_embedding_trust",
        "memory_substring_trust", "rag_substring_trust", "web_substring_trust",
        "tier3_active", "tier3_band_active",
        "tier3_high_confidence_ceiling", "tier3_damping",
    )

    def test_all_tier1_keys_still_present(self) -> None:
        d = CognitiveRouterV4().route(_NEUTRAL_QUERY)
        for k in self._TIER1_KEYS:
            self.assertIn(k, d, f"Tier-1 key {k!r} missing — additive contract broken.")

    def test_all_tier2_keys_still_present(self) -> None:
        d = CognitiveRouterV4().route(_NEUTRAL_QUERY)
        for k in self._TIER2_KEYS:
            self.assertIn(k, d, f"Tier-2 key {k!r} missing — additive contract broken.")

    def test_all_tier3_keys_present(self) -> None:
        d = CognitiveRouterV4().route(_NEUTRAL_QUERY)
        for k in self._TIER3_KEYS:
            self.assertIn(k, d, f"Tier-3 key {k!r} missing.")

    def test_tier3_active_flag_reflects_observation_count(self) -> None:
        router = CognitiveRouterV4()
        d = router.route(_NEUTRAL_QUERY)
        self.assertFalse(d["tier3_active"])
        _feed(router, MIN_OBSERVATIONS, route="memory", success=True)
        d = router.route(_NEUTRAL_QUERY)
        self.assertTrue(d["tier3_active"])

    def test_lane_success_rate_is_none_when_undersampled(self) -> None:
        router = CognitiveRouterV4()
        _feed(router, MIN_OBSERVATIONS - 1, route="memory", success=True)
        d = router.route(_NEUTRAL_QUERY)
        self.assertIsNone(d["memory_lane_success_rate"])

    def test_lane_success_rate_is_float_when_above_min(self) -> None:
        router = CognitiveRouterV4()
        _feed(router, 20, route="memory", success=True)
        d = router.route(_NEUTRAL_QUERY)
        self.assertIsInstance(d["memory_lane_success_rate"], float)
        self.assertAlmostEqual(d["memory_lane_success_rate"], 1.0, places=6)

    def test_trust_starts_at_half(self) -> None:
        d = CognitiveRouterV4().route(_NEUTRAL_QUERY)
        for k in (
            "memory_embedding_trust", "rag_embedding_trust", "web_embedding_trust",
            "memory_substring_trust", "rag_substring_trust", "web_substring_trust",
        ):
            self.assertAlmostEqual(d[k], 0.5, places=6, msg=f"{k} != 0.5 at boot")

    def test_lane_bias_in_decision_matches_damped_offset_when_in_band(self) -> None:
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        _feed(router, 20, route="memory", success=False)
        d = router.route(_NEUTRAL_QUERY, intent_vector=_in_band_vec(0.5))
        self.assertAlmostEqual(
            d["memory_lane_bias"],
            router.lane_stats.adaptive_offset("memory") * LANE_BIAS_DAMPING,
            places=9,
        )

    def test_high_confidence_ceiling_and_damping_match_constants(self) -> None:
        d = CognitiveRouterV4().route(_NEUTRAL_QUERY)
        self.assertAlmostEqual(d["tier3_high_confidence_ceiling"], HIGH_CONFIDENCE_CEILING, places=9)
        self.assertAlmostEqual(d["tier3_damping"], LANE_BIAS_DAMPING, places=9)


# ============================================================
# 5. Regression — Tier 1 + Tier 2 unchanged when no feedback exists
# ============================================================
class Tier3DormantRegressionTests(unittest.TestCase):
    """A fresh router with zero feedback events MUST behave
    bit-identically to the pre-Tier-3 router for every Tier 1 and
    Tier 2 scenario. We sample one canonical case from each layer."""

    def test_tier1_substring_routing_unchanged(self) -> None:
        router = CognitiveRouterV4()
        d = router.route("according to my notes")
        # 2/7 ≈ 0.286 > 0.25 base; rag must fire.
        self.assertGreater(d["rag_score"], router.base_rag_threshold)
        self.assertIn(d["route"], ("rag", "hybrid"))
        # And Tier 3 contributes nothing.
        self.assertFalse(d["tier3_active"])
        self.assertEqual(d["rag_lane_bias"], 0.0)

    def test_tier1_two_keyword_memory_query_still_fires(self) -> None:
        router = CognitiveRouterV4()
        d = router.route("remember, did i ever like coffee?")
        self.assertIn(d["route"], ("memory", "hybrid"))
        self.assertEqual(d["memory_lane_bias"], 0.0)

    def test_tier2_embedding_route_unchanged(self) -> None:
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        v = _intent_vec(m=0.6)
        d = router.route(_NEUTRAL_QUERY, intent_vector=v)
        self.assertEqual(d["route"], "memory")
        self.assertEqual(d["memory_score_source"], "embedding")
        # Tier 3 still dormant — lane_bias is zero with no feedback.
        self.assertEqual(d["memory_lane_bias"], 0.0)

    def test_tier2_floor_downgrade_unchanged(self) -> None:
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        v = _intent_vec(m=0.27)  # weak embedding-only signal, below floor
        d = router.route(_NEUTRAL_QUERY, intent_vector=v)
        self.assertEqual(d["route"], "none")
        self.assertEqual(d["memory_lane_bias"], 0.0)


# ============================================================
# 6. Reversibility — bidirectional correction within damped envelope
# ============================================================
class Tier3ReversibilityTests(unittest.TestCase):
    """The bounded recency window means 50 successes after 50 failures
    flip the applied lane_bias to the success extreme. End-to-end
    'no permanent drift' under the damped envelope."""

    def test_offset_flips_with_window_after_trend_reverses(self) -> None:
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        in_band = _in_band_vec(0.5)

        # Phase A: full window of failures → +0.03 applied bias.
        _feed(router, RECENT_WINDOW_SIZE, route="memory", success=False)
        d_a = router.route(_NEUTRAL_QUERY, intent_vector=in_band)
        self.assertAlmostEqual(d_a["memory_lane_bias"], +_DAMPED_BIAS, places=9)

        # Phase B: full window of successes flushes phase A → -0.03.
        _feed(router, RECENT_WINDOW_SIZE, route="memory", success=True)
        d_b = router.route(_NEUTRAL_QUERY, intent_vector=in_band)
        self.assertAlmostEqual(d_b["memory_lane_bias"], -_DAMPED_BIAS, places=9)

        # And the threshold tracks the bias symmetrically (within the
        # damped envelope, NOT the raw clamp).
        self.assertAlmostEqual(
            d_b["memory_threshold"] - d_a["memory_threshold"],
            -2 * _DAMPED_BIAS,
            places=9,
        )


if __name__ == "__main__":
    unittest.main()
