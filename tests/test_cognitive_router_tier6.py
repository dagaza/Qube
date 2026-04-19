"""Tier 6 integration tests for ``CognitiveRouterV4`` —
Routing Arbitration Layer (RAL), v1 (observability-only).

Pins three guarantees, mirroring the Tier 5 layout:

  * **Decision-dict additive contract** — four new ``tier6_*`` keys
    are present with the documented types in every decision dict.
  * **Layer activates regardless of intent_vector** — like Tier 5,
    Tier 6 always runs; the derived signals just go neutral when
    Tier 4 is dormant. So a no-intent-vector turn returns
    ``tier6_active=True`` and ``tier6_conflict_flags=[]``.
  * **Route field never modified** — for a randomized 200-query walk
    a router with the live arbitration layer produces byte-identical
    ``route`` values to a control router whose layer is replaced
    with a no-op that always returns the empty / stable decision.

Plus a cross-tier consistency smoke test that drives three crafted
turns through a real router and asserts the conflict flags + the
interpretation enum.
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

from mcp.cognitive_router import CognitiveRouterV4
from mcp.router_lane_stats import RECENT_WINDOW_SIZE, RouteFeedbackEvent
from mcp.routing_arbitration_layer import (
    CONFLICT_ADAPTIVE_CONFLICT,
    CONFLICT_FLAGS_VALUES,
    CONFLICT_STABILITY_OVERRIDE,
    CONFLICT_STRUCTURAL_INSTABILITY,
    INTERPRETATION_ADAPTIVE_PRESSURE,
    INTERPRETATION_PASSTHROUGH,
    INTERPRETATION_STABLE,
    INTERPRETATION_STRUCTURAL_UNSTABLE,
    INTERPRETATION_VALUES,
    RoutingArbitrationDecision,
)
from mcp.routing_policy_engine import (
    POLICY_ACCEPT,
    POLICY_STABILIZE,
    POLICY_SUPPRESS_FLIP,
)


_NEUTRAL_QUERY = "the quick brown fox jumps over the lazy dog"


# Synthetic centroids in R^6 (mirror of Tier 2/3/4/5 tests).
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


def _evt(route: str = "memory", success: bool = True) -> RouteFeedbackEvent:
    return RouteFeedbackEvent(
        route=route,
        top_intent=route if route != "hybrid" else "rag",
        top_source="substring",
        confidence_margin=0.20,
        latency_ms=50.0,
        success=success,
        drift=False,
        per_lane_hits={},
    )


class _NullArbitrationLayer:
    """Drop-in stub matching ``RoutingArbitrationLayer.evaluate`` —
    always returns the empty / stable decision. Used by the
    ``Tier6RouteFieldNeverModified`` control router."""

    def evaluate(self, _inputs) -> RoutingArbitrationDecision:  # noqa: D401
        return RoutingArbitrationDecision((), INTERPRETATION_STABLE, {})


# ============================================================
# 1. Decision-dict additive contract
# ============================================================
class Tier6ObservabilityContract(unittest.TestCase):

    _TIER6_KEYS = (
        "tier6_active",
        "tier6_conflict_flags",
        "tier6_interpretation",
        "tier6_inputs_snapshot",
    )

    def test_all_tier6_keys_present_without_intent_vector(self) -> None:
        d = CognitiveRouterV4().route(_NEUTRAL_QUERY)
        for k in self._TIER6_KEYS:
            self.assertIn(k, d)
        self.assertTrue(d["tier6_active"])
        self.assertEqual(d["tier6_conflict_flags"], [])
        self.assertEqual(d["tier6_interpretation"], INTERPRETATION_STABLE)

    def test_all_tier6_keys_present_with_intent_vector(self) -> None:
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        d = router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(m=0.5))
        for k in self._TIER6_KEYS:
            self.assertIn(k, d)

    def test_field_types_match_documented_contract(self) -> None:
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        d = router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(m=0.5))
        self.assertIsInstance(d["tier6_active"], bool)
        self.assertIsInstance(d["tier6_conflict_flags"], list)
        for flag in d["tier6_conflict_flags"]:
            self.assertIn(flag, CONFLICT_FLAGS_VALUES)
        self.assertIsInstance(d["tier6_interpretation"], str)
        self.assertIn(d["tier6_interpretation"], INTERPRETATION_VALUES)
        self.assertIsInstance(d["tier6_inputs_snapshot"], dict)

    def test_inputs_snapshot_carries_expected_keys(self) -> None:
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        d = router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(m=0.5))
        snap = d["tier6_inputs_snapshot"]
        for k in ("stability_score", "oscillation_index",
                  "lane_bias_max_abs", "confidence_margin",
                  "tier5_policy"):
            self.assertIn(k, snap, f"snapshot key {k!r} missing")


# ============================================================
# 2. Layer always activates (mirrors Tier 5 contract)
# ============================================================
class Tier6AlwaysActiveTests(unittest.TestCase):

    def test_layer_active_without_intent_vector(self) -> None:
        d = CognitiveRouterV4().route(_NEUTRAL_QUERY)
        self.assertTrue(d["tier6_active"])
        self.assertEqual(d["tier6_conflict_flags"], [])
        self.assertEqual(d["tier6_interpretation"], INTERPRETATION_STABLE)
        self.assertEqual(d["tier6_inputs_snapshot"]["stability_score"], 0.0)
        self.assertEqual(d["tier6_inputs_snapshot"]["oscillation_index"], 0.0)

    def test_layer_active_when_tier4_below_min_observations(self) -> None:
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        d = router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(m=0.5))
        self.assertTrue(d["tier6_active"])
        self.assertEqual(d["tier6_inputs_snapshot"]["stability_score"], 0.0)


# ============================================================
# 3. Schema additivity (one canonical key per prior tier)
# ============================================================
class Tier6SchemaAdditivity(unittest.TestCase):

    def test_tier1_through_5_keys_still_present(self) -> None:
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        d = router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(m=0.5))
        for k in (
            "route",                # Tier 1
            "memory_score_final",   # Tier 2
            "memory_lane_bias",     # Tier 3
            "tier3_band_active",    # Tier 3 refinement
            "tier4_active",         # Tier 4
            "tier4_cluster_id",     # Tier 4
            "tier5_active",         # Tier 5
            "tier5_policy",         # Tier 5
            "tier6_active",         # Tier 6 (this tier)
            "tier6_conflict_flags", # Tier 6 (this tier)
        ):
            self.assertIn(k, d)


# ============================================================
# 4. Cross-tier consistency smoke test
# ============================================================
class Tier6CrossTierConsistencyTests(unittest.TestCase):
    """Drive three crafted turns through a real router and assert
    the corresponding ``tier6_conflict_flags`` + interpretation."""

    def _new_router(self) -> CognitiveRouterV4:
        r = CognitiveRouterV4()
        _install_tier2_centroids(r)
        return r

    def test_neutral_turn_yields_stable_with_no_flags(self) -> None:
        router = self._new_router()
        d = router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(m=0.5))
        self.assertEqual(d["tier5_policy"], POLICY_ACCEPT)
        self.assertEqual(d["tier6_conflict_flags"], [])
        self.assertEqual(d["tier6_interpretation"], INTERPRETATION_STABLE)

    def test_oscillating_cluster_yields_structural_unstable(self) -> None:
        """Pre-cook a 3-way oscillating cluster + tied scores. With
        3 routes cycled the dominant_frequency drops below 0.50, so
        Tier 6's stricter oscillation gate (> 0.50) fires alongside
        Tier 5's ``suppress_flip``. Both flags fire; interpretation
        collapses to ``structural_unstable`` per the precedence in
        the layer's ``_interpret``."""
        router = self._new_router()
        v = _intent_vec(m=0.5)
        # 3-way cycle: dominant_frequency ~ 0.33 -> oscillation ~ 0.67
        for r in ["memory", "rag", "web"] * 6:
            router.stability_tracker.observe(v, r)
        v_close = _intent_vec(m=0.27, r=0.27)
        d = router.route(_NEUTRAL_QUERY, intent_vector=v_close)
        # Tier 5 prerequisite: oscillation_index > 0.40.
        self.assertEqual(d["tier5_policy"], POLICY_SUPPRESS_FLIP)
        # Tier 6 sanity: oscillation > 0.50, margin < 0.10.
        self.assertGreater(d["tier6_inputs_snapshot"]["oscillation_index"], 0.50)
        self.assertLess(d["confidence_margin"], 0.10)
        # Both flags must fire.
        self.assertIn(CONFLICT_STABILITY_OVERRIDE, d["tier6_conflict_flags"])
        self.assertIn(CONFLICT_STRUCTURAL_INSTABILITY, d["tier6_conflict_flags"])
        # Structural wins the interpretation.
        self.assertEqual(d["tier6_interpretation"],
                         INTERPRETATION_STRUCTURAL_UNSTABLE)

    def test_stabilize_turn_yields_adaptive_pressure(self) -> None:
        """Saturate memory feedback so the damped lane bias > 0.02 and
        Tier 5 emits ``stabilize``. Tier 6 must fire
        ``adaptive_conflict`` -> interpretation ``adaptive_pressure``."""
        router = self._new_router()
        for _ in range(RECENT_WINDOW_SIZE):
            router.observe_feedback(_evt(route="memory", success=True))
        v_close = _intent_vec(m=0.5, r=0.5)
        d = router.route(_NEUTRAL_QUERY, intent_vector=v_close)
        # Tier 5 prerequisite
        self.assertEqual(d["tier5_policy"], POLICY_STABILIZE)
        # Tier 6
        self.assertIn(CONFLICT_ADAPTIVE_CONFLICT, d["tier6_conflict_flags"])
        # No structural instability (Tier 4 dormant).
        self.assertNotIn(CONFLICT_STRUCTURAL_INSTABILITY,
                         d["tier6_conflict_flags"])
        self.assertEqual(d["tier6_interpretation"],
                         INTERPRETATION_ADAPTIVE_PRESSURE)


# ============================================================
# 5. Route-field never modified — the canonical v1 invariant
# ============================================================
class Tier6RouteFieldNeverModified(unittest.TestCase):
    """200-query randomized walk against two routers — one with the
    live arbitration layer, one with a ``_NullArbitrationLayer`` —
    must produce byte-identical ``route`` values for every turn."""

    def _make_live(self) -> CognitiveRouterV4:
        r = CognitiveRouterV4()
        _install_tier2_centroids(r)
        return r

    def _make_control(self) -> CognitiveRouterV4:
        r = CognitiveRouterV4()
        _install_tier2_centroids(r)
        r.arbitration_layer = _NullArbitrationLayer()  # type: ignore[assignment]
        return r

    def test_random_walk_routes_byte_identical_against_null_layer(self) -> None:
        rng = random.Random(0xC0FFEE)
        live = self._make_live()
        ctrl = self._make_control()

        amp_choices = (0.0, 0.10, 0.27, 0.40, 0.50, 0.65, 0.80, 0.92)
        axis_choices = ("m", "r", "w", None)

        for _ in range(200):
            axis = rng.choice(axis_choices)
            if axis is None:
                v = None
            else:
                amp = rng.choice(amp_choices)
                v = _intent_vec(**{axis: amp})
            d_live = live.route(_NEUTRAL_QUERY, intent_vector=v)
            d_ctrl = ctrl.route(_NEUTRAL_QUERY, intent_vector=v)
            self.assertEqual(
                d_live["route"], d_ctrl["route"],
                msg=f"route diverged: live={d_live['route']} ctrl={d_ctrl['route']} (v={v})",
            )

    def test_route_field_unchanged_when_conflict_flags_fire(self) -> None:
        """Even when the layer reports the strongest interpretation,
        v1 must leave ``route`` alone. Pre-cook a 3-way oscillation so
        Tier 6 fires ``structural_instability`` (the strongest
        interpretation)."""
        live = self._make_live()
        ctrl = self._make_control()
        v = _intent_vec(m=0.27, r=0.27)
        for r in ["memory", "rag", "web"] * 6:
            live.stability_tracker.observe(v, r)
            ctrl.stability_tracker.observe(v, r)
        d_live = live.route(_NEUTRAL_QUERY, intent_vector=v)
        d_ctrl = ctrl.route(_NEUTRAL_QUERY, intent_vector=v)
        self.assertEqual(d_live["tier6_interpretation"],
                         INTERPRETATION_STRUCTURAL_UNSTABLE)
        self.assertEqual(d_ctrl["tier6_interpretation"], INTERPRETATION_STABLE)
        self.assertEqual(d_live["route"], d_ctrl["route"])


# ============================================================
# 6. Defensive — layer exception is swallowed
# ============================================================
class Tier6DefensiveTests(unittest.TestCase):

    def test_layer_exception_does_not_crash_route(self) -> None:
        class _Boom:
            def evaluate(self, *_args, **_kwargs):
                raise RuntimeError("synthetic layer failure")

        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        router.arbitration_layer = _Boom()  # type: ignore[assignment]
        with self.assertLogs("Qube.CognitiveRouterV4", level="WARNING") as cm:
            d = router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(m=0.5))
        self.assertIn("tier6 evaluate raised", "\n".join(cm.output))
        self.assertFalse(d["tier6_active"])
        self.assertEqual(d["tier6_conflict_flags"], [])
        self.assertEqual(d["tier6_interpretation"], INTERPRETATION_PASSTHROUGH)
        self.assertEqual(d["tier6_inputs_snapshot"], {})
        # Rest of the decision dict is fully populated as usual.
        self.assertIn("route", d)
        self.assertIn("tier5_policy", d)


# ============================================================
# 7. Dormant-regression — varied stream stays stable
# ============================================================
class Tier6DormantRegressionTests(unittest.TestCase):
    """Without any feedback (Tier 3 dormant) and without any
    pre-cooked Tier 4 oscillation, a varied query stream MUST
    produce ``tier6_conflict_flags == []`` for every turn.
    Confirms the layer adds no spurious flags."""

    def test_varied_stream_yields_no_flags_throughout(self) -> None:
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        rng = random.Random(0xB16B00B5)
        amps_per_axis = {"m": (0.50, 0.65), "r": (0.55, 0.70), "w": (0.60, 0.75)}
        for _ in range(50):
            axis = rng.choice(("m", "r", "w"))
            amp = rng.choice(amps_per_axis[axis])
            v = _intent_vec(**{axis: amp})
            d = router.route(_NEUTRAL_QUERY, intent_vector=v)
            self.assertEqual(
                d["tier6_conflict_flags"], [],
                msg=f"unexpected flags {d['tier6_conflict_flags']} on axis={axis} amp={amp}",
            )
            self.assertEqual(d["tier6_interpretation"], INTERPRETATION_STABLE)


if __name__ == "__main__":
    unittest.main()
