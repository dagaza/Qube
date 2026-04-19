"""Tier 5 integration tests for ``CognitiveRouterV4`` —
Routing Control Policy Layer (RCPL), v1 (observability-only).

Pins three guarantees:

  * **Decision-dict additive contract** — four new ``tier5_*`` keys
    are present with the documented types in every decision dict.
  * **Engine activates regardless of intent_vector** — unlike Tier 4,
    Tier 5 always runs; just the derived signals go neutral when
    Tier 4 is dormant. So a no-intent-vector turn returns
    ``tier5_active=True`` and ``tier5_policy=POLICY_ACCEPT``.
  * **Route field never modified** — for a randomized 200-query walk
    a router with the live engine produces byte-identical ``route``
    values to a control router whose engine is replaced with a no-op
    that always returns ``accept``.

We also drive the user-spec integration smoke test that crafts three
turns (neutral, stabilize-trigger, suppress-flip-trigger) on a real
router and asserts the corresponding ``tier5_policy`` values.
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
from mcp.router_lane_stats import (
    MIN_OBSERVATIONS,
    RECENT_WINDOW_SIZE,
    RouteFeedbackEvent,
)
from mcp.routing_policy_engine import (
    POLICY_ACCEPT,
    POLICY_NO_ACTION,
    POLICY_STABILIZE,
    POLICY_SUPPRESS_FLIP,
    POLICY_VALUES,
    RoutingPolicyDecision,
)


_NEUTRAL_QUERY = "the quick brown fox jumps over the lazy dog"


# Synthetic centroids in R^6 (mirror of Tier 2 / Tier 3 / Tier 4 tests).
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


class _NullPolicyEngine:
    """Drop-in stub matching ``RoutingPolicyEngine.evaluate`` —
    always returns ``POLICY_ACCEPT``. Used by the
    ``Tier5RouteFieldNeverModified`` control router."""

    def evaluate(self, _inputs) -> RoutingPolicyDecision:  # noqa: D401
        return RoutingPolicyDecision(POLICY_ACCEPT, None, {})


# ============================================================
# 1. Decision-dict additive contract
# ============================================================
class Tier5ObservabilityContract(unittest.TestCase):

    _TIER5_KEYS = (
        "tier5_active",
        "tier5_policy",
        "tier5_policy_reason",
        "tier5_policy_inputs",
    )

    def test_all_tier5_keys_present_without_intent_vector(self) -> None:
        d = CognitiveRouterV4().route(_NEUTRAL_QUERY)
        for k in self._TIER5_KEYS:
            self.assertIn(k, d)
        self.assertTrue(d["tier5_active"])
        self.assertEqual(d["tier5_policy"], POLICY_ACCEPT)

    def test_all_tier5_keys_present_with_intent_vector(self) -> None:
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        d = router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(m=0.5))
        for k in self._TIER5_KEYS:
            self.assertIn(k, d)

    def test_field_types_match_documented_contract(self) -> None:
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        d = router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(m=0.5))
        self.assertIsInstance(d["tier5_active"], bool)
        self.assertIsInstance(d["tier5_policy"], str)
        self.assertIn(d["tier5_policy"], POLICY_VALUES)
        # ``tier5_policy_reason`` is None on accept; str on rule fired.
        self.assertTrue(d["tier5_policy_reason"] is None
                        or isinstance(d["tier5_policy_reason"], str))
        self.assertIsInstance(d["tier5_policy_inputs"], dict)

    def test_inputs_snapshot_carries_expected_keys(self) -> None:
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        d = router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(m=0.5))
        snap = d["tier5_policy_inputs"]
        for k in ("stability_score", "oscillation_index",
                  "lane_bias_max_abs", "confidence_margin",
                  "would_upgrade_to_hybrid"):
            self.assertIn(k, snap, f"snapshot key {k!r} missing")


# ============================================================
# 2. Engine always activates (unlike Tier 4)
# ============================================================
class Tier5AlwaysActiveTests(unittest.TestCase):
    """Tier 5 differs from Tier 4: it does NOT depend on
    ``intent_vector``. The engine still runs (and reports ``accept``)
    when no embedding is available; only the Tier 4-derived signals
    go neutral."""

    def test_engine_active_without_intent_vector(self) -> None:
        d = CognitiveRouterV4().route(_NEUTRAL_QUERY)
        self.assertTrue(d["tier5_active"])
        self.assertEqual(d["tier5_policy"], POLICY_ACCEPT)
        self.assertEqual(d["tier5_policy_inputs"]["stability_score"], 0.0)
        self.assertEqual(d["tier5_policy_inputs"]["oscillation_index"], 0.0)

    def test_engine_active_when_tier4_below_min_observations(self) -> None:
        # Fresh cluster (size < MIN_CLUSTER_OBSERVATIONS) -> Tier 4
        # reports None for dominant_frequency. Engine derives 0.0.
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        d = router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(m=0.5))
        self.assertTrue(d["tier5_active"])
        self.assertEqual(d["tier5_policy_inputs"]["stability_score"], 0.0)


# ============================================================
# 3. Schema additivity (one canonical key per prior tier)
# ============================================================
class Tier5SchemaAdditivity(unittest.TestCase):

    def test_tier1_2_3_4_keys_still_present(self) -> None:
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        d = router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(m=0.5))
        for k in (
            "route",                   # Tier 1
            "memory_score_final",      # Tier 2
            "memory_lane_bias",        # Tier 3
            "tier3_band_active",       # Tier 3 refinement
            "tier4_active",            # Tier 4
            "tier4_cluster_id",        # Tier 4
        ):
            self.assertIn(k, d)


# ============================================================
# 4. The user-spec integration smoke test
# ============================================================
class Tier5IntegrationSmokeTest(unittest.TestCase):
    """`tier2_tier3_tier4_integration_smoke_test` — drive a real
    router through three crafted turns and assert the corresponding
    ``tier5_policy`` values."""

    def _new_router(self) -> CognitiveRouterV4:
        r = CognitiveRouterV4()
        _install_tier2_centroids(r)
        return r

    def test_neutral_turn_yields_accept(self) -> None:
        router = self._new_router()
        d = router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(m=0.5))
        self.assertEqual(d["tier5_policy"], POLICY_ACCEPT)
        self.assertIsNone(d["tier5_policy_reason"])

    def test_suppress_flip_turn_via_precooked_oscillation(self) -> None:
        """Pre-feed the Tier 4 tracker with an alternating route
        pattern so the cluster becomes oscillating BEFORE the next
        route() call. Combined with a low confidence_margin (close
        memory/rag scores) the engine must emit suppress_flip."""
        router = self._new_router()
        v = _intent_vec(m=0.5)
        # Pre-cook oscillation: 8 alternations of memory/rag drives
        # dominant_frequency to ~0.50 (oscillation_index 0.50 > 0.40)
        # and recent_routes contains both routes.
        for r in ["memory", "rag"] * 8:
            router.stability_tracker.observe(v, r)
        # Drive a turn whose memory/rag scores are close. With m=0.27
        # and r=0.27 the cosines are tied and confidence_margin == 0.0.
        v_close = _intent_vec(m=0.27, r=0.27)
        d = router.route(_NEUTRAL_QUERY, intent_vector=v_close)
        # Sanity: oscillation_index > 0.40 and margin < 0.10
        self.assertGreater(d["tier5_policy_inputs"]["oscillation_index"], 0.40)
        self.assertLess(d["confidence_margin"], 0.10)
        self.assertEqual(d["tier5_policy"], POLICY_SUPPRESS_FLIP)
        self.assertEqual(d["tier5_policy_reason"], "suppress_flip")

    def test_stabilize_turn_via_lane_bias_under_low_margin(self) -> None:
        """Build memory-success feedback so the memory adaptive_offset
        saturates at +MAX_ABS_OFFSET. The damped applied bias is
        ~0.03 > STABILIZE_LANE_BIAS_MIN(0.02). Drive a close-margin
        in-band turn so confidence_margin < 0.15 and the band-gate
        applies the bias. Engine must emit stabilize."""
        router = self._new_router()
        # Saturate memory lane with successes. The Tier 3 convention
        # is that a reliable lane gets a NEGATIVE offset (lowers the
        # threshold = more lenient routing); ``lane_bias_max_abs`` in
        # the snapshot uses ``abs()`` so the sign does not matter for
        # the stabilize trigger.
        for _ in range(RECENT_WINDOW_SIZE):
            router.observe_feedback(_evt(route="memory", success=True))
        self.assertNotEqual(
            router.lane_stats.adaptive_offset("memory"), 0.0,
            msg="raw offset should be saturated after RECENT_WINDOW_SIZE successes",
        )
        # In-band tied scores -> margin 0.0, top_score in [0.30, 0.75]
        v_close = _intent_vec(m=0.5, r=0.5)
        d = router.route(_NEUTRAL_QUERY, intent_vector=v_close)
        self.assertTrue(d["tier3_band_active"])
        self.assertGreater(d["tier5_policy_inputs"]["lane_bias_max_abs"], 0.02)
        self.assertLess(d["confidence_margin"], 0.15)
        # Either stabilize OR override_to_hybrid could fire (the
        # close margin + in-retrieval-pair second_intent satisfies
        # would_upgrade_to_hybrid). With Tier 4 dormant
        # (size < MIN_CLUSTER_OBSERVATIONS) stability_score == 0.0
        # so override CANNOT fire and stabilize wins.
        self.assertEqual(d["tier5_policy"], POLICY_STABILIZE)
        self.assertEqual(d["tier5_policy_reason"], "stabilize")


# ============================================================
# 5. Route-field never modified — the canonical v1 invariant
# ============================================================
class Tier5RouteFieldNeverModified(unittest.TestCase):
    """200-query randomized walk against two routers — one with the
    live engine, one with a ``_NullPolicyEngine`` — must produce
    byte-identical ``route`` values for every turn. This is the
    canonical "v1 is observability-only" pin, mirroring the Tier 4
    invariant test."""

    def _make_live(self) -> CognitiveRouterV4:
        r = CognitiveRouterV4()
        _install_tier2_centroids(r)
        return r

    def _make_control(self) -> CognitiveRouterV4:
        r = CognitiveRouterV4()
        _install_tier2_centroids(r)
        r.policy_engine = _NullPolicyEngine()  # type: ignore[assignment]
        return r

    def test_random_walk_routes_byte_identical_against_null_engine(self) -> None:
        rng = random.Random(0xCAFE)
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

    def test_route_field_unchanged_when_policy_is_suppress_flip(self) -> None:
        """Even when the engine emits the strongest policy signal,
        v1 must leave ``route`` alone."""
        live = self._make_live()
        ctrl = self._make_control()
        v = _intent_vec(m=0.27, r=0.27)
        # Pre-cook oscillation on BOTH so the only thing that diverges
        # is the engine.
        for r in ["memory", "rag"] * 8:
            live.stability_tracker.observe(v, r)
            ctrl.stability_tracker.observe(v, r)
        d_live = live.route(_NEUTRAL_QUERY, intent_vector=v)
        d_ctrl = ctrl.route(_NEUTRAL_QUERY, intent_vector=v)
        self.assertEqual(d_live["tier5_policy"], POLICY_SUPPRESS_FLIP)
        self.assertEqual(d_ctrl["tier5_policy"], POLICY_ACCEPT)
        self.assertEqual(d_live["route"], d_ctrl["route"])


# ============================================================
# 6. Defensive — engine exception is swallowed
# ============================================================
class Tier5DefensiveTests(unittest.TestCase):

    def test_engine_exception_does_not_crash_route(self) -> None:
        class _Boom:
            def evaluate(self, *_args, **_kwargs):
                raise RuntimeError("synthetic engine failure")

        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        router.policy_engine = _Boom()  # type: ignore[assignment]
        with self.assertLogs("Qube.CognitiveRouterV4", level="WARNING") as cm:
            d = router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(m=0.5))
        self.assertIn("tier5 evaluate raised", "\n".join(cm.output))
        self.assertFalse(d["tier5_active"])
        self.assertEqual(d["tier5_policy"], POLICY_NO_ACTION)
        self.assertIsNone(d["tier5_policy_reason"])
        # Rest of the decision dict is fully populated as usual.
        self.assertIn("route", d)


# ============================================================
# 7. Dormant-regression — varied stream stays at accept
# ============================================================
class Tier5DormantRegressionTests(unittest.TestCase):
    """Without any feedback (Tier 3 dormant) and without any
    pre-cooked Tier 4 oscillation, a varied query stream MUST
    produce ``tier5_policy == POLICY_ACCEPT`` for every turn.
    Confirms the engine adds no spurious policies."""

    def test_varied_stream_yields_accept_throughout(self) -> None:
        # Use distinct, non-zero amplitudes per axis so each axis lands
        # in its own Tier 4 cluster and inter-lane scores don't tie at
        # zero (a tied 0.0 score with an alternating-route stream would
        # produce oscillation + margin 0.0 = legitimate suppress_flip).
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        rng = random.Random(0xBADD)
        amps_per_axis = {"m": (0.50, 0.65), "r": (0.55, 0.70), "w": (0.60, 0.75)}
        for _ in range(50):
            axis = rng.choice(("m", "r", "w"))
            amp = rng.choice(amps_per_axis[axis])
            v = _intent_vec(**{axis: amp})
            d = router.route(_NEUTRAL_QUERY, intent_vector=v)
            self.assertEqual(
                d["tier5_policy"], POLICY_ACCEPT,
                msg=f"unexpected policy {d['tier5_policy']} on axis={axis} amp={amp}",
            )


if __name__ == "__main__":
    unittest.main()
