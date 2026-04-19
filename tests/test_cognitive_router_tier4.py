"""Tier 4 integration tests for ``CognitiveRouterV4`` —
Routing Stability Layer (RSL), v1 (observe-only).

Pins three guarantees:

  * **Dormant when no embedding** — ``intent_vector=None`` produces a
    sentinel diagnostic; no clusters are created.
  * **Decision-dict additive contract** — eight new ``tier4_*`` keys
    are present, with the documented types.
  * **Route field never modified** — for a randomized 200-query walk
    a router with the live tracker produces byte-identical ``route``
    values to a control router whose tracker is replaced with a
    no-op stub. This is the canonical "v1 is observe-only" pin.

We also cover cluster-id determinism on similar/dissimilar query
streams and oscillation detection at the integration level (with
the INFO log assertion).
"""
from __future__ import annotations

import logging
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
from mcp.routing_stability_tracker import (
    MIN_CLUSTER_OBSERVATIONS,
    ClusterDiagnostic,
    _DORMANT_DIAGNOSTIC,
)


_NEUTRAL_QUERY = "the quick brown fox jumps over the lazy dog"


# Synthetic centroids in R^6 — same shape as Tier 2 / Tier 3 tests
# so the embedding pathway lights up.
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


class _NullStabilityTracker:
    """Drop-in stub matching the public surface of
    ``RoutingStabilityTracker.observe`` — used by the
    ``Tier4RouteFieldNeverModified`` control router."""
    clusters: list = []

    def observe(self, intent_vector, route) -> ClusterDiagnostic:  # noqa: D401
        return _DORMANT_DIAGNOSTIC


# ============================================================
# 1. Dormant when no embedding
# ============================================================
class Tier4DormantWhenNoIntentVector(unittest.TestCase):

    def test_no_intent_vector_yields_dormant_diagnostic(self) -> None:
        router = CognitiveRouterV4()
        d = router.route(_NEUTRAL_QUERY)
        self.assertFalse(d["tier4_active"])
        self.assertIsNone(d["tier4_cluster_id"])
        self.assertEqual(d["tier4_cluster_size"], 0)
        self.assertIsNone(d["tier4_cluster_dominant_route"])
        self.assertIsNone(d["tier4_cluster_dominant_frequency"])
        self.assertFalse(d["tier4_cluster_oscillating"])
        self.assertIsNone(d["tier4_route_consistent_with_cluster"])
        self.assertEqual(d["tier4_total_clusters"], 0)
        # And no clusters were ever created.
        self.assertEqual(len(router.stability_tracker.clusters), 0)

    def test_no_intent_vector_repeatedly_does_not_create_clusters(self) -> None:
        router = CognitiveRouterV4()
        for _ in range(10):
            router.route(_NEUTRAL_QUERY)
        self.assertEqual(len(router.stability_tracker.clusters), 0)


# ============================================================
# 2. Decision-dict additive contract
# ============================================================
class Tier4ObservabilityContract(unittest.TestCase):

    _TIER4_KEYS = (
        "tier4_active",
        "tier4_cluster_id",
        "tier4_cluster_size",
        "tier4_cluster_dominant_route",
        "tier4_cluster_dominant_frequency",
        "tier4_cluster_oscillating",
        "tier4_route_consistent_with_cluster",
        "tier4_total_clusters",
    )

    def test_all_tier4_keys_present_when_dormant(self) -> None:
        d = CognitiveRouterV4().route(_NEUTRAL_QUERY)
        for k in self._TIER4_KEYS:
            self.assertIn(k, d, f"Tier-4 key {k!r} missing from decision dict.")

    def test_all_tier4_keys_present_when_active(self) -> None:
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        d = router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(m=0.5))
        for k in self._TIER4_KEYS:
            self.assertIn(k, d, f"Tier-4 key {k!r} missing from decision dict.")
        self.assertTrue(d["tier4_active"])

    def test_field_types_match_documented_contract(self) -> None:
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        # First two observations are below MIN_CLUSTER_OBSERVATIONS so
        # dominant fields are None; the third lifts them past the
        # threshold (snapshot is pre-update).
        v = _intent_vec(m=0.5)
        for _ in range(MIN_CLUSTER_OBSERVATIONS):
            router.route(_NEUTRAL_QUERY, intent_vector=v)
        d = router.route(_NEUTRAL_QUERY, intent_vector=v)
        self.assertIsInstance(d["tier4_active"], bool)
        self.assertIsInstance(d["tier4_cluster_id"], int)
        self.assertIsInstance(d["tier4_cluster_size"], int)
        self.assertIsInstance(d["tier4_cluster_dominant_route"], str)
        self.assertIsInstance(d["tier4_cluster_dominant_frequency"], float)
        self.assertIsInstance(d["tier4_cluster_oscillating"], bool)
        self.assertIsInstance(d["tier4_route_consistent_with_cluster"], bool)
        self.assertIsInstance(d["tier4_total_clusters"], int)


# ============================================================
# 3. Cluster-id determinism on similar / dissimilar streams
# ============================================================
class Tier4ClusterAssignmentTests(unittest.TestCase):

    def test_three_similar_queries_share_cluster_id(self) -> None:
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        v = _intent_vec(m=0.5)
        ids = [
            router.route(_NEUTRAL_QUERY, intent_vector=v)["tier4_cluster_id"]
            for _ in range(3)
        ]
        self.assertEqual(ids, [0, 0, 0])
        self.assertEqual(len(router.stability_tracker.clusters), 1)

    def test_three_orthogonal_queries_get_three_cluster_ids(self) -> None:
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        ids = [
            router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(m=0.5))["tier4_cluster_id"],
            router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(r=0.5))["tier4_cluster_id"],
            router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(w=0.5))["tier4_cluster_id"],
        ]
        self.assertEqual(sorted(ids), [0, 1, 2])
        self.assertEqual(len(router.stability_tracker.clusters), 3)

    def test_total_clusters_field_matches_tracker_state(self) -> None:
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        for axis in (_intent_vec(m=0.5), _intent_vec(r=0.5), _intent_vec(w=0.5)):
            d = router.route(_NEUTRAL_QUERY, intent_vector=axis)
        self.assertEqual(d["tier4_total_clusters"], 3)


# ============================================================
# 4. Oscillation detection (with INFO log)
# ============================================================
class Tier4OscillationDetection(unittest.TestCase):

    def test_alternating_routes_in_one_cluster_eventually_flag_oscillating(self) -> None:
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        # We can't easily force the router to alternate routes from
        # the same intent_vector; instead, we drive the tracker
        # directly through router.stability_tracker so we exercise
        # the same code path that the router sees on read.
        v = _intent_vec(m=0.5)
        for r in ["memory", "rag", "memory", "rag", "memory", "rag", "memory", "rag"]:
            router.stability_tracker.observe(v, r)
        # Now route once more so the decision dict carries the
        # post-feed diagnostic.
        d = router.route(_NEUTRAL_QUERY, intent_vector=v)
        self.assertTrue(d["tier4_cluster_oscillating"])

    def test_oscillation_emits_info_log(self) -> None:
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        v = _intent_vec(m=0.5)
        for r in ["memory", "rag"] * 6:
            router.stability_tracker.observe(v, r)
        with self.assertLogs("Qube.CognitiveRouterV4", level="INFO") as cm:
            router.route(_NEUTRAL_QUERY, intent_vector=v)
        joined = "\n".join(cm.output)
        self.assertIn("tier4 oscillation", joined)


# ============================================================
# 5. Route-field never modified — the canonical v1 invariant
# ============================================================
class Tier4RouteFieldNeverModified(unittest.TestCase):
    """The v1 promise is that Tier 4 is observe-only. We pin this by
    running a randomized 200-query walk against two routers — one
    with the live tracker, one with a no-op stub tracker — and
    asserting the ``route`` field is byte-identical for every turn."""

    def _make_live_router(self) -> CognitiveRouterV4:
        r = CognitiveRouterV4()
        _install_tier2_centroids(r)
        return r

    def _make_control_router(self) -> CognitiveRouterV4:
        r = CognitiveRouterV4()
        _install_tier2_centroids(r)
        r.stability_tracker = _NullStabilityTracker()  # type: ignore[assignment]
        return r

    def test_random_walk_routes_byte_identical_against_null_tracker(self) -> None:
        rng = random.Random(0xFEED)
        live = self._make_live_router()
        ctrl = self._make_control_router()

        amp_choices = (0.0, 0.10, 0.27, 0.40, 0.50, 0.65, 0.80, 0.92)
        axis_choices = ("m", "r", "w", None)  # None -> no intent vector

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

    def test_route_field_unchanged_when_cluster_is_oscillating(self) -> None:
        """Even when the cluster is hot-flagged as oscillating —
        which is the canonical "Tier 4 wants to do something"
        signal — v1 must still leave ``route`` alone."""
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        v = _intent_vec(m=0.5)
        # Pre-cook the tracker so the cluster is oscillating BEFORE
        # the router even runs.
        for r in ["memory", "rag"] * 8:
            router.stability_tracker.observe(v, r)
        # Snapshot what the router would have decided in isolation
        # by using a control router with a null tracker.
        ctrl = CognitiveRouterV4()
        _install_tier2_centroids(ctrl)
        ctrl.stability_tracker = _NullStabilityTracker()  # type: ignore[assignment]
        expected_route = ctrl.route(_NEUTRAL_QUERY, intent_vector=v)["route"]
        d = router.route(_NEUTRAL_QUERY, intent_vector=v)
        self.assertTrue(d["tier4_cluster_oscillating"])
        self.assertEqual(d["route"], expected_route)


# ============================================================
# 6. Defensive — tracker exception is swallowed
# ============================================================
class Tier4DefensiveTests(unittest.TestCase):

    def test_tracker_exception_does_not_crash_route(self) -> None:
        class _Boom:
            clusters: list = []
            def observe(self, *_args, **_kwargs):
                raise RuntimeError("synthetic failure")

        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        router.stability_tracker = _Boom()  # type: ignore[assignment]
        # route() must succeed and report a dormant Tier 4 diagnostic.
        with self.assertLogs("Qube.CognitiveRouterV4", level="WARNING") as cm:
            d = router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(m=0.5))
        self.assertIn("tier4 observe raised", "\n".join(cm.output))
        self.assertFalse(d["tier4_active"])
        # And the rest of the decision dict is fully populated as usual.
        self.assertIn("route", d)


# ============================================================
# 7. Pre-Tier-4 schema additivity — Tier 1/2/3 keys still present
# ============================================================
class Tier4SchemaAdditivity(unittest.TestCase):
    """The Tier 4 hook must not remove or rename any pre-existing
    decision dict key. We sample one canonical key from each prior
    tier; the broader contract is enforced by the existing
    Tier 1/2/3 contract test classes."""

    def test_tier1_2_3_keys_still_present(self) -> None:
        router = CognitiveRouterV4()
        _install_tier2_centroids(router)
        d = router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(m=0.5))
        # One canonical key per tier.
        for k in (
            "route",                   # Tier 1
            "memory_score_final",      # Tier 2
            "memory_lane_bias",        # Tier 3
            "tier3_band_active",       # Tier 3 refinement
        ):
            self.assertIn(k, d)


if __name__ == "__main__":
    unittest.main()
