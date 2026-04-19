"""Unit tests for ``mcp.routing_stability_tracker`` — Tier 4
data layer (observe-only RSL, v1).

Pure numpy synthetic vectors; no Qt / audio / embedder. Mirrors
the test layout of ``tests/test_router_lane_stats.py``.
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mcp.routing_stability_tracker import (
    MAX_CLUSTERS,
    MIN_CLUSTER_OBSERVATIONS,
    OSCILLATION_DOMINANT_FRAC,
    RECENT_ROUTE_WINDOW,
    SIMILARITY_THRESHOLD,
    ClusterDiagnostic,
    RoutingStabilityTracker,
    _l2_normalize,
)


_DIM = 8


def _axis(i: int) -> np.ndarray:
    """Unit vector along the ``i``-th axis in R^_DIM."""
    v = np.zeros(_DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def _near(i: int, jitter: float = 0.05) -> np.ndarray:
    """Vector close to axis ``i`` (cosine ~ sqrt(1 - jitter^2))."""
    v = _axis(i)
    v[(i + 1) % _DIM] = jitter
    return v


# ============================================================
# 1. Constants pin
# ============================================================
class ConstantsPinTests(unittest.TestCase):
    def test_constants_match_design(self) -> None:
        self.assertEqual(MAX_CLUSTERS, 64)
        self.assertAlmostEqual(SIMILARITY_THRESHOLD, 0.85, places=6)
        self.assertEqual(RECENT_ROUTE_WINDOW, 10)
        self.assertEqual(MIN_CLUSTER_OBSERVATIONS, 3)
        self.assertAlmostEqual(OSCILLATION_DOMINANT_FRAC, 0.60, places=6)


# ============================================================
# 2. _l2_normalize helper
# ============================================================
class L2NormalizeTests(unittest.TestCase):
    def test_normalize_simple(self) -> None:
        v = _l2_normalize(np.array([3.0, 4.0]))
        self.assertIsNotNone(v)
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=5)

    def test_none_returns_none(self) -> None:
        self.assertIsNone(_l2_normalize(None))

    def test_zero_norm_returns_none(self) -> None:
        self.assertIsNone(_l2_normalize(np.zeros(4)))

    def test_non_finite_returns_none(self) -> None:
        self.assertIsNone(_l2_normalize(np.array([1.0, np.inf, 2.0])))
        self.assertIsNone(_l2_normalize(np.array([1.0, np.nan])))

    def test_empty_returns_none(self) -> None:
        self.assertIsNone(_l2_normalize(np.array([])))

    def test_garbage_input_returns_none(self) -> None:
        self.assertIsNone(_l2_normalize("not a vector"))  # type: ignore[arg-type]


# ============================================================
# 3. Cluster create / join
# ============================================================
class ClusterCreateJoinTests(unittest.TestCase):
    def test_first_observation_creates_cluster(self) -> None:
        t = RoutingStabilityTracker()
        d = t.observe(_axis(0), "memory")
        self.assertTrue(d.active)
        self.assertEqual(d.cluster_id, 0)
        self.assertEqual(d.cluster_size, 0)  # PRE-update snapshot
        self.assertEqual(len(t.clusters), 1)
        # And the fold-in actually happened post-snapshot.
        self.assertEqual(t.clusters[0].count, 1)
        self.assertEqual(t.clusters[0].route_counts["memory"], 1)

    def test_similar_query_joins_existing_cluster(self) -> None:
        t = RoutingStabilityTracker()
        t.observe(_axis(0), "memory")
        d2 = t.observe(_near(0, jitter=0.05), "memory")
        # Joined the same cluster (id 0).
        self.assertEqual(d2.cluster_id, 0)
        self.assertEqual(len(t.clusters), 1)
        self.assertEqual(t.clusters[0].count, 2)

    def test_dissimilar_query_creates_new_cluster(self) -> None:
        t = RoutingStabilityTracker()
        t.observe(_axis(0), "memory")
        d2 = t.observe(_axis(1), "rag")  # orthogonal -> sim 0.0 < 0.85
        self.assertEqual(d2.cluster_id, 1)
        self.assertEqual(len(t.clusters), 2)

    def test_three_orthogonal_queries_get_three_cluster_ids(self) -> None:
        t = RoutingStabilityTracker()
        ids = [
            t.observe(_axis(0), "memory").cluster_id,
            t.observe(_axis(1), "rag").cluster_id,
            t.observe(_axis(2), "web").cluster_id,
        ]
        self.assertEqual(ids, [0, 1, 2])

    def test_join_threshold_above_boundary_joins(self) -> None:
        """A query with cosine >= SIMILARITY_THRESHOLD against the
        cluster centroid must join (fresh tracker so the centroid
        is exactly axis 0)."""
        t = RoutingStabilityTracker()
        t.observe(_axis(0), "memory")
        v_above = np.array(
            [0.86, np.sqrt(1 - 0.86**2)] + [0.0] * (_DIM - 2),
            dtype=np.float32,
        )
        d = t.observe(v_above, "memory")
        self.assertEqual(d.cluster_id, 0)

    def test_join_threshold_below_boundary_creates_new_cluster(self) -> None:
        """A query with cosine < SIMILARITY_THRESHOLD against the
        cluster centroid must create a new cluster (fresh tracker
        so the centroid is exactly axis 0)."""
        t = RoutingStabilityTracker()
        t.observe(_axis(0), "memory")
        v_below = np.array(
            [0.84, np.sqrt(1 - 0.84**2)] + [0.0] * (_DIM - 2),
            dtype=np.float32,
        )
        d = t.observe(v_below, "memory")
        self.assertNotEqual(d.cluster_id, 0)


# ============================================================
# 4. Centroid math
# ============================================================
class CentroidUpdateTests(unittest.TestCase):
    def test_centroid_stays_unit_norm_after_updates(self) -> None:
        t = RoutingStabilityTracker()
        for jitter in (0.0, 0.02, 0.04, 0.05, 0.03):
            t.observe(_near(0, jitter=jitter), "memory")
        c = t.clusters[0]
        self.assertAlmostEqual(float(np.linalg.norm(c.centroid)), 1.0, places=5)

    def test_centroid_drifts_toward_mean_of_observations(self) -> None:
        t = RoutingStabilityTracker()
        # First observation locks the centroid to axis 0.
        t.observe(_axis(0), "memory")
        # Many observations on a slightly-shifted vector should pull
        # the centroid in that direction (still inside the cluster).
        shifted = _near(0, jitter=0.05)
        for _ in range(40):
            t.observe(shifted, "memory")
        # The post-mean centroid is closer to ``shifted`` than to ``axis 0``.
        c = t.clusters[0].centroid
        from mcp.routing_stability_tracker import _l2_normalize as norm
        s = norm(shifted)
        self.assertGreater(float(np.dot(c, s)), float(np.dot(c, _axis(0))))


# ============================================================
# 5. LRU eviction
# ============================================================
class LruEvictionTests(unittest.TestCase):
    def test_lru_eviction_at_max_clusters(self) -> None:
        t = RoutingStabilityTracker()
        # Fill to MAX_CLUSTERS using orthogonal-ish vectors. We can't
        # have more than _DIM=8 strictly orthogonal, but we can
        # generate MAX_CLUSTERS + 1 vectors that are pairwise below
        # the join threshold by using random unit vectors with a
        # different generator each time.
        rng = np.random.default_rng(0xC0DE)
        # Use a higher-dimensional space so we can easily fit > 64
        # mutually-distinct clusters.
        D = 256
        t._dim = None  # let observe lock the dim from the first input

        def rand_unit() -> np.ndarray:
            v = rng.standard_normal(D).astype(np.float32)
            v /= np.linalg.norm(v)
            return v

        # Insert MAX_CLUSTERS observations — random unit vectors in
        # high-D are pairwise close to orthogonal.
        for _ in range(MAX_CLUSTERS):
            t.observe(rand_unit(), "memory")
        self.assertEqual(len(t.clusters), MAX_CLUSTERS)
        # One more triggers eviction.
        t.observe(rand_unit(), "memory")
        self.assertEqual(len(t.clusters), MAX_CLUSTERS)

    def test_lru_evicts_least_recently_seen(self) -> None:
        """Touching cluster A keeps it alive; cluster B (untouched)
        is the one evicted when cap is reached."""
        t = RoutingStabilityTracker()
        D = 256
        rng = np.random.default_rng(0xBEEF)

        def rand_unit() -> np.ndarray:
            v = rng.standard_normal(D).astype(np.float32)
            v /= np.linalg.norm(v)
            return v

        first_two = [rand_unit(), rand_unit()]
        t.observe(first_two[0], "memory")  # cluster_id 0, the "kept alive" one
        t.observe(first_two[1], "rag")     # cluster_id 1, the "untouched" one
        # Fill to cap with new clusters.
        for _ in range(MAX_CLUSTERS - 2):
            t.observe(rand_unit(), "memory")
        # Touch cluster 0 again so its last_seen_ts is the most recent.
        t.observe(first_two[0], "memory")
        # Push one more new cluster to force eviction.
        t.observe(rand_unit(), "memory")
        ids = {c.cluster_id for c in t.clusters}
        self.assertIn(0, ids,  "cluster 0 was just touched and must survive")
        self.assertNotIn(1, ids, "cluster 1 was the LRU and must be evicted")


# ============================================================
# 6. Diagnostic — pre-update snapshot semantics
# ============================================================
class DiagnosticSnapshotTests(unittest.TestCase):
    def test_first_observation_returns_zero_size_snapshot(self) -> None:
        t = RoutingStabilityTracker()
        d = t.observe(_axis(0), "memory")
        self.assertEqual(d.cluster_size, 0)
        self.assertIsNone(d.dominant_route)
        self.assertIsNone(d.dominant_frequency)
        self.assertIsNone(d.route_consistent_with_cluster)
        self.assertFalse(d.is_oscillating)

    def test_below_min_observations_dominant_is_none(self) -> None:
        t = RoutingStabilityTracker()
        for _ in range(MIN_CLUSTER_OBSERVATIONS - 1):
            t.observe(_axis(0), "memory")
        d = t.observe(_axis(0), "memory")
        # Snapshot is taken pre-update, so cluster_size == MIN-1.
        self.assertLess(d.cluster_size, MIN_CLUSTER_OBSERVATIONS)
        self.assertIsNone(d.dominant_route)
        self.assertIsNone(d.dominant_frequency)

    def test_above_min_observations_dominant_is_set(self) -> None:
        t = RoutingStabilityTracker()
        for _ in range(MIN_CLUSTER_OBSERVATIONS):
            t.observe(_axis(0), "memory")
        d = t.observe(_axis(0), "memory")
        self.assertEqual(d.dominant_route, "memory")
        self.assertAlmostEqual(d.dominant_frequency, 1.0, places=6)
        self.assertTrue(d.route_consistent_with_cluster)


# ============================================================
# 7. Oscillation detection
# ============================================================
class OscillationDetectionTests(unittest.TestCase):
    def test_settled_cluster_is_not_oscillating(self) -> None:
        t = RoutingStabilityTracker()
        # 9x memory + 1x rag in same cluster -> 0.90 dominant frac > 0.60.
        for _ in range(9):
            t.observe(_axis(0), "memory")
        d = t.observe(_axis(0), "rag")
        self.assertFalse(d.is_oscillating)

    def test_alternating_routes_eventually_oscillating(self) -> None:
        t = RoutingStabilityTracker()
        # Perfect alternation -> dominant frac = 0.5 < 0.60 once enough
        # observations land. Snapshot is pre-update, so we need
        # MIN_CLUSTER_OBSERVATIONS+ pre-update events.
        routes = ["memory", "rag"] * 8  # 16 alternating obs
        last = None
        for r in routes:
            last = t.observe(_axis(0), r)
        assert last is not None
        self.assertTrue(last.is_oscillating)

    def test_oscillation_requires_two_distinct_recent_routes(self) -> None:
        """A cluster with one route in the recent window is never
        flagged oscillating, regardless of dominant_frequency."""
        t = RoutingStabilityTracker()
        for _ in range(20):
            t.observe(_axis(0), "memory")
        d = t.observe(_axis(0), "memory")
        self.assertFalse(d.is_oscillating)


# ============================================================
# 8. Defensive guards — observe is total over all inputs
# ============================================================
class DefensiveGuardTests(unittest.TestCase):
    def test_none_intent_vector_returns_dormant_diagnostic(self) -> None:
        t = RoutingStabilityTracker()
        d = t.observe(None, "memory")
        self.assertFalse(d.active)
        self.assertIsNone(d.cluster_id)
        self.assertEqual(len(t.clusters), 0)

    def test_zero_norm_intent_vector_is_dormant(self) -> None:
        t = RoutingStabilityTracker()
        d = t.observe(np.zeros(_DIM, dtype=np.float32), "memory")
        self.assertFalse(d.active)
        self.assertEqual(len(t.clusters), 0)

    def test_non_finite_intent_vector_is_dormant(self) -> None:
        t = RoutingStabilityTracker()
        bad = np.array([np.inf] + [0.0] * (_DIM - 1), dtype=np.float32)
        d = t.observe(bad, "memory")
        self.assertFalse(d.active)
        self.assertEqual(len(t.clusters), 0)

    def test_dim_mismatch_is_dormant_after_lock(self) -> None:
        t = RoutingStabilityTracker()
        t.observe(_axis(0), "memory")  # locks dim to _DIM
        wrong = np.ones(_DIM + 1, dtype=np.float32) / np.sqrt(_DIM + 1)
        d = t.observe(wrong, "memory")
        self.assertFalse(d.active)
        # Original cluster intact.
        self.assertEqual(len(t.clusters), 1)

    def test_empty_route_is_dormant(self) -> None:
        t = RoutingStabilityTracker()
        d = t.observe(_axis(0), "")
        self.assertFalse(d.active)
        self.assertEqual(len(t.clusters), 0)

    def test_non_string_route_is_dormant(self) -> None:
        t = RoutingStabilityTracker()
        d = t.observe(_axis(0), None)  # type: ignore[arg-type]
        self.assertFalse(d.active)
        d = t.observe(_axis(0), 42)    # type: ignore[arg-type]
        self.assertFalse(d.active)
        self.assertEqual(len(t.clusters), 0)


# ============================================================
# 9. reset()
# ============================================================
class ResetTests(unittest.TestCase):
    def test_reset_clears_everything(self) -> None:
        t = RoutingStabilityTracker()
        for i in range(5):
            t.observe(_axis(i % _DIM), "memory")
        t.reset()
        self.assertEqual(len(t.clusters), 0)
        self.assertIsNone(t._centroid_matrix)
        self.assertIsNone(t._dim)
        # And the next observation starts the cluster_id sequence over.
        d = t.observe(_axis(0), "memory")
        self.assertEqual(d.cluster_id, 0)


# ============================================================
# 10. snapshot() stub
# ============================================================
class SnapshotStubTests(unittest.TestCase):
    def test_snapshot_is_json_serializable_shape(self) -> None:
        import json
        t = RoutingStabilityTracker()
        for i in range(3):
            t.observe(_axis(i), "memory")
        snap = t.snapshot()
        # Round-trip through JSON to confirm primitive types only.
        s = json.dumps(snap)
        round_trip = json.loads(s)
        self.assertEqual(round_trip["version"], 1)
        self.assertEqual(round_trip["dim"], _DIM)
        self.assertEqual(len(round_trip["clusters"]), 3)


# ============================================================
# 11. Diagnostic dataclass shape
# ============================================================
class DiagnosticShapeTests(unittest.TestCase):
    def test_dormant_diagnostic_has_expected_fields(self) -> None:
        from mcp.routing_stability_tracker import _DORMANT_DIAGNOSTIC
        self.assertIsInstance(_DORMANT_DIAGNOSTIC, ClusterDiagnostic)
        self.assertFalse(_DORMANT_DIAGNOSTIC.active)
        self.assertIsNone(_DORMANT_DIAGNOSTIC.cluster_id)
        self.assertEqual(_DORMANT_DIAGNOSTIC.cluster_size, 0)
        self.assertIsNone(_DORMANT_DIAGNOSTIC.dominant_route)
        self.assertIsNone(_DORMANT_DIAGNOSTIC.dominant_frequency)
        self.assertFalse(_DORMANT_DIAGNOSTIC.is_oscillating)
        self.assertIsNone(_DORMANT_DIAGNOSTIC.route_consistent_with_cluster)


if __name__ == "__main__":
    unittest.main()
