"""Tier 4: Routing Stability Layer (RSL), v1 — observe-only.

This module is the only home of Tier 4 state. It is imported by
``mcp/cognitive_router.py`` and the dedicated unit tests; nothing
else in the project should touch it directly.

Design properties (each pinned by a test):

- **Observe-only in v1**: ``RoutingStabilityTracker`` records routing
  decisions and surfaces stability diagnostics, but it never proposes
  a route. The router consumes the diagnostic for telemetry only.
  Override semantics are deferred to a follow-up tier.

- **Embedding-gated**: when ``intent_vector is None`` (or the vector
  is unusable — zero norm, non-finite, or wrong dimension) the
  tracker returns a sentinel ``ClusterDiagnostic`` and performs no
  state mutation. Mirrors the Tier 2 ``any_embedding_centroid`` guard
  pattern.

- **Bounded memory**: the cluster store is hard-capped at
  ``MAX_CLUSTERS`` and uses LRU eviction by ``last_seen_ts``.

- **Deterministic**: tie-breaks use ``argmax`` (lowest index). No RNG.

- **Defensive**: ``observe(...)`` is total over all inputs — no input
  shape can crash a user-facing turn. The router additionally wraps
  it in try/except as a defense-in-depth layer.

- **Pre-update snapshot**: the diagnostic returned to the router is a
  snapshot of the cluster state BEFORE the current observation is
  folded in, so observability reflects historical patterns rather
  than the self-fulfilling current decision.
"""
from __future__ import annotations

import logging
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("Qube.RoutingStabilityTracker")


# ============================================================
# Tier 4 constants — module-level so a future tuning PR is a
# one-line change pinned by tests rather than scattered magic
# numbers.
# ============================================================

# Hard cap on the number of live clusters. The cluster store is
# centroid-keyed, not query-keyed, so 64 is generous for typical
# session diversity. Beyond this, LRU eviction by ``last_seen_ts``
# kicks in.
MAX_CLUSTERS = 64

# Cosine similarity required for an incoming query to JOIN an
# existing cluster. Below this, a new cluster is created (subject
# to the LRU cap). Tuned to match the typical inter-paraphrase
# similarity on transformer embeddings — close paraphrases are
# usually >= 0.85 cosine; topical drift falls below.
SIMILARITY_THRESHOLD = 0.85

# Per-cluster recent-route window, used only for the "is this
# cluster oscillating right now?" check. Independent of total
# ``count`` so the oscillation flag responds to recent behavior.
RECENT_ROUTE_WINDOW = 10

# Below this many observations the cluster's ``dominant_route`` is
# reported as ``None`` and ``is_oscillating`` is forced to ``False``.
# Prevents single-observation clusters from emitting noise.
MIN_CLUSTER_OBSERVATIONS = 3

# A cluster is flagged ``is_oscillating`` only when its dominant
# route's share of observations is below this fraction AND the
# recent window contains at least 2 distinct routes. A clean
# super-majority (>= 0.60) is treated as "settled".
OSCILLATION_DOMINANT_FRAC = 0.60


# ============================================================
# Data structures
# ============================================================

@dataclass
class ClusterStats:
    """Mutable per-cluster bookkeeping.

    ``centroid`` is L2-normalized after every update so cosine
    similarity reduces to a plain dot product on the next lookup.
    ``recent_routes`` is bounded by ``RECENT_ROUTE_WINDOW``;
    ``route_counts`` is the unbounded total over the cluster's
    lifetime (still bounded as a whole because it has at most one
    entry per routing label, of which there are six).
    """
    cluster_id: int
    centroid: np.ndarray
    count: int = 0
    route_counts: Counter = field(default_factory=Counter)
    recent_routes: deque = field(
        default_factory=lambda: deque(maxlen=RECENT_ROUTE_WINDOW)
    )
    last_seen_ts: float = 0.0


@dataclass(frozen=True)
class ClusterDiagnostic:
    """Snapshot of a cluster's state BEFORE the current observation
    is folded in. This is what the router publishes in the decision
    dict, so observability reflects historical patterns, not the
    self-fulfilling current decision.

    The ``active`` field is the canonical signal that the tracker
    actually ran for this turn: ``False`` means an unusable input
    (None, zero norm, non-finite, dim mismatch) and the rest of the
    fields carry sentinel values.
    """
    active: bool
    cluster_id: Optional[int]
    cluster_size: int
    dominant_route: Optional[str]
    dominant_frequency: Optional[float]
    is_oscillating: bool
    route_consistent_with_cluster: Optional[bool]


# Singleton sentinel returned whenever the tracker did not run.
_DORMANT_DIAGNOSTIC = ClusterDiagnostic(
    active=False,
    cluster_id=None,
    cluster_size=0,
    dominant_route=None,
    dominant_frequency=None,
    is_oscillating=False,
    route_consistent_with_cluster=None,
)


# ============================================================
# Helpers
# ============================================================

def _l2_normalize(v: np.ndarray) -> Optional[np.ndarray]:
    """Return ``v / ||v||`` as float32, or ``None`` if the vector is
    unusable (zero norm or non-finite). Total over all inputs."""
    if v is None:
        return None
    try:
        arr = np.asarray(v, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    n = float(np.linalg.norm(arr))
    if n <= 0.0 or not np.isfinite(n):
        return None
    return arr / n


# ============================================================
# Tracker
# ============================================================

class RoutingStabilityTracker:
    """Bounded, embedding-gated cluster bookkeeper.

    State is in-memory and ephemeral per worker restart. Public
    surface is intentionally narrow: ``observe(...)`` is the only
    mutating method; ``reset()`` clears all state; ``snapshot()``
    is stubbed for future persistence and unused in v1.
    """

    def __init__(self) -> None:
        self.clusters: list[ClusterStats] = []
        self._next_id: int = 0
        # Lazily-built ``(N, D)`` matrix mirroring ``self.clusters``.
        # Rebuilt only on cluster create/evict; the row for a joined
        # cluster is updated in place.
        self._centroid_matrix: Optional[np.ndarray] = None
        # The expected dimension is locked in by the first usable
        # vector and enforced thereafter to keep the matrix shape
        # consistent.
        self._dim: Optional[int] = None
        # One-shot guard so we don't spam the log on repeated dim
        # mismatches (e.g., a misconfigured embedder).
        self._warned_dim_mismatch: bool = False

    # ----- public API -------------------------------------------------

    def observe(
        self,
        intent_vector: Optional[np.ndarray],
        route: Optional[str],
    ) -> ClusterDiagnostic:
        """Find or create the cluster nearest to ``intent_vector``,
        return its pre-update diagnostic, then fold ``route`` into it.

        Total over all inputs — no input shape raises. When the input
        is unusable (None, zero norm, non-finite, dim mismatch, or
        ``route`` is falsy/non-str), the dormant sentinel is returned
        and no state is mutated.
        """
        if not isinstance(route, str) or not route:
            return _DORMANT_DIAGNOSTIC

        v = _l2_normalize(intent_vector)
        if v is None:
            return _DORMANT_DIAGNOSTIC

        if self._dim is None:
            self._dim = int(v.size)
        elif v.size != self._dim:
            if not self._warned_dim_mismatch:
                logger.warning(
                    "[RSL] intent_vector dim mismatch (got=%d expected=%d) "
                    "— skipping; further mismatches silenced.",
                    int(v.size), int(self._dim),
                )
                self._warned_dim_mismatch = True
            return _DORMANT_DIAGNOSTIC

        cluster = self._find_or_create_cluster(v)
        diag = self._snapshot_diagnostic(cluster, route)
        self._fold_in(cluster, v, route)
        return diag

    def reset(self) -> None:
        self.clusters.clear()
        self._next_id = 0
        self._centroid_matrix = None
        self._dim = None
        self._warned_dim_mismatch = False

    def snapshot(self) -> dict:
        """Persistence stub — unused in v1. Returns a JSON-serializable
        view that a future Tier 4.1 PR can persist to disk without
        changing any public behavior."""
        return {
            "version": 1,
            "dim": self._dim,
            "next_id": self._next_id,
            "clusters": [
                {
                    "cluster_id": c.cluster_id,
                    "centroid": c.centroid.tolist(),
                    "count": c.count,
                    "route_counts": dict(c.route_counts),
                    "recent_routes": list(c.recent_routes),
                    "last_seen_ts": c.last_seen_ts,
                }
                for c in self.clusters
            ],
        }

    # ----- internals --------------------------------------------------

    def _find_or_create_cluster(self, v: np.ndarray) -> ClusterStats:
        """Vectorized cosine search over the centroid matrix.
        Returns the best-matching existing cluster if its similarity
        meets ``SIMILARITY_THRESHOLD``, else creates a new cluster
        (evicting LRU if at cap)."""
        if self.clusters:
            sims = self._centroid_matrix @ v  # shape (N,)
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            if best_sim >= SIMILARITY_THRESHOLD:
                return self.clusters[best_idx]
        return self._create_cluster(v)

    def _create_cluster(self, v: np.ndarray) -> ClusterStats:
        if len(self.clusters) >= MAX_CLUSTERS:
            self._evict_lru()
        cluster = ClusterStats(
            cluster_id=self._next_id,
            centroid=v.copy(),
            last_seen_ts=time.monotonic(),
        )
        self._next_id += 1
        self.clusters.append(cluster)
        self._rebuild_centroid_matrix()
        return cluster

    def _evict_lru(self) -> None:
        """Evict the cluster with the smallest ``last_seen_ts``.
        ``last_seen_ts`` is updated on every join (not only on
        create), so frequently-revisited clusters survive."""
        idx = min(
            range(len(self.clusters)),
            key=lambda i: self.clusters[i].last_seen_ts,
        )
        del self.clusters[idx]
        self._rebuild_centroid_matrix()

    def _rebuild_centroid_matrix(self) -> None:
        if not self.clusters:
            self._centroid_matrix = None
            return
        self._centroid_matrix = np.stack(
            [c.centroid for c in self.clusters], axis=0
        )

    def _snapshot_diagnostic(
        self, cluster: ClusterStats, current_route: str,
    ) -> ClusterDiagnostic:
        """Compute the diagnostic from the cluster's PRE-update state."""
        size = cluster.count
        if size < MIN_CLUSTER_OBSERVATIONS:
            return ClusterDiagnostic(
                active=True,
                cluster_id=cluster.cluster_id,
                cluster_size=size,
                dominant_route=None,
                dominant_frequency=None,
                is_oscillating=False,
                route_consistent_with_cluster=None,
            )
        most, n = cluster.route_counts.most_common(1)[0]
        dominant_freq = n / size
        distinct = len(set(cluster.recent_routes))
        is_osc = (
            distinct >= 2 and dominant_freq < OSCILLATION_DOMINANT_FRAC
        )
        return ClusterDiagnostic(
            active=True,
            cluster_id=cluster.cluster_id,
            cluster_size=size,
            dominant_route=most,
            dominant_frequency=dominant_freq,
            is_oscillating=is_osc,
            route_consistent_with_cluster=(current_route == most),
        )

    def _fold_in(
        self, cluster: ClusterStats, v: np.ndarray, route: str,
    ) -> None:
        """Incremental centroid update + bookkeeping. Re-normalize the
        centroid so subsequent cosine lookups remain pure dot products,
        and refresh the row in the centroid matrix in place."""
        new_count = cluster.count + 1
        # Running mean: c_new = (c_old * n + v) / (n + 1)
        mixed = cluster.centroid * (cluster.count / new_count) + v / new_count
        norm = float(np.linalg.norm(mixed))
        if norm > 0.0 and np.isfinite(norm):
            cluster.centroid = (mixed / norm).astype(np.float32, copy=False)
        cluster.count = new_count
        cluster.route_counts[route] += 1
        cluster.recent_routes.append(route)
        cluster.last_seen_ts = time.monotonic()
        # Refresh in-place so the next ``_find_or_create_cluster`` sees
        # the updated centroid without rebuilding the whole matrix.
        if self._centroid_matrix is not None:
            try:
                idx = self.clusters.index(cluster)
                self._centroid_matrix[idx] = cluster.centroid
            except ValueError:
                # Cluster was evicted between snapshot and fold-in; the
                # next observe() will rebuild lazily via _create_cluster.
                self._rebuild_centroid_matrix()
