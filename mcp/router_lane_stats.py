"""Tier 3: bounded, observable, feedback-driven calibration layer for
the cognitive router.

This module is the only home of Tier 3 state. It is imported by
``mcp/cognitive_router.py`` and the dedicated unit tests; nothing
else in the project should touch it directly.

Design properties (each pinned by a test):

- **Bounded**: every per-lane statistic that drives a runtime
  decision lives inside a ``deque(maxlen=RECENT_WINDOW_SIZE)``.
  No accumulator that influences thresholds can grow without
  bound, so there is no exponential drift and no unbounded memory.

- **Reversible**: the adaptive offset is a *pure read* over the
  recent window. Flipping the success-rate trend reverses the
  offset within at most ``RECENT_WINDOW_SIZE`` events.

- **Dormant by default**: until a lane has accumulated
  ``MIN_OBSERVATIONS`` events, ``adaptive_offset`` returns exactly
  ``0.0`` so a fresh router with no feedback history is
  bit-identical to the Tier 1 + Tier 2 router.

- **Bounded magnitude**: the offset is hard-clamped to
  ``[-MAX_ABS_OFFSET, +MAX_ABS_OFFSET]`` regardless of inputs.

- **Pure read**: ``adaptive_offset(lane)`` and the trust accessors
  never mutate state. Only ``update(event)`` mutates.

- **Persistence boundary**: ``to_dict()`` / ``from_dict()`` round-
  trip the full state through JSON-serializable primitives so a
  future Tier 3.1 PR can wire disk persistence without changing
  any public behavior.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

# ============================================================
# Tier 3 constants — module-level so a future tuning PR is a
# one-line change pinned by tests rather than scattered magic
# numbers.
# ============================================================

# Maximum number of recent (success: bool) outcomes the per-lane
# rolling window remembers. Bounding this is the central guarantee
# against cumulative drift.
RECENT_WINDOW_SIZE = 50

# Below this many observations on a lane, ``adaptive_offset`` returns
# exactly 0.0 — Tier 3 is a strict no-op until enough data exists to
# justify a calibration nudge.
MIN_OBSERVATIONS = 10

# Hard clamp on |adaptive_offset|. The offset can never push the
# threshold by more than this in either direction, no matter how
# extreme the recent success rate is.
MAX_ABS_OFFSET = 0.05

# ----- Tier 3 calibration refinement: damped lane bias -----
#
# The router consumes ``adaptive_offset(...)`` as a damped, gated
# *lane bias* rather than a raw threshold mutation. ``adaptive_offset``
# itself still returns the raw bounded ``[-MAX_ABS_OFFSET,
# +MAX_ABS_OFFSET]`` value (preserved for observability and so this
# data layer carries no policy); the cognitive router multiplies it
# by ``LANE_BIAS_DAMPING`` and applies it ONLY when the post-fusion
# ``top_score`` sits in the decision-boundary band. See
# ``mcp/cognitive_router.py`` for the application site.
#
# Effective max influence on a threshold is therefore:
#   |applied_lane_bias| <= LANE_BIAS_RAW_CLAMP * LANE_BIAS_DAMPING
#                       == 0.05 * 0.6
#                       == 0.03
#
# These two constants are exposed as a single source of truth so a
# future tuning PR is a one-line change pinned by tests.
LANE_BIAS_RAW_CLAMP = 0.05
LANE_BIAS_DAMPING   = 0.6

# Below this recent success rate, the lane is treated as unreliable
# and the threshold is RAISED (suppress the lane more).
LOW_RELIABILITY_BAND = 0.4

# Above this recent success rate, the lane is treated as reliable
# and the threshold is LOWERED (use the lane more).
HIGH_RELIABILITY_BAND = 0.7

# Laplace / Beta prior added to embedding/substring trust scores so
# they start at 0.5 and only drift with sufficient evidence. Avoids
# divide-by-zero and overconfidence on small N.
BETA_PRIOR = 1.0

# Canonical lane keys. Recall is gated separately by the existing
# recall_threshold + chat-margin rule and is intentionally NOT
# part of the Tier 3 calibration set.
_LANES: tuple[str, ...] = ("memory", "rag", "web")
_RETRIEVAL_PAIR: frozenset[str] = frozenset({"memory", "rag"})


# ============================================================
# Event: the input contract from LLMWorker -> Router -> LaneStats.
# ============================================================
@dataclass(frozen=True)
class RouteFeedbackEvent:
    """One per-turn deterministic feedback signal.

    Constructed in ``workers/llm_worker.py`` after the existing
    post-retrieval downgrade block so ``success`` reflects the
    genuine post-gate state. ``frozen=True`` so the event is safe
    to share across observers and cannot be mutated mid-update.
    """

    # Canonical lowercase route from the decision dict:
    # "memory" | "rag" | "web" | "hybrid" | "none".
    route: str

    # Top intent + its source from the Tier 2 confidence layer.
    # ``top_source`` is "embedding" or "substring" and drives the
    # per-source trust accumulators.
    top_intent: str
    top_source: str

    # From the Tier 2 confidence layer — recorded for observability
    # only; the offset function does not consume it.
    confidence_margin: float

    # Already computed at the existing telemetry site.
    latency_ms: float

    # Deterministic success signal — see module docstring + plan §3.
    # For "hybrid", this is the route-level success (memory_hits>0
    # OR rag_hits>0); per-lane success for hybrid is reconstructed
    # in ``LaneStatsRegistry.update`` from ``per_lane_hits``.
    success: bool

    # Drift turns are no-ops (retrieval was suppressed, no signal).
    drift: bool

    # Per-lane hit counts so hybrid can be attributed cleanly:
    # memory lane gets credit/blame based on memory_hits, RAG lane
    # based on rag_hits, regardless of the overall route.
    per_lane_hits: dict


# ============================================================
# LaneStats: per-lane mutable counters + bounded recency window.
# ============================================================
@dataclass
class LaneStats:
    """Per-lane calibration counters.

    ``_recent`` is the only state the adaptive offset reads. The
    ``total`` / ``success`` / ``embedding_*`` / ``substring_*``
    counters are lifetime totals exposed for observability and trust
    scoring; they do NOT drive routing offsets.
    """

    total: int = 0
    success: int = 0

    embedding_total: int = 0
    embedding_success: int = 0
    substring_total: int = 0
    substring_success: int = 0

    # Diagnostic only — not consumed by ``adaptive_offset``.
    confidence_sum: float = 0.0

    # Bounded by construction. Default factory pinned by
    # ``test_recent_window_is_bounded``.
    _recent: deque = field(
        default_factory=lambda: deque(maxlen=RECENT_WINDOW_SIZE)
    )

    # ---- mutators (only called via LaneStatsRegistry.update) ----
    def record(self, success: bool, source: str, confidence_margin: float) -> None:
        """Record one outcome on this lane.

        ``source`` must be ``"embedding"`` or ``"substring"``;
        anything else is ignored for the per-source counters but the
        global ``total`` / ``success`` and ``_recent`` window are
        still updated so the adaptive offset stays accurate even if
        a future caller emits an unknown source.
        """
        self.total += 1
        if success:
            self.success += 1
        self._recent.append(bool(success))
        self.confidence_sum += float(confidence_margin)

        if source == "embedding":
            self.embedding_total += 1
            if success:
                self.embedding_success += 1
        elif source == "substring":
            self.substring_total += 1
            if success:
                self.substring_success += 1

    # ---- pure readers ------------------------------------------
    def n_recent(self) -> int:
        return len(self._recent)

    def recent_success_rate(self) -> float | None:
        """Returns ``None`` when below ``MIN_OBSERVATIONS`` so callers
        can present "no opinion" instead of a noisy ratio."""
        n = self.n_recent()
        if n < MIN_OBSERVATIONS:
            return None
        return sum(self._recent) / n

    def embedding_trust(self) -> float:
        return (self.embedding_success + BETA_PRIOR) / (
            self.embedding_total + 2.0 * BETA_PRIOR
        )

    def substring_trust(self) -> float:
        return (self.substring_success + BETA_PRIOR) / (
            self.substring_total + 2.0 * BETA_PRIOR
        )

    # ---- persistence boundary stub -----------------------------
    def to_dict(self) -> dict:
        return {
            "total":             self.total,
            "success":           self.success,
            "embedding_total":   self.embedding_total,
            "embedding_success": self.embedding_success,
            "substring_total":   self.substring_total,
            "substring_success": self.substring_success,
            "confidence_sum":    self.confidence_sum,
            "_recent":           [bool(x) for x in self._recent],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LaneStats":
        s = cls()
        s.total             = int(data.get("total", 0))
        s.success           = int(data.get("success", 0))
        s.embedding_total   = int(data.get("embedding_total", 0))
        s.embedding_success = int(data.get("embedding_success", 0))
        s.substring_total   = int(data.get("substring_total", 0))
        s.substring_success = int(data.get("substring_success", 0))
        s.confidence_sum    = float(data.get("confidence_sum", 0.0))
        recent: Iterable = data.get("_recent", [])
        for v in recent:
            s._recent.append(bool(v))
        return s


# ============================================================
# LaneStatsRegistry: owns one LaneStats per lane, exposes the
# router-side public API.
# ============================================================
class LaneStatsRegistry:
    """The single Tier 3 state container held on ``CognitiveRouterV4``.

    Public API:
      - ``update(event)``           — mutating, called per turn
      - ``adaptive_offset(lane)``   — pure read, called from
                                       ``_compute_dynamic_thresholds``
      - ``recent_success_rate(lane)``
      - ``embedding_trust(lane)`` / ``substring_trust(lane)``
      - ``tier3_active()``          — True if any lane has ``>=
                                       MIN_OBSERVATIONS`` observations
      - ``snapshot()`` / ``to_dict()`` / ``from_dict()`` — persistence
                                       boundary stub
      - ``reset_lane(lane)``        — for tests

    FUTURE (Tier 3.1, OFF by default, gated by min sample size):
        # Once ``embedding_trust(L)`` / ``substring_trust(L)`` have
        # accumulated significant evidence (e.g. >= 30 observations
        # of EACH source on the lane), an OPT-IN weighted fusion
        # could be introduced as:
        #   fused = max(
        #       substring_score * substring_trust(L),
        #       embedding_score * embedding_trust(L),
        #   )
        # Tier 3 deliberately does NOT implement this — default
        # fusion stays max() and the trust scores are observed only.
    """

    def __init__(self) -> None:
        self._lanes: dict[str, LaneStats] = {
            lane: LaneStats() for lane in _LANES
        }

    # ---- mutating API ------------------------------------------
    def update(self, event: RouteFeedbackEvent) -> None:
        """Attribute one feedback event to the correct lane(s).

        Skips:
          - drift turns (retrieval was suppressed; no signal),
          - ``route == "none"`` (no retrieval was attempted),
          - unknown routes.

        For ``"hybrid"`` BOTH ``memory`` and ``rag`` lanes are
        updated independently with their own per-lane success
        derived from ``per_lane_hits`` — never inheriting the
        route-level success blob.
        """
        if event is None:
            return
        if event.drift:
            return
        route = event.route
        if route == "none" or route not in ("memory", "rag", "web", "hybrid"):
            return

        per_lane_hits = event.per_lane_hits or {}

        if route in ("memory", "rag", "web"):
            self._lanes[route].record(
                success=bool(event.success),
                source=event.top_source,
                confidence_margin=event.confidence_margin,
            )
            return

        # route == "hybrid": credit each retrieval lane independently
        # based on its own hit count. Web is never part of hybrid.
        for lane in _RETRIEVAL_PAIR:
            lane_hits = int(per_lane_hits.get(lane, 0))
            lane_success = lane_hits > 0
            self._lanes[lane].record(
                success=lane_success,
                source=event.top_source,
                confidence_margin=event.confidence_margin,
            )

    # ---- pure readers ------------------------------------------
    def adaptive_offset(self, lane: str) -> float:
        """Bounded, dormant-by-default calibration offset for ``lane``.

        Returns a value in ``[-MAX_ABS_OFFSET, +MAX_ABS_OFFSET]``.
        Returns exactly ``0.0`` when:
          - the lane is unknown (defensive),
          - the lane has fewer than ``MIN_OBSERVATIONS`` recent events,
          - the recent success rate is in the neutral band
            ``(LOW_RELIABILITY_BAND, HIGH_RELIABILITY_BAND)``.
        """
        stats = self._lanes.get(lane)
        if stats is None:
            return 0.0

        n = stats.n_recent()
        if n < MIN_OBSERVATIONS:
            return 0.0

        r = sum(stats._recent) / n

        if r >= HIGH_RELIABILITY_BAND:
            scale = min(1.0, (r - HIGH_RELIABILITY_BAND) / (1.0 - HIGH_RELIABILITY_BAND))
            offset = -MAX_ABS_OFFSET * scale
        elif r <= LOW_RELIABILITY_BAND:
            denom = LOW_RELIABILITY_BAND if LOW_RELIABILITY_BAND > 0 else 1.0
            scale = min(1.0, (LOW_RELIABILITY_BAND - r) / denom)
            offset = +MAX_ABS_OFFSET * scale
        else:
            offset = 0.0

        # Defensive final clamp — should never matter mathematically
        # but pinned by the random-events stability test.
        if offset > MAX_ABS_OFFSET:
            return MAX_ABS_OFFSET
        if offset < -MAX_ABS_OFFSET:
            return -MAX_ABS_OFFSET
        return offset

    def recent_success_rate(self, lane: str) -> float | None:
        stats = self._lanes.get(lane)
        if stats is None:
            return None
        return stats.recent_success_rate()

    def embedding_trust(self, lane: str) -> float:
        stats = self._lanes.get(lane)
        if stats is None:
            return 0.5
        return stats.embedding_trust()

    def substring_trust(self, lane: str) -> float:
        stats = self._lanes.get(lane)
        if stats is None:
            return 0.5
        return stats.substring_trust()

    def tier3_active(self) -> bool:
        """True iff any lane has reached ``MIN_OBSERVATIONS`` events.

        Mirrors the Tier 2 ``any_embedding_centroid`` activation flag:
        a single boolean a UI / log line can grep to know whether the
        Tier 3 layer is influencing routing this turn.
        """
        return any(s.n_recent() >= MIN_OBSERVATIONS for s in self._lanes.values())

    # ---- test / debug ------------------------------------------
    def reset_lane(self, lane: str) -> None:
        if lane in self._lanes:
            self._lanes[lane] = LaneStats()

    def snapshot(self) -> dict:
        """Lightweight, JSON-friendly debug view of every lane."""
        return {
            lane: {
                "total":                stats.total,
                "success":              stats.success,
                "n_recent":             stats.n_recent(),
                "recent_success_rate":  stats.recent_success_rate(),
                "adaptive_offset":      self.adaptive_offset(lane),
                "embedding_trust":      stats.embedding_trust(),
                "substring_trust":      stats.substring_trust(),
            }
            for lane, stats in self._lanes.items()
        }

    # ---- persistence boundary stub -----------------------------
    def to_dict(self) -> dict:
        return {lane: stats.to_dict() for lane, stats in self._lanes.items()}

    @classmethod
    def from_dict(cls, data: dict) -> "LaneStatsRegistry":
        reg = cls()
        for lane in _LANES:
            lane_data = (data or {}).get(lane)
            if isinstance(lane_data, dict):
                reg._lanes[lane] = LaneStats.from_dict(lane_data)
        return reg
