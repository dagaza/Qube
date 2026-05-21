"""Tier 5: Routing Control Policy Layer (RCPL), v1 — observability-only.

This module is the only home of Tier 5 logic. It is imported by
``mcp/cognitive_router.py`` and the dedicated unit tests; nothing
else in the project should touch it directly.

Design properties (each pinned by a test):

- **Pure function**: ``RoutingPolicyEngine.evaluate(inputs)`` is a
  deterministic function of a small frozen dataclass. No I/O, no
  state, no mutation of any tier. Trivially safe to disable.

- **Observability-only in v1**: the engine emits a recommendation
  via ``RoutingPolicyDecision``. The router publishes it in the
  decision dict but never assigns it back to the ``route`` field.

- **Tier 4 numeric signals derived locally**: the spec uses
  ``oscillation_index`` and ``stability_score`` as floats, but the
  Tier 4 tracker only exposes ``is_oscillating: bool`` and
  ``dominant_frequency: float | None``. Deriving here keeps Tier 4
  byte-identical.

- **Shares Tier 2 thresholds via import**: ``AMBIGUITY_MARGIN`` and
  ``MIN_CONFIDENCE_FLOOR`` are imported from
  ``mcp.cognitive_router`` so the "would-upgrade-to-hybrid"
  predicate stays in lock-step with the actual upgrade predicate.

- **Explicit precedence**: when multiple rules match the order is
  ``suppress_flip > override_to_hybrid > stabilize > accept``.
  Pinned by a unit test.

- **Defensive failure mode**: ``no_action`` (distinguishable from
  ``accept`` in telemetry), one WARNING per failure type, never
  raises.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("Qube.RoutingPolicyEngine")


# ============================================================
# Policy enum (string values so the decision dict stays JSON-safe)
# ============================================================

POLICY_ACCEPT          = "accept"
POLICY_STABILIZE       = "stabilize"
POLICY_OVERRIDE_HYBRID = "override_to_hybrid"
POLICY_SUPPRESS_FLIP   = "suppress_flip"
POLICY_NO_ACTION       = "no_action"

POLICY_VALUES: tuple[str, ...] = (
    POLICY_ACCEPT,
    POLICY_STABILIZE,
    POLICY_OVERRIDE_HYBRID,
    POLICY_SUPPRESS_FLIP,
    POLICY_NO_ACTION,
)


# ============================================================
# Tier 5 thresholds — module-level so a future tuning PR is a
# one-line change pinned by tests.
# ============================================================

# Rule A — suppress_flip: oscillating cluster AND tight margin.
SUPPRESS_FLIP_OSCILLATION_MIN = 0.40
SUPPRESS_FLIP_CONF_MARGIN_MAX = 0.10

# Rule B — stabilize: Tier 3 lane bias is non-trivial AND margin is
# tight (the adaptive layer is "leaning" in low-confidence territory).
STABILIZE_LANE_BIAS_MIN  = 0.02
STABILIZE_CONF_MARGIN_MAX = 0.15

# Rule C — override_to_hybrid: Tier 2 ambiguity-upgrade WOULD fire
# AND the Tier 4 cluster is settled enough to support a deliberate
# action recommendation. Observability-only in v1 — the engine
# publishes the recommendation; the router does NOT act on it.
OVERRIDE_HYBRID_STABILITY_MIN = 0.70


# ============================================================
# Data structures
# ============================================================

@dataclass(frozen=True)
class RoutingPolicyInputs:
    """Frozen snapshot of the upstream signals the engine needs.

    Built by the router from values already in scope at the Tier 5
    hook point. The engine treats this as opaque read-only input;
    no side effects propagate back through it.
    """
    # ----- Tier 2 inputs -----
    confidence_margin: float
    top_intent: str
    top_score: float
    second_best_score: float
    second_best_intent: Optional[str]
    tier2_active: bool
    # ----- Tier 3 inputs -----
    memory_lane_bias: float
    rag_lane_bias: float
    web_lane_bias: float
    tier3_band_active: bool
    # ----- Tier 4 inputs (engine derives the numeric signals) -----
    tier4_active: bool
    tier4_cluster_size: int
    tier4_cluster_dominant_frequency: Optional[float]
    tier4_cluster_oscillating: bool


@dataclass(frozen=True)
class RoutingPolicyDecision:
    """Engine output — observability-only in v1.

    ``policy`` is the canonical signal. ``reason`` is a short label
    naming the rule that fired (None on accept / no_action).
    ``inputs_snapshot`` is the tiny JSON-serializable dict of the
    numeric values the rule was evaluated against, for telemetry.
    """
    policy: str
    reason: Optional[str]
    inputs_snapshot: dict


_EMPTY_SNAPSHOT: dict = {}


# ============================================================
# Engine
# ============================================================

class RoutingPolicyEngine:
    """Stateless policy synthesis layer.

    The engine is a pure function in the unit-test sense: an
    instance only carries ``_warned`` (a small set used to dedupe
    failure-mode WARNINGs). No per-turn state.
    """

    def __init__(self) -> None:
        # One WARNING per failure type, total. Bounded by the small
        # surface of exception kinds Python could raise inside
        # ``_evaluate_inner``.
        self._warned: set[str] = set()

    # ----- public API -------------------------------------------------

    def evaluate(self, inputs: RoutingPolicyInputs) -> RoutingPolicyDecision:
        """Deterministic, total over all inputs.

        On any unexpected exception inside the inner evaluation,
        returns ``POLICY_NO_ACTION`` and emits a single WARNING per
        exception type. The router additionally wraps this call in
        try/except as defense-in-depth, so even a misbehaving
        ``_warned`` set cannot kill a user-facing turn.
        """
        try:
            return self._evaluate_inner(inputs)
        except Exception as exc:  # pragma: no cover - defensive only
            key = type(exc).__name__
            if key not in self._warned:
                self._warned.add(key)
                logger.warning(
                    "[Tier5Policy] evaluate raised %s: %r — degrading to no_action. "
                    "Further occurrences of this type are silenced.",
                    key, exc,
                )
            return RoutingPolicyDecision(POLICY_NO_ACTION, None, _EMPTY_SNAPSHOT)

    # ----- internals --------------------------------------------------

    def _evaluate_inner(self, inputs: RoutingPolicyInputs) -> RoutingPolicyDecision:
        # Import here to avoid an import cycle (cognitive_router
        # imports this module). This is a one-time module-level
        # lookup, not a per-call cost.
        from mcp.cognitive_router import AMBIGUITY_MARGIN, MIN_CONFIDENCE_FLOOR

        # ---- Derived signals (single source of truth) ----
        if (
            inputs.tier4_active
            and inputs.tier4_cluster_dominant_frequency is not None
        ):
            stability_score = float(inputs.tier4_cluster_dominant_frequency)
            oscillation_index = 1.0 - stability_score
        else:
            stability_score = 0.0
            oscillation_index = 0.0

        lane_bias_max_abs = max(
            abs(inputs.memory_lane_bias),
            abs(inputs.rag_lane_bias),
            abs(inputs.web_lane_bias),
        )

        # Mirrors the existing Tier 2 ambiguity-upgrade predicate in
        # ``mcp/cognitive_router.py`` (the if-block at the end of the
        # confidence layer). We re-evaluate it here without acting on
        # it; the router's own predicate is unchanged.
        retrieval_pair = ("rag", "memory")
        would_upgrade_to_hybrid = bool(
            inputs.tier2_active
            and inputs.top_intent in retrieval_pair
            and inputs.second_best_intent in retrieval_pair
            and inputs.second_best_intent != inputs.top_intent
            and inputs.confidence_margin < AMBIGUITY_MARGIN
            and inputs.second_best_score >= MIN_CONFIDENCE_FLOOR
        )

        snapshot = {
            "stability_score":          stability_score,
            "oscillation_index":        oscillation_index,
            "lane_bias_max_abs":        lane_bias_max_abs,
            "confidence_margin":        float(inputs.confidence_margin),
            "would_upgrade_to_hybrid":  would_upgrade_to_hybrid,
        }

        # ---- Rule precedence: A > C > B > accept ----
        # Rule A — suppress_flip
        if (
            oscillation_index > SUPPRESS_FLIP_OSCILLATION_MIN
            and inputs.confidence_margin < SUPPRESS_FLIP_CONF_MARGIN_MAX
        ):
            return RoutingPolicyDecision(
                POLICY_SUPPRESS_FLIP, "suppress_flip", snapshot,
            )

        # Rule C — override_to_hybrid (observability-only)
        if (
            would_upgrade_to_hybrid
            and stability_score > OVERRIDE_HYBRID_STABILITY_MIN
        ):
            return RoutingPolicyDecision(
                POLICY_OVERRIDE_HYBRID, "override_to_hybrid", snapshot,
            )

        # Rule B — stabilize
        if (
            lane_bias_max_abs > STABILIZE_LANE_BIAS_MIN
            and inputs.confidence_margin < STABILIZE_CONF_MARGIN_MAX
        ):
            return RoutingPolicyDecision(
                POLICY_STABILIZE, "stabilize", snapshot,
            )

        # Default
        return RoutingPolicyDecision(POLICY_ACCEPT, None, snapshot)
