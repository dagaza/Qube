"""Tier 6: Routing Arbitration Layer (RAL), v1 — observability-only.

This module is the only home of Tier 6 logic. It is imported by
``mcp/cognitive_router.py`` and the dedicated unit tests; nothing
else in the project should touch it directly.

Design properties (each pinned by a test):

- **Pure function**: ``RoutingArbitrationLayer.evaluate(inputs)`` is
  a deterministic function of a small frozen dataclass. No I/O, no
  state, no mutation of any tier.

- **Conflict-and-explanation only**: emits a list of conflict flags
  and an interpretation enum. The router publishes both in the
  decision dict but never assigns them back to the ``route`` field.

- **Reads Tier 5 output, never re-evaluates it**: depends only on
  ``tier5_decision.policy``. Tier 5's pure-function guarantee is
  what makes this safe — the same Tier 2/3/4 inputs that produced
  ``tier5_policy`` are also fed back here, so the decision is
  logically consistent without re-running the Tier 5 rules.

- **Defensive failure mode**: degrades to the passthrough sentinel
  (``flags=()``, ``interpretation="passthrough"``); one WARNING
  per failure type, never raises.

- **Bounded log noise**: the layer itself logs nothing on the happy
  path. The router emits one INFO line per turn iff at least one
  conflict flag fires.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("Qube.RoutingArbitrationLayer")


# ============================================================
# Conflict flag enum (string values so the decision dict stays
# JSON-safe).
# ============================================================

CONFLICT_STABILITY_OVERRIDE     = "stability_override"
CONFLICT_STRUCTURAL_INSTABILITY = "structural_instability"
CONFLICT_ADAPTIVE_CONFLICT      = "adaptive_conflict"

CONFLICT_FLAGS_VALUES: tuple[str, ...] = (
    CONFLICT_STABILITY_OVERRIDE,
    CONFLICT_STRUCTURAL_INSTABILITY,
    CONFLICT_ADAPTIVE_CONFLICT,
)


# ============================================================
# Interpretation enum.
# ============================================================

INTERPRETATION_STABLE              = "stable"
INTERPRETATION_POLICY_DOMINANT     = "policy_dominant"
INTERPRETATION_STRUCTURAL_UNSTABLE = "structural_unstable"
INTERPRETATION_ADAPTIVE_PRESSURE   = "adaptive_pressure"
INTERPRETATION_PASSTHROUGH         = "passthrough"

INTERPRETATION_VALUES: tuple[str, ...] = (
    INTERPRETATION_STABLE,
    INTERPRETATION_POLICY_DOMINANT,
    INTERPRETATION_STRUCTURAL_UNSTABLE,
    INTERPRETATION_ADAPTIVE_PRESSURE,
    INTERPRETATION_PASSTHROUGH,
)


# ============================================================
# Tier 6 thresholds — module-level so a future tuning PR is a
# one-line change pinned by tests.
# ============================================================

# Rule B — structural_instability: oscillating cluster AND tight
# margin. Slightly stricter than Tier 5's ``suppress_flip``
# oscillation gate (0.40) so structural-instability is a stronger
# claim than Tier 5's "prefer to be conservative" recommendation.
STRUCTURAL_OSCILLATION_MIN = 0.50
STRUCTURAL_CONF_MARGIN_MAX = 0.10

# Rule C — adaptive_conflict: Tier 3 lane bias is non-trivial AND
# Tier 5 is recommending stabilize. Mirrors Tier 5's
# ``STABILIZE_LANE_BIAS_MIN`` so the two thresholds stay aligned.
ARBITRATION_LANE_BIAS_MIN = 0.02


# ============================================================
# Data structures
# ============================================================

@dataclass(frozen=True)
class RoutingArbitrationInputs:
    """Frozen snapshot of the upstream signals the layer needs.

    Built by the router from values already in scope at the Tier 6
    hook point (immediately after the Tier 5 block). The layer
    treats this as opaque read-only input.
    """
    # ----- Tier 2 inputs -----
    confidence_margin: float
    top_intent: str
    # ----- Tier 3 inputs -----
    memory_lane_bias: float
    rag_lane_bias: float
    web_lane_bias: float
    # ----- Tier 4 inputs (layer derives the numeric signals) -----
    tier4_active: bool
    tier4_cluster_dominant_frequency: Optional[float]
    tier4_cluster_oscillating: bool
    # ----- Tier 5 inputs -----
    tier5_policy: str
    tier5_policy_reason: Optional[str]


@dataclass(frozen=True)
class RoutingArbitrationDecision:
    """Layer output — observability-only in v1.

    ``conflict_flags`` is a deterministic tuple in
    ``CONFLICT_FLAGS_VALUES`` order; the empty tuple means no
    conflict. ``interpretation`` is one of ``INTERPRETATION_VALUES``.
    ``inputs_snapshot`` is the small JSON-serializable dict of the
    numeric values the rules were evaluated against.
    """
    conflict_flags: tuple[str, ...]
    interpretation: str
    inputs_snapshot: dict


_PASSTHROUGH_DECISION = RoutingArbitrationDecision(
    conflict_flags=(),
    interpretation=INTERPRETATION_PASSTHROUGH,
    inputs_snapshot={},
)


# ============================================================
# Layer
# ============================================================

class RoutingArbitrationLayer:
    """Stateless conflict-and-explanation layer.

    The layer is a pure function in the unit-test sense: an
    instance only carries ``_warned`` (a small set used to dedupe
    failure-mode WARNINGs).
    """

    def __init__(self) -> None:
        self._warned: set[str] = set()

    # ----- public API -------------------------------------------------

    def evaluate(self, inputs: RoutingArbitrationInputs) -> RoutingArbitrationDecision:
        """Deterministic, total over all inputs.

        On any unexpected exception inside the inner evaluation,
        returns the passthrough sentinel and emits a single WARNING
        per exception type. The router additionally wraps this call
        in try/except as defense-in-depth, so even a misbehaving
        ``_warned`` set cannot kill a user-facing turn.
        """
        try:
            return self._evaluate_inner(inputs)
        except Exception as exc:  # pragma: no cover - defensive only
            key = type(exc).__name__
            if key not in self._warned:
                self._warned.add(key)
                logger.warning(
                    "[Tier6RAL] evaluate raised %s: %r — degrading to passthrough. "
                    "Further occurrences of this type are silenced.",
                    key, exc,
                )
            return _PASSTHROUGH_DECISION

    # ----- internals --------------------------------------------------

    def _evaluate_inner(
        self, inputs: RoutingArbitrationInputs,
    ) -> RoutingArbitrationDecision:
        # Lazy import to avoid an import cycle (cognitive_router
        # imports this module). Module-level lookup, not per-call.
        from mcp.routing_policy_engine import (
            POLICY_STABILIZE,
            POLICY_SUPPRESS_FLIP,
        )

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

        snapshot = {
            "stability_score":   stability_score,
            "oscillation_index": oscillation_index,
            "lane_bias_max_abs": lane_bias_max_abs,
            "confidence_margin": float(inputs.confidence_margin),
            "tier5_policy":      inputs.tier5_policy,
        }

        # ---- Conflict rules (each independent; non-exclusive) ----
        fired: list[str] = []

        # Rule A — policy dominance
        if inputs.tier5_policy == POLICY_SUPPRESS_FLIP:
            fired.append(CONFLICT_STABILITY_OVERRIDE)

        # Rule B — structural dominance
        if (
            oscillation_index > STRUCTURAL_OSCILLATION_MIN
            and inputs.confidence_margin < STRUCTURAL_CONF_MARGIN_MAX
        ):
            fired.append(CONFLICT_STRUCTURAL_INSTABILITY)

        # Rule C — adaptive conflict
        if (
            lane_bias_max_abs > ARBITRATION_LANE_BIAS_MIN
            and inputs.tier5_policy == POLICY_STABILIZE
        ):
            fired.append(CONFLICT_ADAPTIVE_CONFLICT)

        flags = tuple(fired)
        interpretation = self._interpret(flags)
        return RoutingArbitrationDecision(flags, interpretation, snapshot)

    @staticmethod
    def _interpret(flags: tuple[str, ...]) -> str:
        """Total mapping from the 8 possible flag sets to one of
        ``INTERPRETATION_VALUES`` (excluding ``passthrough`` which
        is reserved for the failure path)."""
        flag_set = set(flags)
        if not flag_set:
            return INTERPRETATION_STABLE
        # Structural instability is the strongest signal; if it is
        # present the interpretation collapses to it regardless of
        # any co-fired flags.
        if CONFLICT_STRUCTURAL_INSTABILITY in flag_set:
            return INTERPRETATION_STRUCTURAL_UNSTABLE
        if CONFLICT_STABILITY_OVERRIDE in flag_set:
            # Policy beats adaptive when structure is OK.
            return INTERPRETATION_POLICY_DOMINANT
        if CONFLICT_ADAPTIVE_CONFLICT in flag_set:
            return INTERPRETATION_ADAPTIVE_PRESSURE
        # Unreachable given the current enum, but safe.
        return INTERPRETATION_STABLE  # pragma: no cover
