"""Unit tests for ``mcp.routing_arbitration_layer`` — Tier 6 RAL v1.

Pure-function tests; no router instantiation. Mirrors the layout
of ``tests/test_routing_policy_engine.py``.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import unittest
from dataclasses import replace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mcp.routing_arbitration_layer import (
    ARBITRATION_LANE_BIAS_MIN,
    CONFLICT_ADAPTIVE_CONFLICT,
    CONFLICT_FLAGS_VALUES,
    CONFLICT_STABILITY_OVERRIDE,
    CONFLICT_STRUCTURAL_INSTABILITY,
    INTERPRETATION_ADAPTIVE_PRESSURE,
    INTERPRETATION_PASSTHROUGH,
    INTERPRETATION_POLICY_DOMINANT,
    INTERPRETATION_STABLE,
    INTERPRETATION_STRUCTURAL_UNSTABLE,
    INTERPRETATION_VALUES,
    STRUCTURAL_CONF_MARGIN_MAX,
    STRUCTURAL_OSCILLATION_MIN,
    RoutingArbitrationDecision,
    RoutingArbitrationInputs,
    RoutingArbitrationLayer,
)
from mcp.routing_policy_engine import (
    POLICY_ACCEPT,
    POLICY_NO_ACTION,
    POLICY_OVERRIDE_HYBRID,
    POLICY_STABILIZE,
    POLICY_SUPPRESS_FLIP,
)


# ============================================================
# Fixtures
# ============================================================

def _neutral_inputs(**overrides) -> RoutingArbitrationInputs:
    """Inputs that should produce ``flags=()`` / ``stable`` by default:
    Tier 4 stable cluster, zero lane bias, comfortable margin,
    Tier 5 = accept."""
    base = dict(
        confidence_margin=0.30,
        top_intent="rag",
        memory_lane_bias=0.0,
        rag_lane_bias=0.0,
        web_lane_bias=0.0,
        tier4_active=True,
        tier4_cluster_dominant_frequency=0.95,
        tier4_cluster_oscillating=False,
        tier5_policy=POLICY_ACCEPT,
        tier5_policy_reason=None,
    )
    base.update(overrides)
    return RoutingArbitrationInputs(**base)


# ============================================================
# 1. Constants pin
# ============================================================
class ConstantsPinTests(unittest.TestCase):

    def test_constants_match_design(self) -> None:
        self.assertAlmostEqual(STRUCTURAL_OSCILLATION_MIN, 0.50, places=6)
        self.assertAlmostEqual(STRUCTURAL_CONF_MARGIN_MAX, 0.10, places=6)
        self.assertAlmostEqual(ARBITRATION_LANE_BIAS_MIN, 0.02, places=6)

    def test_conflict_flag_enum(self) -> None:
        self.assertEqual(
            set(CONFLICT_FLAGS_VALUES),
            {CONFLICT_STABILITY_OVERRIDE,
             CONFLICT_STRUCTURAL_INSTABILITY,
             CONFLICT_ADAPTIVE_CONFLICT},
        )
        self.assertEqual(len(CONFLICT_FLAGS_VALUES), 3)

    def test_interpretation_enum(self) -> None:
        self.assertEqual(
            set(INTERPRETATION_VALUES),
            {INTERPRETATION_STABLE, INTERPRETATION_POLICY_DOMINANT,
             INTERPRETATION_STRUCTURAL_UNSTABLE,
             INTERPRETATION_ADAPTIVE_PRESSURE,
             INTERPRETATION_PASSTHROUGH},
        )
        self.assertEqual(len(INTERPRETATION_VALUES), 5)


# ============================================================
# 2. Default / stable path
# ============================================================
class DefaultStableTests(unittest.TestCase):

    def test_neutral_inputs_yield_no_flags_and_stable(self) -> None:
        d = RoutingArbitrationLayer().evaluate(_neutral_inputs())
        self.assertEqual(d.conflict_flags, ())
        self.assertEqual(d.interpretation, INTERPRETATION_STABLE)

    def test_stable_path_carries_inputs_snapshot(self) -> None:
        d = RoutingArbitrationLayer().evaluate(_neutral_inputs())
        for k in ("stability_score", "oscillation_index",
                  "lane_bias_max_abs", "confidence_margin",
                  "tier5_policy"):
            self.assertIn(k, d.inputs_snapshot)


# ============================================================
# 3. Rule A — policy dominance
# ============================================================
class PolicyDominanceTests(unittest.TestCase):

    def test_policy_dominance(self) -> None:
        d = RoutingArbitrationLayer().evaluate(_neutral_inputs(
            tier5_policy=POLICY_SUPPRESS_FLIP,
        ))
        self.assertIn(CONFLICT_STABILITY_OVERRIDE, d.conflict_flags)
        self.assertEqual(d.interpretation, INTERPRETATION_POLICY_DOMINANT)

    def test_no_policy_dominance_when_tier5_is_accept(self) -> None:
        d = RoutingArbitrationLayer().evaluate(_neutral_inputs(
            tier5_policy=POLICY_ACCEPT,
        ))
        self.assertNotIn(CONFLICT_STABILITY_OVERRIDE, d.conflict_flags)

    def test_no_policy_dominance_when_tier5_is_no_action(self) -> None:
        d = RoutingArbitrationLayer().evaluate(_neutral_inputs(
            tier5_policy=POLICY_NO_ACTION,
        ))
        self.assertNotIn(CONFLICT_STABILITY_OVERRIDE, d.conflict_flags)


# ============================================================
# 4. Rule B — structural instability
# ============================================================
class StructuralInstabilityTests(unittest.TestCase):

    def test_structural_instability(self) -> None:
        # oscillation = 1 - 0.40 = 0.60 (> 0.50), margin 0.05 (< 0.10)
        d = RoutingArbitrationLayer().evaluate(_neutral_inputs(
            tier4_cluster_dominant_frequency=0.40,
            confidence_margin=0.05,
        ))
        self.assertIn(CONFLICT_STRUCTURAL_INSTABILITY, d.conflict_flags)
        self.assertEqual(d.interpretation, INTERPRETATION_STRUCTURAL_UNSTABLE)

    def test_does_not_fire_at_oscillation_boundary(self) -> None:
        # oscillation = 0.50 exactly -> NOT > 0.50
        d = RoutingArbitrationLayer().evaluate(_neutral_inputs(
            tier4_cluster_dominant_frequency=0.50,
            confidence_margin=0.05,
        ))
        self.assertNotIn(CONFLICT_STRUCTURAL_INSTABILITY, d.conflict_flags)

    def test_does_not_fire_at_margin_boundary(self) -> None:
        # margin = 0.10 exactly -> NOT < 0.10
        d = RoutingArbitrationLayer().evaluate(_neutral_inputs(
            tier4_cluster_dominant_frequency=0.40,
            confidence_margin=0.10,
        ))
        self.assertNotIn(CONFLICT_STRUCTURAL_INSTABILITY, d.conflict_flags)

    def test_does_not_fire_when_tier4_dormant(self) -> None:
        d = RoutingArbitrationLayer().evaluate(_neutral_inputs(
            tier4_active=False,
            tier4_cluster_dominant_frequency=None,
            confidence_margin=0.05,
        ))
        self.assertNotIn(CONFLICT_STRUCTURAL_INSTABILITY, d.conflict_flags)


# ============================================================
# 5. Rule C — adaptive conflict
# ============================================================
class AdaptiveConflictTests(unittest.TestCase):

    def test_adaptive_conflict(self) -> None:
        d = RoutingArbitrationLayer().evaluate(_neutral_inputs(
            memory_lane_bias=0.04,
            tier5_policy=POLICY_STABILIZE,
        ))
        self.assertIn(CONFLICT_ADAPTIVE_CONFLICT, d.conflict_flags)
        self.assertEqual(d.interpretation, INTERPRETATION_ADAPTIVE_PRESSURE)

    def test_adaptive_uses_max_abs_across_lanes(self) -> None:
        # Negative bias should drive the trigger via abs().
        d = RoutingArbitrationLayer().evaluate(_neutral_inputs(
            memory_lane_bias=0.0,
            rag_lane_bias=0.0,
            web_lane_bias=-0.04,
            tier5_policy=POLICY_STABILIZE,
        ))
        self.assertIn(CONFLICT_ADAPTIVE_CONFLICT, d.conflict_flags)

    def test_does_not_fire_at_lane_bias_boundary(self) -> None:
        d = RoutingArbitrationLayer().evaluate(_neutral_inputs(
            memory_lane_bias=0.02,  # NOT > 0.02
            tier5_policy=POLICY_STABILIZE,
        ))
        self.assertNotIn(CONFLICT_ADAPTIVE_CONFLICT, d.conflict_flags)

    def test_does_not_fire_when_tier5_is_not_stabilize(self) -> None:
        d = RoutingArbitrationLayer().evaluate(_neutral_inputs(
            memory_lane_bias=0.04,
            tier5_policy=POLICY_OVERRIDE_HYBRID,
        ))
        self.assertNotIn(CONFLICT_ADAPTIVE_CONFLICT, d.conflict_flags)


# ============================================================
# 6. Multi-flag co-firing
# ============================================================
class MultiFlagCoFiringTests(unittest.TestCase):

    def test_stability_override_plus_structural_instability(self) -> None:
        # Tier 5 = suppress_flip AND oscillation > 0.50 AND margin < 0.10.
        # All three signals are coherent (oscillating + tight scores
        # are exactly what made Tier 5 say suppress_flip in the first
        # place). Both flags must fire; structural wins for the
        # interpretation per the precedence in section 3.3.
        d = RoutingArbitrationLayer().evaluate(_neutral_inputs(
            tier5_policy=POLICY_SUPPRESS_FLIP,
            tier4_cluster_dominant_frequency=0.40,
            confidence_margin=0.05,
        ))
        self.assertIn(CONFLICT_STABILITY_OVERRIDE, d.conflict_flags)
        self.assertIn(CONFLICT_STRUCTURAL_INSTABILITY, d.conflict_flags)
        self.assertEqual(d.interpretation, INTERPRETATION_STRUCTURAL_UNSTABLE)

    def test_flag_order_is_canonical(self) -> None:
        """``conflict_flags`` is reported in ``CONFLICT_FLAGS_VALUES``
        order, so downstream consumers can rely on a stable shape."""
        d = RoutingArbitrationLayer().evaluate(_neutral_inputs(
            tier5_policy=POLICY_SUPPRESS_FLIP,
            tier4_cluster_dominant_frequency=0.40,
            confidence_margin=0.05,
        ))
        # stability_override (0) appears before structural_instability (1).
        idx = [CONFLICT_FLAGS_VALUES.index(f) for f in d.conflict_flags]
        self.assertEqual(idx, sorted(idx))


# ============================================================
# 7. Interpretation mapping is total
# ============================================================
class InterpretationMappingTotalTests(unittest.TestCase):
    """Pin the deterministic mapping across all 8 possible
    flag-set combinations."""

    def test_interpretation_mapping_total(self) -> None:
        layer = RoutingArbitrationLayer()
        cases = {
            (): INTERPRETATION_STABLE,
            (CONFLICT_STABILITY_OVERRIDE,): INTERPRETATION_POLICY_DOMINANT,
            (CONFLICT_STRUCTURAL_INSTABILITY,): INTERPRETATION_STRUCTURAL_UNSTABLE,
            (CONFLICT_ADAPTIVE_CONFLICT,): INTERPRETATION_ADAPTIVE_PRESSURE,
            (CONFLICT_STABILITY_OVERRIDE,
             CONFLICT_STRUCTURAL_INSTABILITY): INTERPRETATION_STRUCTURAL_UNSTABLE,
            (CONFLICT_STABILITY_OVERRIDE,
             CONFLICT_ADAPTIVE_CONFLICT): INTERPRETATION_POLICY_DOMINANT,
            (CONFLICT_STRUCTURAL_INSTABILITY,
             CONFLICT_ADAPTIVE_CONFLICT): INTERPRETATION_STRUCTURAL_UNSTABLE,
            (CONFLICT_STABILITY_OVERRIDE,
             CONFLICT_STRUCTURAL_INSTABILITY,
             CONFLICT_ADAPTIVE_CONFLICT): INTERPRETATION_STRUCTURAL_UNSTABLE,
        }
        for flags, expected in cases.items():
            with self.subTest(flags=flags):
                self.assertEqual(
                    layer._interpret(flags), expected,
                    msg=f"interpret({flags}) expected {expected}",
                )


# ============================================================
# 8. Snapshot is JSON-serializable
# ============================================================
class SnapshotJsonTests(unittest.TestCase):

    def test_inputs_snapshot_is_json_serializable(self) -> None:
        d = RoutingArbitrationLayer().evaluate(_neutral_inputs(
            memory_lane_bias=0.04,
            tier5_policy=POLICY_STABILIZE,
        ))
        s = json.dumps(d.inputs_snapshot)
        rt = json.loads(s)
        for k in ("stability_score", "oscillation_index",
                  "lane_bias_max_abs", "confidence_margin",
                  "tier5_policy"):
            self.assertIn(k, rt)
        self.assertIsInstance(rt["tier5_policy"], str)


# ============================================================
# 9. Failure degradation
# ============================================================
class FailureDegradationTests(unittest.TestCase):

    def test_failure_degradation_returns_passthrough(self) -> None:
        layer = RoutingArbitrationLayer()

        def _boom(_inputs):
            raise RuntimeError("synthetic layer failure")

        layer._evaluate_inner = _boom  # type: ignore[assignment]
        with self.assertLogs("Qube.RoutingArbitrationLayer", level="WARNING") as cm:
            d = layer.evaluate(_neutral_inputs())
        self.assertEqual(d.conflict_flags, ())
        self.assertEqual(d.interpretation, INTERPRETATION_PASSTHROUGH)
        self.assertEqual(d.inputs_snapshot, {})
        self.assertIn("RuntimeError", "\n".join(cm.output))

    def test_failure_warnings_are_deduped_per_type(self) -> None:
        layer = RoutingArbitrationLayer()

        def _boom(_inputs):
            raise RuntimeError("repeat")

        layer._evaluate_inner = _boom  # type: ignore[assignment]
        with self.assertLogs("Qube.RoutingArbitrationLayer", level="WARNING"):
            layer.evaluate(_neutral_inputs())
        logger = logging.getLogger("Qube.RoutingArbitrationLayer")
        with self.assertNoLogs(logger=logger.name, level="WARNING"):
            layer.evaluate(_neutral_inputs())
            layer.evaluate(_neutral_inputs())

    def test_passthrough_carries_empty_inputs_snapshot(self) -> None:
        layer = RoutingArbitrationLayer()
        layer._evaluate_inner = lambda _i: (_ for _ in ()).throw(  # type: ignore[assignment]
            ValueError("synthetic"),
        )
        with self.assertLogs("Qube.RoutingArbitrationLayer", level="WARNING"):
            d = layer.evaluate(_neutral_inputs())
        self.assertEqual(d.inputs_snapshot, {})

    def test_evaluate_never_raises_under_pathological_inputs(self) -> None:
        layer = RoutingArbitrationLayer()
        weird = RoutingArbitrationInputs(
            confidence_margin=float("nan"),
            top_intent="",
            memory_lane_bias=float("nan"),
            rag_lane_bias=float("inf"),
            web_lane_bias=float("-inf"),
            tier4_active=True,
            tier4_cluster_dominant_frequency=float("nan"),
            tier4_cluster_oscillating=True,
            tier5_policy="some-unknown-policy",
            tier5_policy_reason=None,
        )
        # Must not raise.
        d = layer.evaluate(weird)
        self.assertIn(d.interpretation, INTERPRETATION_VALUES)
        for f in d.conflict_flags:
            self.assertIn(f, CONFLICT_FLAGS_VALUES)


# ============================================================
# 10. Decision dataclass shape
# ============================================================
class DecisionShapeTests(unittest.TestCase):

    def test_interpretation_value_is_always_in_enum(self) -> None:
        layer = RoutingArbitrationLayer()
        for inputs in (
            _neutral_inputs(),
            _neutral_inputs(tier5_policy=POLICY_SUPPRESS_FLIP),
            _neutral_inputs(memory_lane_bias=0.04, tier5_policy=POLICY_STABILIZE),
        ):
            d = layer.evaluate(inputs)
            self.assertIn(d.interpretation, INTERPRETATION_VALUES)
            for flag in d.conflict_flags:
                self.assertIn(flag, CONFLICT_FLAGS_VALUES)

    def test_decision_dataclass_is_frozen(self) -> None:
        d = RoutingArbitrationDecision((), INTERPRETATION_STABLE, {})
        with self.assertRaises(Exception):
            d.interpretation = INTERPRETATION_PASSTHROUGH  # type: ignore[misc]

    def test_inputs_dataclass_is_frozen(self) -> None:
        i = _neutral_inputs()
        with self.assertRaises(Exception):
            i.confidence_margin = 0.0  # type: ignore[misc]
        i2 = replace(i, confidence_margin=0.05)
        self.assertEqual(i2.confidence_margin, 0.05)


if __name__ == "__main__":
    unittest.main()
