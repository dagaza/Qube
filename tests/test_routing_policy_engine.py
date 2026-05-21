"""Unit tests for ``mcp.routing_policy_engine`` — Tier 5 RCPL v1.

Pure-function tests; no router instantiation. Mirrors the layout
of ``tests/test_router_lane_stats.py`` and
``tests/test_routing_stability_tracker.py``.
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

from mcp.routing_policy_engine import (
    OVERRIDE_HYBRID_STABILITY_MIN,
    POLICY_ACCEPT,
    POLICY_NO_ACTION,
    POLICY_OVERRIDE_HYBRID,
    POLICY_STABILIZE,
    POLICY_SUPPRESS_FLIP,
    POLICY_VALUES,
    STABILIZE_CONF_MARGIN_MAX,
    STABILIZE_LANE_BIAS_MIN,
    SUPPRESS_FLIP_CONF_MARGIN_MAX,
    SUPPRESS_FLIP_OSCILLATION_MIN,
    RoutingPolicyDecision,
    RoutingPolicyEngine,
    RoutingPolicyInputs,
)


# ============================================================
# Fixtures
# ============================================================

def _neutral_inputs(**overrides) -> RoutingPolicyInputs:
    """Inputs that should produce ``POLICY_ACCEPT`` by default:
    Tier 4 dormant, zero lane bias, comfortable margin."""
    base = dict(
        confidence_margin=0.30,
        top_intent="rag",
        top_score=0.70,
        second_best_score=0.40,
        second_best_intent="memory",
        tier2_active=True,
        memory_lane_bias=0.0,
        rag_lane_bias=0.0,
        web_lane_bias=0.0,
        tier3_band_active=False,
        tier4_active=True,
        tier4_cluster_size=10,
        tier4_cluster_dominant_frequency=0.95,  # very stable
        tier4_cluster_oscillating=False,
    )
    base.update(overrides)
    return RoutingPolicyInputs(**base)


# ============================================================
# 1. Constants pin
# ============================================================
class ConstantsPinTests(unittest.TestCase):
    def test_constants_match_design(self) -> None:
        self.assertAlmostEqual(SUPPRESS_FLIP_OSCILLATION_MIN, 0.40, places=6)
        self.assertAlmostEqual(SUPPRESS_FLIP_CONF_MARGIN_MAX, 0.10, places=6)
        self.assertAlmostEqual(STABILIZE_LANE_BIAS_MIN, 0.02, places=6)
        self.assertAlmostEqual(STABILIZE_CONF_MARGIN_MAX, 0.15, places=6)
        self.assertAlmostEqual(OVERRIDE_HYBRID_STABILITY_MIN, 0.70, places=6)

    def test_policy_values_contains_all_five(self) -> None:
        self.assertEqual(
            set(POLICY_VALUES),
            {POLICY_ACCEPT, POLICY_STABILIZE, POLICY_OVERRIDE_HYBRID,
             POLICY_SUPPRESS_FLIP, POLICY_NO_ACTION},
        )
        self.assertEqual(len(POLICY_VALUES), 5)


# ============================================================
# 2. Default / accept path
# ============================================================
class PolicyAcceptDefaultTests(unittest.TestCase):
    def test_neutral_inputs_yield_accept(self) -> None:
        d = RoutingPolicyEngine().evaluate(_neutral_inputs())
        self.assertEqual(d.policy, POLICY_ACCEPT)
        self.assertIsNone(d.reason)

    def test_accept_still_carries_inputs_snapshot(self) -> None:
        d = RoutingPolicyEngine().evaluate(_neutral_inputs())
        for k in ("stability_score", "oscillation_index",
                  "lane_bias_max_abs", "confidence_margin",
                  "would_upgrade_to_hybrid"):
            self.assertIn(k, d.inputs_snapshot)


# ============================================================
# 3. Rule A — suppress_flip
# ============================================================
class SuppressFlipTriggerTests(unittest.TestCase):
    def test_suppress_flip_trigger(self) -> None:
        # oscillation 0.50 (= 1 - 0.50 dominant_freq), margin 0.05.
        d = RoutingPolicyEngine().evaluate(_neutral_inputs(
            tier4_cluster_dominant_frequency=0.50,
            confidence_margin=0.05,
        ))
        self.assertEqual(d.policy, POLICY_SUPPRESS_FLIP)
        self.assertEqual(d.reason, "suppress_flip")

    def test_suppress_flip_does_not_fire_when_oscillation_at_boundary(self) -> None:
        # oscillation_index = 1 - 0.60 = 0.40 → strictly NOT > 0.40 → no fire.
        d = RoutingPolicyEngine().evaluate(_neutral_inputs(
            tier4_cluster_dominant_frequency=0.60,
            confidence_margin=0.05,
            tier4_active=True,
        ))
        # margin alone keeps stabilize from firing (lane_bias is 0).
        self.assertEqual(d.policy, POLICY_ACCEPT)

    def test_suppress_flip_does_not_fire_when_margin_at_boundary(self) -> None:
        # confidence_margin = 0.10 → strictly NOT < 0.10 → no fire.
        d = RoutingPolicyEngine().evaluate(_neutral_inputs(
            tier4_cluster_dominant_frequency=0.50,
            confidence_margin=0.10,
        ))
        self.assertEqual(d.policy, POLICY_ACCEPT)


# ============================================================
# 4. Rule B — stabilize
# ============================================================
class StabilizeTriggerTests(unittest.TestCase):
    def test_stabilize_trigger(self) -> None:
        d = RoutingPolicyEngine().evaluate(_neutral_inputs(
            memory_lane_bias=0.025,
            confidence_margin=0.10,
            tier4_cluster_dominant_frequency=0.95,  # not oscillating
        ))
        self.assertEqual(d.policy, POLICY_STABILIZE)
        self.assertEqual(d.reason, "stabilize")

    def test_stabilize_uses_max_abs_across_lanes(self) -> None:
        # Negative web bias should drive the trigger via abs().
        d = RoutingPolicyEngine().evaluate(_neutral_inputs(
            memory_lane_bias=0.0,
            rag_lane_bias=0.0,
            web_lane_bias=-0.025,
            confidence_margin=0.10,
        ))
        self.assertEqual(d.policy, POLICY_STABILIZE)

    def test_stabilize_does_not_fire_at_lane_bias_boundary(self) -> None:
        d = RoutingPolicyEngine().evaluate(_neutral_inputs(
            memory_lane_bias=0.02,  # NOT strictly > 0.02
            confidence_margin=0.10,
        ))
        self.assertEqual(d.policy, POLICY_ACCEPT)


# ============================================================
# 5. Rule C — override_to_hybrid (observability-only)
# ============================================================
class OverrideToHybridTriggerTests(unittest.TestCase):
    def test_override_to_hybrid_trigger_non_executing(self) -> None:
        # would_upgrade_to_hybrid: top=rag, second=memory in retrieval pair,
        # margin 0.05 < AMBIGUITY_MARGIN(0.10), second_best 0.40 >= 0.30.
        # stability_score = 0.80 > 0.70.
        d = RoutingPolicyEngine().evaluate(_neutral_inputs(
            top_intent="rag",
            second_best_intent="memory",
            confidence_margin=0.05,
            second_best_score=0.40,
            top_score=0.45,
            tier4_cluster_dominant_frequency=0.80,
        ))
        self.assertEqual(d.policy, POLICY_OVERRIDE_HYBRID)
        self.assertEqual(d.reason, "override_to_hybrid")
        # Engine never carries a "new route" — caller is free to ignore.
        self.assertNotIn("new_route", d.inputs_snapshot)

    def test_override_does_not_fire_when_second_intent_not_in_retrieval_pair(self) -> None:
        d = RoutingPolicyEngine().evaluate(_neutral_inputs(
            top_intent="rag",
            second_best_intent="web",  # not in {rag, memory}
            confidence_margin=0.05,
            tier4_cluster_dominant_frequency=0.80,
        ))
        # Falls through; suppress_flip would need oscillation_index > 0.40.
        self.assertNotEqual(d.policy, POLICY_OVERRIDE_HYBRID)

    def test_override_does_not_fire_when_tier2_inactive(self) -> None:
        d = RoutingPolicyEngine().evaluate(_neutral_inputs(
            tier2_active=False,
            top_intent="rag",
            second_best_intent="memory",
            confidence_margin=0.05,
            tier4_cluster_dominant_frequency=0.80,
        ))
        self.assertNotEqual(d.policy, POLICY_OVERRIDE_HYBRID)

    def test_override_does_not_fire_when_stability_at_boundary(self) -> None:
        d = RoutingPolicyEngine().evaluate(_neutral_inputs(
            top_intent="rag",
            second_best_intent="memory",
            confidence_margin=0.05,
            tier4_cluster_dominant_frequency=0.70,  # NOT > 0.70
        ))
        self.assertNotEqual(d.policy, POLICY_OVERRIDE_HYBRID)


# ============================================================
# 6. Precedence
# ============================================================
class PrecedenceTests(unittest.TestCase):
    """Pin the precedence order: suppress_flip > override_to_hybrid > stabilize."""

    def test_precedence_suppress_flip_wins_over_others(self) -> None:
        # Construct inputs that satisfy ALL three rules simultaneously.
        # - suppress_flip: oscillation > 0.40, margin < 0.10
        # - override_to_hybrid: would_upgrade + stability > 0.70
        #   These two are mutually exclusive in stability_score:
        #   oscillation > 0.40 means stability < 0.60, so override
        #   cannot ALSO fire from the same dominant_frequency. We test
        #   suppress_flip vs. stabilize instead, then override vs.
        #   stabilize separately.
        d = RoutingPolicyEngine().evaluate(_neutral_inputs(
            tier4_cluster_dominant_frequency=0.50,  # osc = 0.50
            confidence_margin=0.05,                 # < 0.10 and < 0.15
            memory_lane_bias=0.04,                  # > 0.02
        ))
        self.assertEqual(d.policy, POLICY_SUPPRESS_FLIP)

    def test_precedence_override_wins_over_stabilize(self) -> None:
        # Stability 0.80 (override fires), margin 0.05 (also satisfies
        # stabilize), lane bias 0.04 (also satisfies stabilize).
        d = RoutingPolicyEngine().evaluate(_neutral_inputs(
            top_intent="rag",
            second_best_intent="memory",
            confidence_margin=0.05,
            tier4_cluster_dominant_frequency=0.80,
            memory_lane_bias=0.04,
        ))
        self.assertEqual(d.policy, POLICY_OVERRIDE_HYBRID)


# ============================================================
# 7. Derived signals — neutrality when Tier 4 dormant
# ============================================================
class DerivedSignalsTests(unittest.TestCase):
    def test_derived_signals_neutral_when_tier4_dormant(self) -> None:
        d = RoutingPolicyEngine().evaluate(_neutral_inputs(
            tier4_active=False,
            tier4_cluster_dominant_frequency=None,
            confidence_margin=0.05,  # would have triggered suppress_flip
        ))
        self.assertEqual(d.inputs_snapshot["stability_score"], 0.0)
        self.assertEqual(d.inputs_snapshot["oscillation_index"], 0.0)
        # And suppress_flip cannot fire.
        self.assertNotEqual(d.policy, POLICY_SUPPRESS_FLIP)

    def test_derived_signals_neutral_when_dominant_frequency_none(self) -> None:
        d = RoutingPolicyEngine().evaluate(_neutral_inputs(
            tier4_active=True,
            tier4_cluster_dominant_frequency=None,
            confidence_margin=0.05,
        ))
        self.assertEqual(d.inputs_snapshot["stability_score"], 0.0)
        self.assertEqual(d.inputs_snapshot["oscillation_index"], 0.0)
        self.assertNotEqual(d.policy, POLICY_SUPPRESS_FLIP)

    def test_lane_bias_max_abs_uses_max_of_three_lanes(self) -> None:
        d = RoutingPolicyEngine().evaluate(_neutral_inputs(
            memory_lane_bias=0.005,
            rag_lane_bias=-0.030,
            web_lane_bias=0.020,
        ))
        self.assertAlmostEqual(d.inputs_snapshot["lane_bias_max_abs"], 0.030, places=6)


# ============================================================
# 8. Snapshot is JSON-serializable
# ============================================================
class SnapshotJsonTests(unittest.TestCase):
    def test_inputs_snapshot_is_json_serializable(self) -> None:
        d = RoutingPolicyEngine().evaluate(_neutral_inputs(
            memory_lane_bias=0.025,
            confidence_margin=0.10,
        ))
        s = json.dumps(d.inputs_snapshot)
        rt = json.loads(s)
        self.assertIn("stability_score", rt)
        self.assertIn("would_upgrade_to_hybrid", rt)
        self.assertIsInstance(rt["would_upgrade_to_hybrid"], bool)


# ============================================================
# 9. Safe failure mode
# ============================================================
class SafeFailureModeTests(unittest.TestCase):
    def test_safe_failure_mode_returns_no_action(self) -> None:
        engine = RoutingPolicyEngine()

        def _boom(_inputs):
            raise RuntimeError("synthetic engine failure")

        engine._evaluate_inner = _boom  # type: ignore[assignment]
        with self.assertLogs("Qube.RoutingPolicyEngine", level="WARNING") as cm:
            d = engine.evaluate(_neutral_inputs())
        self.assertEqual(d.policy, POLICY_NO_ACTION)
        self.assertIsNone(d.reason)
        self.assertEqual(d.inputs_snapshot, {})
        joined = "\n".join(cm.output)
        self.assertIn("RuntimeError", joined)

    def test_safe_failure_mode_dedupes_warnings_per_type(self) -> None:
        engine = RoutingPolicyEngine()

        def _boom(_inputs):
            raise RuntimeError("repeat")

        engine._evaluate_inner = _boom  # type: ignore[assignment]
        # First call emits the WARNING.
        with self.assertLogs("Qube.RoutingPolicyEngine", level="WARNING"):
            engine.evaluate(_neutral_inputs())
        # Subsequent calls of the same exception type do NOT log.
        logger = logging.getLogger("Qube.RoutingPolicyEngine")
        with self.assertNoLogs(logger=logger.name, level="WARNING"):
            engine.evaluate(_neutral_inputs())
            engine.evaluate(_neutral_inputs())

    def test_evaluate_never_raises_even_under_extreme_inputs(self) -> None:
        engine = RoutingPolicyEngine()
        # Pathological inputs: NaN, infinities, empty strings, None
        # where typing says str. `evaluate` is contractually total.
        weird = RoutingPolicyInputs(
            confidence_margin=float("nan"),
            top_intent="",
            top_score=float("inf"),
            second_best_score=float("-inf"),
            second_best_intent=None,
            tier2_active=True,
            memory_lane_bias=float("nan"),
            rag_lane_bias=float("inf"),
            web_lane_bias=float("-inf"),
            tier3_band_active=True,
            tier4_active=True,
            tier4_cluster_size=0,
            tier4_cluster_dominant_frequency=float("nan"),
            tier4_cluster_oscillating=True,
        )
        # Must not raise.
        d = engine.evaluate(weird)
        self.assertIn(d.policy, POLICY_VALUES)


# ============================================================
# 10. Decision dataclass shape
# ============================================================
class DecisionShapeTests(unittest.TestCase):
    def test_policy_value_is_always_in_enum(self) -> None:
        engine = RoutingPolicyEngine()
        for inputs in (
            _neutral_inputs(),
            _neutral_inputs(memory_lane_bias=0.025, confidence_margin=0.10),
            _neutral_inputs(tier4_cluster_dominant_frequency=0.50,
                            confidence_margin=0.05),
        ):
            d = engine.evaluate(inputs)
            self.assertIn(d.policy, POLICY_VALUES)

    def test_decision_dataclass_is_frozen(self) -> None:
        d = RoutingPolicyDecision(POLICY_ACCEPT, None, {})
        with self.assertRaises(Exception):
            d.policy = POLICY_STABILIZE  # type: ignore[misc]

    def test_inputs_dataclass_is_frozen(self) -> None:
        i = _neutral_inputs()
        with self.assertRaises(Exception):
            i.confidence_margin = 0.0  # type: ignore[misc]
        # `replace` works for building variants in tests.
        i2 = replace(i, confidence_margin=0.05)
        self.assertEqual(i2.confidence_margin, 0.05)


if __name__ == "__main__":
    unittest.main()
