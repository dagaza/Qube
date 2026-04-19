"""Decision Trace Layer tests — observability-only.

Pins four guarantees:

  * **Contract** — ``decision["trace"]`` is always present with the
    documented top-level schema, all numeric leaves are finite plain
    Python floats, and the whole object is JSON-serialisable.
  * **Explanation correctness** — the ``winning_reason`` enum maps
    1:1 to the priority-tree branch (or Tier 2 post-tree modifier)
    that set the final route; ``losing_candidates[*].reason`` maps
    to the correct exclusion taxonomy.
  * **Edge cases** — NONE / HYBRID / recall-active / drift / no
    intent vector / Tier 4 dormant / Tier 5 failure / Tier 6
    failure all produce a sane trace.
  * **Route field never modified** — a 200-query randomised walk
    against a control router whose ``_build_decision_trace`` is
    stubbed to return ``{}`` produces byte-identical ``route``
    values. Mirrors the Tier 4/5/6 ``RouteFieldNeverModified``
    pattern.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mcp.cognitive_router import (
    AMBIGUITY_MARGIN,
    HIGH_CONFIDENCE_CEILING,
    MIN_CONFIDENCE_FLOOR,
    CognitiveRouterV4,
)
from mcp.router_lane_stats import (
    LANE_BIAS_DAMPING,
    RECENT_WINDOW_SIZE,
    RouteFeedbackEvent,
)
from mcp.routing_arbitration_layer import (
    CONFLICT_FLAGS_VALUES,
    INTERPRETATION_PASSTHROUGH,
    INTERPRETATION_STABLE,
    INTERPRETATION_VALUES,
)
from mcp.routing_policy_engine import (
    POLICY_ACCEPT,
    POLICY_NO_ACTION,
    POLICY_VALUES,
)


_NEUTRAL_QUERY = "the quick brown fox jumps over the lazy dog"


# Synthetic unit-length centroids in R^6 (mirror of Tier 2/3/4/5/6).
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


def _new_router() -> CognitiveRouterV4:
    r = CognitiveRouterV4()
    _install_tier2_centroids(r)
    return r


# Enum of every possible winning_reason string emitted by
# ``_build_decision_trace``. The ``*_unknown`` sentinels are defensive
# fallbacks; the taxonomy-is-total test asserts none of them leak.
_VALID_WINNING_REASONS = {
    "complexity_forced_hybrid",
    "internet_enabled",
    "recall_override_hybrid",
    "dual_threshold_hybrid",
    "single_rag",
    "single_memory",
    "no_lane_cleared_threshold",
    "confidence_floor_downgrade_to_none",
    "ambiguity_upgrade_to_hybrid",
    "hybrid_unknown",
    "unknown_route",
}

_VALID_LOSER_REASONS = {
    "below_threshold",
    "lower_score_than_winner",
    "hybrid_fused",
    "confidence_floor_downgraded_winner",
    "drift_suppressed",
    "blocked_by_chat_margin",
}

_TRACE_TOP_KEYS = (
    "selected_route",
    "winning_reason",
    "winning_signal",
    "losing_candidates",
    "confidence",
    "tier3",
    "tier4",
    "tier5_6",
    "context",
)


def _iter_numeric_leaves(obj, path=""):
    """Yield ``(path, value)`` for every numeric leaf. Booleans are
    excluded (``bool`` is a subclass of ``int`` in Python).
    """
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield path, obj
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _iter_numeric_leaves(v, f"{path}.{k}" if path else k)
        return
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _iter_numeric_leaves(v, f"{path}[{i}]")
        return


# ============================================================
# 1. Core behavior
# ============================================================
class TraceCoreBehaviorTests(unittest.TestCase):

    def test_trace_key_present_without_intent_vector(self) -> None:
        d = CognitiveRouterV4().route(_NEUTRAL_QUERY)
        self.assertIn("trace", d)
        self.assertIsInstance(d["trace"], dict)
        for k in _TRACE_TOP_KEYS:
            self.assertIn(k, d["trace"], f"missing top-level trace key {k!r}")

    def test_trace_key_present_with_intent_vector(self) -> None:
        router = _new_router()
        d = router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(m=0.5))
        self.assertIn("trace", d)
        for k in _TRACE_TOP_KEYS:
            self.assertIn(k, d["trace"])

    def test_selected_route_mirrors_decision_route(self) -> None:
        router = _new_router()
        d = router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(w=0.8))
        self.assertEqual(d["trace"]["selected_route"], d["route"])


class TraceDoesNotAffectRoute(unittest.TestCase):
    """200-query randomised walk vs a control router whose trace
    builder is stubbed to ``{}`` — ``route`` must be byte-identical
    for every turn. Mirrors the Tier 5/6 ``RouteFieldNeverModified``
    pattern.
    """

    @staticmethod
    def _make_control() -> CognitiveRouterV4:
        r = CognitiveRouterV4()
        _install_tier2_centroids(r)
        # Replace with a no-op so the trace value cannot possibly
        # influence the decision dict's ``route`` field.
        r._build_decision_trace = lambda **_kw: {}  # type: ignore[assignment]
        return r

    def test_random_walk_routes_byte_identical_against_null_trace(self) -> None:
        rng = random.Random(0xDECAF)
        live = _new_router()
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
                msg=(f"route diverged: live={d_live['route']} "
                     f"ctrl={d_ctrl['route']} (v={v})"),
            )

    def test_control_router_has_empty_trace(self) -> None:
        ctrl = self._make_control()
        d = ctrl.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(m=0.5))
        self.assertEqual(d["trace"], {})


# ============================================================
# 2. Explanation correctness
# ============================================================
class TraceExplanationCorrectnessTests(unittest.TestCase):

    def test_web_wins_due_to_higher_embedding_score(self) -> None:
        router = _new_router()
        d = router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(w=0.9))
        self.assertEqual(d["route"], "web")
        trace = d["trace"]
        self.assertEqual(trace["winning_reason"], "internet_enabled")
        self.assertEqual(trace["winning_signal"]["lane"], "web")
        self.assertEqual(trace["winning_signal"]["source"], "embedding")
        self.assertAlmostEqual(trace["winning_signal"]["score"], 0.9, places=4)
        self.assertEqual(trace["confidence"]["top_intent"], "web")
        self.assertEqual(trace["confidence"]["top_intent_source"], "embedding")

    def test_rag_loses_due_to_threshold(self) -> None:
        router = _new_router()
        # Memory wins cleanly; rag embedding score (0.20) is below the
        # base 0.25 rag threshold, so rag is ``below_threshold``.
        d = router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(m=0.6, r=0.20))
        self.assertEqual(d["route"], "memory")
        trace = d["trace"]
        self.assertEqual(trace["winning_reason"], "single_memory")
        by_lane = {c["lane"]: c for c in trace["losing_candidates"]}
        self.assertIn("rag", by_lane)
        self.assertEqual(by_lane["rag"]["reason"], "below_threshold")

    def test_memory_loses_due_to_lower_score_than_winner(self) -> None:
        """Web wins via internet-first priority; memory's embedding
        score (0.30) is ABOVE the base 0.25 memory threshold, so the
        loser reason must be ``lower_score_than_winner`` — NOT
        ``below_threshold``."""
        router = _new_router()
        d = router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(w=0.5, m=0.30))
        self.assertEqual(d["route"], "web")
        trace = d["trace"]
        self.assertEqual(trace["winning_reason"], "internet_enabled")
        by_lane = {c["lane"]: c for c in trace["losing_candidates"]}
        self.assertIn("memory", by_lane)
        self.assertEqual(by_lane["memory"]["reason"], "lower_score_than_winner")

    def test_winning_reason_taxonomy_is_total(self) -> None:
        """Drive a variety of scenarios and assert EVERY trace's
        ``winning_reason`` is one of the documented strings and that
        no defensive sentinel (``*_unknown``) is emitted."""
        scenarios = [
            ("none",            {"intent_vector": None}),
            ("web-embedding",   {"intent_vector": _intent_vec(w=0.9)}),
            ("single-memory",   {"intent_vector": _intent_vec(m=0.6)}),
            ("single-rag",      {"intent_vector": _intent_vec(r=0.6)}),
            ("dual-hybrid",     {"intent_vector": _intent_vec(m=0.5, r=0.5)}),
            ("complexity",      {"intent_vector": _intent_vec(m=0.5),
                                 "estimated_complexity": 0.85}),
        ]
        reasons_seen = set()
        for label, kwargs in scenarios:
            router = _new_router()
            d = router.route(_NEUTRAL_QUERY, **kwargs)
            reason = d["trace"]["winning_reason"]
            self.assertIn(
                reason, _VALID_WINNING_REASONS,
                msg=f"scenario={label}: reason={reason!r} not in enum",
            )
            self.assertNotIn("unknown", reason, msg=f"defensive sentinel fired for {label}")
            reasons_seen.add(reason)
        # Sanity: we hit at least a spread of reasons.
        self.assertGreaterEqual(len(reasons_seen), 4)


# ============================================================
# 3. Edge cases
# ============================================================
class TraceEdgeCaseTests(unittest.TestCase):

    def test_none_route_explains_failure(self) -> None:
        """Neutral query, no centroids, no intent vector → every lane
        below threshold; winning_reason is the base-case failure and
        every retrieval lane is listed with ``below_threshold``."""
        d = CognitiveRouterV4().route(_NEUTRAL_QUERY)
        self.assertEqual(d["route"], "none")
        trace = d["trace"]
        self.assertEqual(trace["winning_reason"], "no_lane_cleared_threshold")
        self.assertFalse(trace["confidence"]["floor_applied"])
        self.assertFalse(trace["confidence"]["ambiguity_applied"])
        retrieval_reasons = {
            c["lane"]: c["reason"]
            for c in trace["losing_candidates"]
            if c["lane"] in ("memory", "rag", "web")
        }
        for lane in ("memory", "rag", "web"):
            self.assertEqual(retrieval_reasons.get(lane), "below_threshold")

    def test_none_route_via_confidence_floor_downgrade(self) -> None:
        """Embedding-only memory signal at 0.28 — above the 0.25
        memory threshold but below the 0.30 floor → Tier 2 floor
        block fires; winning_reason must be the downgrade string."""
        router = _new_router()
        d = router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(m=0.28))
        self.assertEqual(d["route"], "none")
        trace = d["trace"]
        self.assertEqual(trace["winning_reason"], "confidence_floor_downgrade_to_none")
        self.assertTrue(trace["confidence"]["floor_applied"])
        self.assertFalse(trace["confidence"]["ambiguity_applied"])
        # The winning "lane" for a downgrade-to-none is "none", with
        # top_intent's numbers surfaced in the signal block.
        self.assertEqual(trace["winning_signal"]["lane"], "none")
        self.assertEqual(trace["confidence"]["top_intent"], "memory")
        # Memory is the lane that WOULD have won → special loser reason.
        by_lane = {c["lane"]: c for c in trace["losing_candidates"]}
        self.assertEqual(by_lane["memory"]["reason"],
                         "confidence_floor_downgraded_winner")

    def test_hybrid_explains_dual_threshold(self) -> None:
        router = _new_router()
        d = router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(m=0.5, r=0.5))
        self.assertEqual(d["route"], "hybrid")
        trace = d["trace"]
        self.assertEqual(trace["winning_reason"], "dual_threshold_hybrid")
        # Both fused lanes report ``hybrid_fused``.
        by_lane = {c["lane"]: c for c in trace["losing_candidates"]}
        self.assertEqual(by_lane["memory"]["reason"], "hybrid_fused")
        self.assertEqual(by_lane["rag"]["reason"], "hybrid_fused")

    def test_hybrid_explains_ambiguity_upgrade(self) -> None:
        """Push the base rag threshold to 0.40 so a rag score of 0.39
        stays below-threshold (rag NOT enabled), but still above the
        0.30 floor and within 0.10 of the memory top. The Tier 2
        ambiguity block then upgrades memory → hybrid."""
        router = _new_router()
        router.base_rag_threshold = 0.40
        d = router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(m=0.45, r=0.39))
        self.assertEqual(d["route"], "hybrid")
        trace = d["trace"]
        self.assertEqual(trace["winning_reason"], "ambiguity_upgrade_to_hybrid")
        self.assertTrue(trace["confidence"]["ambiguity_applied"])
        self.assertFalse(trace["confidence"]["floor_applied"])
        self.assertLess(trace["confidence"]["margin"], AMBIGUITY_MARGIN)
        by_lane = {c["lane"]: c for c in trace["losing_candidates"]}
        self.assertEqual(by_lane["memory"]["reason"], "hybrid_fused")
        self.assertEqual(by_lane["rag"]["reason"], "hybrid_fused")

    def test_recall_active_produces_valid_trace(self) -> None:
        """No recall/chat centroid installed → substring fallback sets
        recall_score=1.0 ≥ 0.62; recall_active=True forces hybrid."""
        router = _new_router()
        d = router.route("tell me about Plato", intent_vector=_intent_vec(m=0.2))
        self.assertTrue(d["recall_active"])
        self.assertEqual(d["route"], "hybrid")
        trace = d["trace"]
        self.assertEqual(trace["winning_reason"], "recall_override_hybrid")
        self.assertEqual(trace["winning_signal"]["lane"], "recall")
        self.assertEqual(trace["winning_signal"]["source"], "substring")
        self.assertAlmostEqual(
            trace["winning_signal"]["threshold"], router.recall_threshold,
        )
        # Recall is the winner → excluded from losing_candidates.
        lanes = {c["lane"] for c in trace["losing_candidates"]}
        self.assertNotIn("recall", lanes)

    def test_no_intent_vector_produces_valid_trace(self) -> None:
        router = _new_router()
        d = router.route(_NEUTRAL_QUERY, intent_vector=None)
        trace = d["trace"]
        self.assertFalse(trace["context"]["intent_vector_present"])
        self.assertFalse(trace["tier4"]["active"])
        # All numeric leaves finite.
        for path, val in _iter_numeric_leaves(trace):
            self.assertTrue(
                math.isfinite(val),
                msg=f"non-finite numeric at {path}: {val!r}",
            )

    def test_drift_explains_lane_suppression(self) -> None:
        router = _new_router()
        # Force drift globally via monkeypatch — this is the Tier-1
        # ``drift`` flag consumed by ``route()``.
        router._detect_intent_drift = lambda _v: True  # type: ignore[assignment]
        d = router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(m=0.6, r=0.6))
        self.assertTrue(d["drift"])
        self.assertEqual(d["route"], "none")
        trace = d["trace"]
        retrieval_losers = {
            c["lane"]: c["reason"]
            for c in trace["losing_candidates"]
            if c["lane"] in ("memory", "rag", "web")
        }
        for lane in ("memory", "rag", "web"):
            self.assertEqual(retrieval_losers.get(lane), "drift_suppressed",
                             msg=f"lane={lane}")

    def test_tier4_dormant_trace_has_none_cluster_fields(self) -> None:
        d = CognitiveRouterV4().route(_NEUTRAL_QUERY)
        t4 = d["trace"]["tier4"]
        self.assertFalse(t4["active"])
        self.assertIsNone(t4["cluster_id"])
        self.assertIsNone(t4["cluster_dominant_route"])
        self.assertIsNone(t4["cluster_dominant_frequency"])
        self.assertEqual(t4["cluster_size"], 0)

    def test_tier5_failure_trace_is_safe(self) -> None:
        class _Boom:
            def evaluate(self, *_a, **_kw):
                raise RuntimeError("synthetic tier5 failure")

        router = _new_router()
        router.policy_engine = _Boom()  # type: ignore[assignment]
        d = router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(m=0.5))
        trace = d["trace"]
        self.assertFalse(trace["tier5_6"]["tier5_active"])
        self.assertEqual(trace["tier5_6"]["policy"], POLICY_NO_ACTION)

    def test_tier6_failure_trace_is_safe(self) -> None:
        class _Boom:
            def evaluate(self, *_a, **_kw):
                raise RuntimeError("synthetic tier6 failure")

        router = _new_router()
        router.arbitration_layer = _Boom()  # type: ignore[assignment]
        d = router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(m=0.5))
        trace = d["trace"]
        self.assertFalse(trace["tier5_6"]["tier6_active"])
        self.assertEqual(trace["tier5_6"]["interpretation"],
                         INTERPRETATION_PASSTHROUGH)
        self.assertEqual(trace["tier5_6"]["conflicts"], [])


# ============================================================
# 4. Contract
# ============================================================
class TraceContractTests(unittest.TestCase):

    def test_required_keys_always_present_over_random_walk(self) -> None:
        rng = random.Random(0xBEEF)
        router = _new_router()
        amp_choices = (0.0, 0.10, 0.30, 0.50, 0.75, 0.92)
        axis_choices = ("m", "r", "w", None)
        for _ in range(50):
            axis = rng.choice(axis_choices)
            if axis is None:
                v = None
            else:
                v = _intent_vec(**{axis: rng.choice(amp_choices)})
            trace = router.route(_NEUTRAL_QUERY, intent_vector=v)["trace"]
            for k in _TRACE_TOP_KEYS:
                self.assertIn(k, trace)

    def test_numeric_fields_are_python_floats_not_nan(self) -> None:
        """Every numeric leaf must be a plain Python number (``int``
        or ``float`` — NumPy scalars are forbidden) and finite.
        Booleans are skipped by ``_iter_numeric_leaves``. ``None``
        is permitted for Tier 4 dormant optionals and never surfaces
        here because ``_iter_numeric_leaves`` skips it.
        ``cluster_id`` / ``cluster_size`` are legitimately ``int``;
        everything else is expected to be ``float``."""
        router = _new_router()
        d = router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(m=0.5, r=0.5))
        _int_ok_paths = {"tier4.cluster_id", "tier4.cluster_size"}
        for path, val in _iter_numeric_leaves(d["trace"]):
            # NumPy scalars are forbidden — they break ``json.dumps``.
            self.assertIs(
                type(val) in (int, float), True,
                msg=(f"{path}: expected plain Python int/float, "
                     f"got {type(val).__name__} ({val!r})"),
            )
            if path not in _int_ok_paths:
                self.assertIsInstance(
                    val, float,
                    msg=f"{path}: expected float, got {type(val).__name__}",
                )
            self.assertTrue(
                math.isfinite(val),
                msg=f"{path}: non-finite value {val!r}",
            )

    def test_types_are_correct(self) -> None:
        router = _new_router()
        d = router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(w=0.8))
        trace = d["trace"]
        self.assertIsInstance(trace["selected_route"], str)
        self.assertIn(trace["winning_reason"], _VALID_WINNING_REASONS)

        ws = trace["winning_signal"]
        self.assertIsInstance(ws, dict)
        self.assertIsInstance(ws["lane"], str)
        self.assertIsInstance(ws["score"], float)
        self.assertIsInstance(ws["threshold"], float)
        self.assertIn(ws["source"], ("embedding", "substring"))

        self.assertIsInstance(trace["losing_candidates"], list)
        for cand in trace["losing_candidates"]:
            self.assertIsInstance(cand, dict)
            self.assertIsInstance(cand["lane"], str)
            self.assertIsInstance(cand["score"], float)
            self.assertIsInstance(cand["threshold"], float)
            self.assertIn(cand["reason"], _VALID_LOSER_REASONS,
                          msg=f"lane={cand['lane']} reason={cand['reason']!r}")

        conf = trace["confidence"]
        self.assertIsInstance(conf["top_intent"], str)
        self.assertIsInstance(conf["top_intent_source"], str)
        self.assertIsInstance(conf["floor_applied"], bool)
        self.assertIsInstance(conf["ambiguity_applied"], bool)
        self.assertIsInstance(conf["tier2_active"], bool)
        self.assertEqual(conf["floor"], MIN_CONFIDENCE_FLOOR)
        self.assertEqual(conf["ambiguity_margin"], AMBIGUITY_MARGIN)

        t3 = trace["tier3"]
        self.assertIsInstance(t3["band_active"], bool)
        self.assertEqual(t3["high_confidence_ceiling"], HIGH_CONFIDENCE_CEILING)
        self.assertEqual(t3["damping"], LANE_BIAS_DAMPING)
        self.assertIsInstance(t3["lane_bias"], dict)
        for lane in ("memory", "rag", "web"):
            self.assertIn(lane, t3["lane_bias"])
            self.assertIsInstance(t3["lane_bias"][lane], float)

        t56 = trace["tier5_6"]
        self.assertIsInstance(t56["tier5_active"], bool)
        self.assertIn(t56["policy"], POLICY_VALUES)
        self.assertIsInstance(t56["tier6_active"], bool)
        self.assertIsInstance(t56["conflicts"], list)
        for flag in t56["conflicts"]:
            self.assertIn(flag, CONFLICT_FLAGS_VALUES)
        self.assertIn(t56["interpretation"], INTERPRETATION_VALUES)

        ctx = trace["context"]
        self.assertIsInstance(ctx["drift"], bool)
        self.assertIsInstance(ctx["recall_active"], bool)
        self.assertIsInstance(ctx["complexity_forced"], bool)
        self.assertIsInstance(ctx["rag_penalty"], float)
        self.assertIsInstance(ctx["intent_vector_present"], bool)

    def test_trace_is_json_serializable(self) -> None:
        router = _new_router()
        # Saturate some feedback so lane_bias is non-zero; install a
        # Tier 4 cluster so the dormant-optional path is exercised too.
        for _ in range(RECENT_WINDOW_SIZE):
            router.observe_feedback(RouteFeedbackEvent(
                route="memory",
                top_intent="memory",
                top_source="substring",
                confidence_margin=0.20,
                latency_ms=50.0,
                success=True,
                drift=False,
                per_lane_hits={},
            ))
        v = _intent_vec(m=0.5)
        for _ in range(5):
            router.stability_tracker.observe(v, "memory")
        d = router.route(_NEUTRAL_QUERY, intent_vector=v)
        encoded = json.dumps(d["trace"])
        self.assertIsInstance(encoded, str)
        # Round-trip equality on the trace.
        self.assertEqual(json.loads(encoded), d["trace"])

    def test_tier6_and_prior_tier_keys_still_present(self) -> None:
        """Schema-additivity regression: the trace must not replace
        or shadow any prior-tier key on the top-level decision dict."""
        router = _new_router()
        d = router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(m=0.5))
        for k in (
            "route",                    # Tier 1
            "memory_score_final",       # Tier 2
            "memory_lane_bias",         # Tier 3
            "tier3_band_active",        # Tier 3 refinement
            "tier4_active",             # Tier 4
            "tier5_policy",             # Tier 5
            "tier6_interpretation",     # Tier 6
            "trace",                    # this layer
            "strategy",                 # canonical tail
        ):
            self.assertIn(k, d, msg=f"decision dict missing key {k!r}")
        # Canonical tail ordering: ``trace`` lives right before
        # ``strategy`` by construction.
        keys = list(d.keys())
        self.assertEqual(keys[-2:], ["trace", "strategy"])


# ============================================================
# 5. Additional smoke — Tier 5 neutral policy for a clean turn
# ============================================================
class TraceSmokeTests(unittest.TestCase):

    def test_clean_turn_policy_is_accept(self) -> None:
        router = _new_router()
        d = router.route(_NEUTRAL_QUERY, intent_vector=_intent_vec(m=0.5))
        self.assertEqual(d["trace"]["tier5_6"]["policy"], POLICY_ACCEPT)
        self.assertEqual(d["trace"]["tier5_6"]["interpretation"],
                         INTERPRETATION_STABLE)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
