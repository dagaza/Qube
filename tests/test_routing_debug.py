"""Tests for mcp/routing_debug observability helpers."""

from __future__ import annotations

import copy
import dataclasses
import json
import threading
import unittest
from typing import Any

from mcp.cognitive_router import AMBIGUITY_MARGIN, MIN_CONFIDENCE_FLOOR
from mcp.routing_debug import (
    MAX_RECORDS,
    RoutingDebugBuffer,
    RoutingDebugRecord,
    build_record,
    build_route_summary,
    synthesize_trace_stub,
)


def _base_trace(**overrides: Any) -> dict[str, Any]:
    t: dict[str, Any] = {
        "selected_route": "rag",
        "winning_reason": "single_rag",
        "winning_signal": {
            "lane": "rag",
            "score": 0.5,
            "threshold": 0.3,
            "source": "embedding",
        },
        "losing_candidates": [],
        "confidence": {
            "top_intent": "rag",
            "top_intent_source": "embedding",
            "top_score": 0.55,
            "second_best_score": 0.2,
            "margin": 0.15,
            "floor": float(MIN_CONFIDENCE_FLOOR),
            "ambiguity_margin": float(AMBIGUITY_MARGIN),
            "floor_applied": False,
            "ambiguity_applied": False,
            "tier2_active": True,
        },
        "tier3": {
            "band_active": False,
            "high_confidence_ceiling": 0.0,
            "damping": 0.0,
            "lane_bias": {"memory": 0.0, "rag": 0.0, "web": 0.0},
        },
        "tier4": {
            "active": False,
            "cluster_id": None,
            "cluster_size": 0,
            "cluster_dominant_route": None,
            "cluster_dominant_frequency": None,
            "cluster_oscillating": False,
        },
        "tier5_6": {
            "tier5_active": True,
            "policy": "accept",
            "policy_reason": None,
            "tier6_active": True,
            "conflicts": [],
            "interpretation": "stable",
        },
        "context": {},
    }
    t.update(overrides)
    return t


class RoutingDebugBufferTests(unittest.TestCase):
    def test_capacity_fifo(self) -> None:
        buf = RoutingDebugBuffer(maxlen=3)
        for i in range(5):
            buf.append(
                RoutingDebugRecord(
                    timestamp=float(i),
                    session_id="s",
                    turn_id=i,
                    query=str(i),
                    route="none",
                    route_pre_policy="none",
                    strategy="x",
                    trace_level="minimal",
                    top_intent=None,
                    top_score=None,
                    summary="",
                    trace={},
                    decision={},
                )
            )
        snap = buf.snapshot()
        self.assertEqual(len(snap), 3)
        self.assertEqual(snap[0].turn_id, 2)
        self.assertEqual(snap[-1].turn_id, 4)

    def test_snapshot_is_copy(self) -> None:
        buf = RoutingDebugBuffer()
        buf.append(
            RoutingDebugRecord(
                timestamp=1.0,
                session_id=None,
                turn_id=1,
                query="q",
                route="web",
                route_pre_policy="web",
                strategy="adaptive_v4",
                trace_level="full",
                top_intent="web",
                top_score=0.5,
                summary="s",
                trace={},
                decision={},
            )
        )
        s1 = buf.snapshot()
        s2 = buf.snapshot()
        self.assertIsNot(s1, s2)
        self.assertEqual(len(s1), 1)

    def test_threaded_append(self) -> None:
        buf = RoutingDebugBuffer(maxlen=200)

        def worker(start: int) -> None:
            for i in range(50):
                n = start + i
                buf.append(
                    RoutingDebugRecord(
                        timestamp=float(n),
                        session_id=None,
                        turn_id=n,
                        query=str(n),
                        route="none",
                        route_pre_policy="none",
                        strategy="x",
                        trace_level="minimal",
                        top_intent=None,
                        top_score=None,
                        summary="",
                        trace={},
                        decision={},
                    )
                )

        threads = [threading.Thread(target=worker, args=(i * 100,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(buf.snapshot()), 200)


class SynthesizeStubTests(unittest.TestCase):
    def test_shape_parity(self) -> None:
        d = {"route": "rag", "strategy": "explicit_file_search"}
        stub = synthesize_trace_stub(d)
        for key in (
            "selected_route",
            "winning_reason",
            "winning_signal",
            "losing_candidates",
            "confidence",
            "tier3",
            "tier4",
            "tier5_6",
            "context",
        ):
            self.assertIn(key, stub)
        self.assertEqual(stub["winning_reason"], "explicit_file_search")


class BuildRouteSummaryTests(unittest.TestCase):
    def _d(
        self,
        route: str,
        trace: dict[str, Any],
        **fields: Any,
    ) -> dict[str, Any]:
        base = {
            "route": route,
            "strategy": "adaptive_v4",
            "trace": trace,
            "top_intent": trace.get("confidence", {}).get("top_intent", "rag"),
            "top_score": trace.get("confidence", {}).get("top_score", 0.5),
            "confidence_margin": trace.get("confidence", {}).get("margin", 0.1),
            "second_best_score": 0.2,
            "internet_threshold": 0.25,
            "recall_score": 0.8,
            "recall_threshold": 0.35,
        }
        base.update(fields)
        return base

    def test_all_winning_reasons_non_empty(self) -> None:
        cases = [
            ("internet_enabled", _base_trace(winning_reason="internet_enabled", selected_route="web")),
            ("single_memory", _base_trace(winning_reason="single_memory", selected_route="memory")),
            ("single_rag", _base_trace(winning_reason="single_rag")),
            ("dual_threshold_hybrid", _base_trace(winning_reason="dual_threshold_hybrid", selected_route="hybrid")),
            ("ambiguity_upgrade_to_hybrid", _base_trace(
                winning_reason="ambiguity_upgrade_to_hybrid",
                selected_route="hybrid",
                losing_candidates=[
                    {"lane": "memory", "score": 0.4, "threshold": 0.3, "reason": "x"},
                    {"lane": "rag", "score": 0.45, "threshold": 0.3, "reason": "y"},
                ],
            )),
            ("recall_override_hybrid", _base_trace(winning_reason="recall_override_hybrid", selected_route="hybrid")),
            ("complexity_forced_hybrid", _base_trace(winning_reason="complexity_forced_hybrid", selected_route="hybrid")),
            ("confidence_floor_downgrade_to_none", _base_trace(
                winning_reason="confidence_floor_downgrade_to_none",
                selected_route="none",
            )),
            ("no_lane_cleared_threshold", _base_trace(
                winning_reason="no_lane_cleared_threshold",
                selected_route="none",
            )),
            ("hybrid_unknown", _base_trace(winning_reason="hybrid_unknown", selected_route="hybrid")),
            ("unknown_route", _base_trace(winning_reason="unknown_route", selected_route="weird")),
        ]
        for _name, tr in cases:
            s = build_route_summary(self._d(tr.get("selected_route", "none"), tr))
            self.assertTrue(s.strip(), msg=s)

    def test_override_strategies(self) -> None:
        for strat, needle in (
            ("explicit_remember", "Explicit remember"),
            ("explicit_file_search", "File-search"),
            ("narrative_recap", "Narrative recap"),
            ("fallback", "fallback"),
        ):
            d = {"route": "none" if strat != "explicit_file_search" else "rag", "strategy": strat}
            s = build_route_summary(d)
            self.assertIn(needle, s)

    def test_defensive_malformed(self) -> None:
        s = build_route_summary({"route": "web", "strategy": "adaptive_v4", "trace": {"winning_reason": None}})
        self.assertTrue(len(s) > 0)

    def test_asdict_json_serializable(self) -> None:
        d = self._d("rag", _base_trace())
        rec = build_record(query="hi", decision=d, session_id="1", turn_id=1)
        payload = dataclasses.asdict(rec)
        json.dumps(payload, default=str)


class BuildRecordTests(unittest.TestCase):
    def test_trace_passthrough_vs_stub(self) -> None:
        full = {"route": "rag", "strategy": "adaptive_v4", "trace": _base_trace(), "top_intent": "rag", "top_score": 0.55}
        r1 = build_record(query="q", decision=full, session_id=None, turn_id=1)
        self.assertEqual(r1.trace_level, "full")

        minimal = {"route": "rag", "strategy": "explicit_file_search"}
        r2 = build_record(query="q", decision=minimal, session_id=None, turn_id=2)
        self.assertEqual(r2.trace_level, "minimal")
        self.assertEqual(r2.trace.get("winning_reason"), "explicit_file_search")

    def test_effective_route(self) -> None:
        d = {"route": "memory", "strategy": "adaptive_v4", "trace": _base_trace(selected_route="memory")}
        r = build_record(
            query="q",
            decision=d,
            session_id=None,
            turn_id=1,
            effective_route="web",
        )
        self.assertEqual(r.route, "web")
        self.assertEqual(r.route_pre_policy, "memory")

    def test_shallow_trace_copy_top_level(self) -> None:
        tr = _base_trace()
        d = {"route": "rag", "strategy": "adaptive_v4", "trace": tr, "top_intent": "rag", "top_score": 0.5}
        r = build_record(query="q", decision=d, session_id=None, turn_id=1)
        r.trace["extra_top"] = 1
        self.assertNotIn("extra_top", d["trace"])

    def test_decision_deepcopy_isolation(self) -> None:
        d = {"route": "rag", "strategy": "adaptive_v4", "trace": _base_trace(), "top_intent": "rag", "top_score": 0.5}
        r = build_record(query="q", decision=d, session_id=None, turn_id=1)
        r.decision["mutated"] = True
        self.assertNotIn("mutated", d)

    def test_routing_unaffected_decision(self) -> None:
        d = {"route": "rag", "strategy": "adaptive_v4", "trace": _base_trace(), "top_intent": "rag", "top_score": 0.5}
        before = copy.deepcopy(d)
        _ = build_record(query="q", decision=d, session_id=None, turn_id=1, effective_route="hybrid")
        self.assertEqual(d, before)


class MaxRecordsConstantTests(unittest.TestCase):
    def test_max_records(self) -> None:
        self.assertEqual(MAX_RECORDS, 100)


if __name__ == "__main__":
    unittest.main()
