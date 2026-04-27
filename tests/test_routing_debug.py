"""Tests for mcp/routing_debug observability helpers."""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import threading
import logging
import unittest
from typing import Any
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from mcp.cognitive_router import AMBIGUITY_MARGIN, MIN_CONFIDENCE_FLOOR
from mcp.routing_debug import (
    MAX_RECORDS,
    RoutingDebugBuffer,
    RoutingDebugRecord,
    build_chat_contract_trace,
    build_engine_input_trace,
    build_model_router_trace,
    build_record,
    build_route_summary,
    routing_debug_log_enabled,
    routing_debug_log_redact_query,
    routing_debug_log_verbose,
    serialize_record_for_log,
    synthesize_trace_stub,
)
from core.routing_debug_sink import (
    ROUTING_DEBUG_LOGGER_NAME,
    attach_routing_debug_file_sink,
    detach_routing_debug_file_sink_for_tests,
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


class RoutingDebugSerializeForLogTests(unittest.TestCase):
    def _record(self) -> RoutingDebugRecord:
        trace = _base_trace()
        trace["model_router"] = {"selected_model": "m.gguf"}
        trace["chat_contract"] = {"format": "chatml"}
        trace["engine_input_trace"] = {"trace_id": 42}
        return RoutingDebugRecord(
            timestamp=123.0,
            session_id="s1",
            turn_id=9,
            query="who is john?",
            route="hybrid",
            route_pre_policy="memory",
            strategy="adaptive_v4",
            trace_level="full",
            top_intent="memory",
            top_score=0.77,
            summary="HYBRID",
            trace=trace,
            decision={"route": "hybrid", "trace": trace},
        )

    def test_compact_schema_and_optional_blocks(self) -> None:
        payload = serialize_record_for_log(self._record(), verbose=False, redact_query=False)
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["session_id"], "s1")
        self.assertIn("tier3_lane_bias", payload)
        self.assertIn("model_router", payload)
        self.assertIn("chat_contract", payload)
        self.assertIn("engine_input_trace", payload)
        self.assertNotIn("trace", payload)
        self.assertNotIn("decision", payload)

    def test_verbose_includes_full_payloads(self) -> None:
        payload = serialize_record_for_log(self._record(), verbose=True, redact_query=False)
        self.assertIn("trace", payload)
        self.assertIn("decision", payload)

    def test_redacted_query(self) -> None:
        payload = serialize_record_for_log(self._record(), verbose=False, redact_query=True)
        self.assertTrue(str(payload["query"]).startswith("[redacted sha256:"))
        self.assertNotEqual(payload["query"], "who is john?")


class RoutingDebugEnvFlagsTests(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_flags_default_off(self) -> None:
        self.assertFalse(routing_debug_log_enabled())
        self.assertFalse(routing_debug_log_verbose())
        self.assertFalse(routing_debug_log_redact_query())

    @patch.dict(
        os.environ,
        {
            "QUBE_ROUTING_DEBUG_LOG": "1",
            "QUBE_ROUTING_DEBUG_LOG_VERBOSE": "true",
            "QUBE_ROUTING_DEBUG_LOG_REDACT_QUERY": "yes",
        },
        clear=True,
    )
    def test_flags_parse_on_values(self) -> None:
        self.assertTrue(routing_debug_log_enabled())
        self.assertTrue(routing_debug_log_verbose())
        self.assertTrue(routing_debug_log_redact_query())


class RoutingDebugSinkTests(unittest.TestCase):
    def tearDown(self) -> None:
        detach_routing_debug_file_sink_for_tests()
        lg = logging.getLogger(ROUTING_DEBUG_LOGGER_NAME)
        lg.propagate = True

    def test_sink_writes_file(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "routing_debug.log"
            attach_routing_debug_file_sink(log_path=path, max_bytes=2048, backup_count=1)
            lg = logging.getLogger(ROUTING_DEBUG_LOGGER_NAME)
            lg.setLevel(logging.INFO)
            lg.propagate = False
            lg.info('{"schema_version":1,"route":"hybrid"}')
            for h in lg.handlers:
                try:
                    h.flush()
                except Exception:
                    pass
            self.assertTrue(path.is_file())
            text = path.read_text(encoding="utf-8", errors="replace")
            self.assertIn('"route":"hybrid"', text)


class MaxRecordsConstantTests(unittest.TestCase):
    def test_max_records(self) -> None:
        self.assertEqual(MAX_RECORDS, 100)


class ModelRouterTraceTests(unittest.TestCase):
    def test_build_model_router_trace_none_without_engine(self) -> None:
        self.assertIsNone(build_model_router_trace(None))

    def test_build_model_router_trace_shape(self) -> None:
        class _Eng:
            def get_model_reasoning_telemetry(self) -> dict[str, Any]:
                return {
                    "router_selected_model": "mistral.gguf",
                    "router_confidence": 0.87,
                    "router_task": "coding",
                    "router_scores": {"mistral.gguf": 2.1, "phi.gguf": 1.0},
                    "router_reasoning": ["matched_task=coding", "runner_up=phi"],
                    "model_basename": "mistral.gguf",
                    "model_name": "mistral",
                }

        mr = build_model_router_trace(_Eng())
        assert mr is not None
        self.assertEqual(mr["selected_model"], "mistral.gguf")
        self.assertEqual(mr["confidence"], 0.87)
        self.assertIn("phi.gguf", mr["alternatives"])
        self.assertTrue(any("coding" in x for x in mr["reasons"]))
        self.assertIn("signals", mr)
        self.assertIn("performance", mr)

    def test_backward_compat_no_model_router_until_merged(self) -> None:
        buf = RoutingDebugBuffer()
        buf.append(
            RoutingDebugRecord(
                timestamp=1.0,
                session_id=None,
                turn_id=1,
                query="q",
                route="none",
                route_pre_policy="none",
                strategy="adaptive_v4",
                trace_level="full",
                top_intent=None,
                top_score=None,
                summary="s",
                trace=_base_trace(),
                decision={"route": "none", "strategy": "adaptive_v4", "trace": _base_trace()},
            )
        )
        self.assertIsNone(buf.merge_model_router_into_latest(None))
        self.assertNotIn("model_router", buf.snapshot()[-1].trace)

    def test_merge_model_router_into_latest(self) -> None:
        buf = RoutingDebugBuffer()
        self.assertIsNone(buf.merge_model_router_into_latest(None))
        rec0 = RoutingDebugRecord(
            timestamp=1.0,
            session_id="s",
            turn_id=7,
            query="q",
            route="chat",
            route_pre_policy="chat",
            strategy="adaptive_v4",
            trace_level="full",
            top_intent=None,
            top_score=None,
            summary="sum",
            trace={"winning_reason": "x"},
            decision={},
        )
        buf.append(rec0)
        patch = {
            "selected_model": "m.gguf",
            "alternatives": ["a.gguf"],
            "reasons": ["r1"],
            "signals": {},
            "performance": {},
            "confidence": 0.5,
        }
        out = buf.merge_model_router_into_latest(patch)
        assert out is not None
        snap = buf.snapshot()
        self.assertEqual(snap[-1].trace.get("model_router"), patch)
        self.assertEqual(snap[-1].turn_id, 7)


class ChatContractTraceTests(unittest.TestCase):
    def test_build_chat_contract_trace_none_without_engine(self) -> None:
        self.assertIsNone(build_chat_contract_trace(None))

    def test_build_chat_contract_trace_shape(self) -> None:
        class _Eng:
            def get_model_reasoning_telemetry(self) -> dict[str, Any]:
                return {
                    "chat_contract": {
                        "model": "m.gguf",
                        "format": "chatml",
                        "source": "fallback",
                        "locked": True,
                    }
                }

        cc = build_chat_contract_trace(_Eng())
        assert cc is not None
        self.assertEqual(cc["format"], "chatml")
        self.assertEqual(cc["model"], "m.gguf")
        self.assertTrue(cc["locked"])

    def test_build_chat_contract_trace_includes_template_safety_only(self) -> None:
        class _Eng:
            def get_model_reasoning_telemetry(self) -> dict[str, Any]:
                return {
                    "chat_contract": {
                        "template_safety": {"unsafe": True, "reasons": ["contains <|channel|>"]},
                    }
                }

        cc = build_chat_contract_trace(_Eng())
        assert cc is not None
        self.assertEqual(cc.get("template_safety", {}).get("unsafe"), True)

    def test_merge_chat_contract_into_latest(self) -> None:
        buf = RoutingDebugBuffer()
        self.assertIsNone(buf.merge_chat_contract_into_latest(None))
        rec0 = RoutingDebugRecord(
            timestamp=1.0,
            session_id="s",
            turn_id=3,
            query="q",
            route="chat",
            route_pre_policy="chat",
            strategy="adaptive_v4",
            trace_level="full",
            top_intent=None,
            top_score=None,
            summary="sum",
            trace={"winning_reason": "x"},
            decision={},
        )
        buf.append(rec0)
        patch = {"model": "x.gguf", "format": "mistral-instruct", "source": "gguf", "locked": True}
        out = buf.merge_chat_contract_into_latest(patch)
        assert out is not None
        snap = buf.snapshot()
        self.assertEqual(snap[-1].trace.get("chat_contract"), patch)
        self.assertEqual(snap[-1].turn_id, 3)

    def test_merge_model_router_then_chat_contract_preserves_both(self) -> None:
        buf = RoutingDebugBuffer()
        buf.append(
            RoutingDebugRecord(
                timestamp=1.0,
                session_id=None,
                turn_id=9,
                query="q",
                route="none",
                route_pre_policy="none",
                strategy="adaptive_v4",
                trace_level="full",
                top_intent=None,
                top_score=None,
                summary="s",
                trace=_base_trace(),
                decision={},
            )
        )
        mr = {
            "selected_model": "a.gguf",
            "alternatives": [],
            "reasons": [],
            "signals": {},
            "performance": {},
            "confidence": 0.1,
        }
        cc = {"model": "a.gguf", "format": "chatml", "source": "fallback", "locked": True}
        buf.merge_model_router_into_latest(mr)
        buf.merge_chat_contract_into_latest(cc)
        last = buf.snapshot()[-1].trace
        self.assertEqual(last.get("model_router"), mr)
        self.assertEqual(last.get("chat_contract"), cc)


class EngineInputTraceRoutingTests(unittest.TestCase):
    def tearDown(self) -> None:
        from core.engine_input_trace import EngineInputTracer

        EngineInputTracer._instance = None  # type: ignore[attr-defined]

    @patch.dict(os.environ, {"QUBE_ENGINE_INPUT_TRACE": "0"})
    def test_build_engine_input_trace_none_when_flag_off(self) -> None:
        self.assertIsNone(build_engine_input_trace(None))

    @patch.dict(os.environ, {"QUBE_ENGINE_INPUT_TRACE": "1"})
    def test_build_engine_input_trace_shape(self) -> None:
        from core.engine_input_trace import EngineInputTrace, EngineInputTracer

        EngineInputTracer().log(
            EngineInputTrace(
                model_name="x.gguf",
                timestamp=42.0,
                input_mode="completion",
                messages=[{"role": "user", "content": "hi"}],
                prompt="formatted",
                serialized_input="formatted",
                chat_format="chatml",
                stop_tokens=["</s>"],
                source="llama_cpp_completion",
                capture_notes="prompt_contract_mode=messages",
            )
        )
        d = build_engine_input_trace(object())
        assert d is not None
        self.assertEqual(d["source"], "llama_cpp_completion")
        self.assertEqual(d["serialized_input"], "formatted")
        self.assertEqual(d["chat_format"], "chatml")
        self.assertEqual(d["input_mode"], "completion")

    def test_merge_engine_input_into_latest(self) -> None:
        buf = RoutingDebugBuffer()
        self.assertIsNone(buf.merge_engine_input_into_latest(None))
        rec0 = RoutingDebugRecord(
            timestamp=1.0,
            session_id="s",
            turn_id=3,
            query="q",
            route="chat",
            route_pre_policy="chat",
            strategy="adaptive_v4",
            trace_level="full",
            top_intent=None,
            top_score=None,
            summary="sum",
            trace={"winning_reason": "x"},
            decision={},
        )
        buf.append(rec0)
        patch_ei = {
            "trace_id": 1,
            "model_name": "m.gguf",
            "timestamp": 99.0,
            "input_mode": "completion",
            "messages": [],
            "prompt": "abc",
            "serialized_input": "abc",
            "chat_format": "chatml",
            "stop_tokens": [],
            "source": "llama_cpp_completion",
            "capture_notes": None,
        }
        out = buf.merge_engine_input_into_latest(patch_ei)
        assert out is not None
        snap = buf.snapshot()
        self.assertEqual(snap[-1].trace.get("engine_input_trace"), patch_ei)
        self.assertEqual(snap[-1].turn_id, 3)


if __name__ == "__main__":
    unittest.main()
