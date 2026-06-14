"""Tests for offline router evaluation harness."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.router_evaluation import (
    RouterEvalCase,
    RouterEvalConfig,
    RouterEvalResult,
    build_summary,
    classify_failure_reason,
    compare_runs,
    corpus_fingerprint,
    evaluate_case,
    family_match,
    is_over_retrieval,
    is_under_retrieval,
    load_corpus,
    normalize_route,
    route_family,
    simulate_execution_route,
    write_run_json,
)
from mcp.cognitive_router import CognitiveRouterV4


class NormalizeRouteTests(unittest.TestCase):
    def test_aliases(self) -> None:
        self.assertEqual(normalize_route("CHAT"), "none")
        self.assertEqual(normalize_route("internet"), "web")

    def test_route_families(self) -> None:
        self.assertEqual(route_family("none"), "CHAT")
        self.assertEqual(route_family("memory"), "RETRIEVAL")
        self.assertEqual(route_family("hybrid"), "RETRIEVAL")
        self.assertTrue(family_match("memory", "hybrid"))
        self.assertTrue(family_match("rag", "memory"))
        self.assertFalse(family_match("web", "none"))


class LoadCorpusTests(unittest.TestCase):
    def test_load_baseline(self) -> None:
        path = Path(ROOT) / "eval" / "router_corpus" / "v1_baseline.json"
        self.assertTrue(path.is_file())
        meta, cases = load_corpus(path)
        self.assertEqual(meta.get("schema"), "qube.router_corpus.v1")
        self.assertEqual(len(cases), 125)
        self.assertEqual(cases[0].id, "gk_001")

    def test_reject_invalid_route(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "bad.json"
            bad.write_text(
                json.dumps(
                    {
                        "schema": "qube.router_corpus.v1",
                        "cases": [
                            {
                                "id": "x",
                                "prompt": "hi",
                                "expected_route": "invalid",
                                "category": "t",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_corpus(bad)


class SimulateExecutionRouteTests(unittest.TestCase):
    def test_explicit_remember_short_circuit(self) -> None:
        decision = {"route": "hybrid"}
        route, reason = simulate_execution_route(
            prompt="Please remember that my cat is Luna.",
            decision=decision,
            config=RouterEvalConfig(),
        )
        self.assertEqual(route, "NONE")
        self.assertEqual(reason, "explicit_remember")

    def test_recall_fusion(self) -> None:
        decision = {"route": "none"}
        route, reason = simulate_execution_route(
            prompt="Tell me about Dr. Evelyn.",
            decision=decision,
            config=RouterEvalConfig(),
        )
        self.assertEqual(route, "HYBRID")
        self.assertEqual(reason, "recall_fusion")

    def test_web_veto_when_disabled(self) -> None:
        # Router-picked WEB without explicit live-web triggers (weather/today
        # count as manual_web and bypass the veto, matching LLMWorker).
        decision = {"route": "web", "internet_enabled": True}
        route, reason = simulate_execution_route(
            prompt="Population of Estonia",
            decision=decision,
            config=RouterEvalConfig(internet_enabled=False),
        )
        self.assertEqual(route, "NONE")
        self.assertEqual(reason, "web_veto_tool_disabled")

    def test_rag_veto_when_master_disabled(self) -> None:
        decision = {
            "route": "rag",
            "top_intent": "rag",
            "rag_score_final": 0.4,
            "rag_score_source": "substring",
        }
        route, reason = simulate_execution_route(
            prompt="What does the policy say about refunds?",
            decision=decision,
            config=RouterEvalConfig(mcp_rag_enabled=False),
        )
        self.assertEqual(route, "NONE")
        self.assertEqual(reason, "rag_veto_tool_disabled")
        self.assertTrue(decision.get("rag_vetoed_tool_disabled"))

    def test_hybrid_downgrades_to_memory_when_master_disabled(self) -> None:
        decision = {"route": "hybrid", "recall_fusion": True}
        route, reason = simulate_execution_route(
            prompt="Tell me about Dr. Evelyn.",
            decision=decision,
            config=RouterEvalConfig(mcp_rag_enabled=False),
        )
        self.assertEqual(route, "MEMORY")
        self.assertTrue(decision.get("rag_library_leg_skipped"))
        self.assertIn(reason, ("recall_fusion", "rag_veto_tool_disabled"))

    def test_file_search_bypasses_rag_veto(self) -> None:
        decision = {"route": "rag"}
        route, reason = simulate_execution_route(
            prompt="Search my documents for API retry behavior.",
            decision=decision,
            config=RouterEvalConfig(mcp_rag_enabled=False),
        )
        self.assertEqual(route, "RAG")
        self.assertEqual(reason, "explicit_file_search")


class EvaluateCaseTests(unittest.TestCase):
    def test_general_knowledge_no_embeddings(self) -> None:
        case = RouterEvalCase(
            id="t1",
            prompt="Why is the sky blue?",
            expected_route="none",
            category="general_knowledge",
        )
        router = CognitiveRouterV4()
        result = evaluate_case(
            case,
            router=router,
            embed_fn=None,
            config=RouterEvalConfig(install_centroids=False),
        )
        self.assertEqual(result.case_id, "t1")
        self.assertFalse(result.error)
        self.assertIn(result.router_route, ("none", "rag", "memory", "hybrid", "web"))

    def test_evaluate_with_synthetic_embedding(self) -> None:
        case = RouterEvalCase(
            id="t2",
            prompt="the quick brown fox",
            expected_route="none",
            category="general_knowledge",
        )
        router = CognitiveRouterV4()
        router.set_chat_centroid(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        router.set_recall_centroid(np.array([1.0, 0.0, 0.0], dtype=np.float32))

        def _embed(_text: str) -> np.ndarray:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)

        result = evaluate_case(
            case,
            router=router,
            embed_fn=_embed,
            config=RouterEvalConfig(install_centroids=False),
        )
        self.assertGreaterEqual(result.chat_score, 0.0)


class RetrievalCalibrationTests(unittest.TestCase):
    def test_over_retrieval_chat_with_hits(self) -> None:
        r = RouterEvalResult(
            case_id="x",
            prompt="p",
            expected_route="none",
            category="general_knowledge_retrieval_tempting",
            notes="",
            router_route="hybrid",
            execution_route_pre_retrieval="hybrid",
            execution_route_final="hybrid",
            top_intent="recall",
            top_score=0.8,
            chat_score=0.7,
            confidence_margin=0.1,
            memory_hits=1,
            rag_hits=2,
            web_hits=0,
            downgrade_fired=False,
            rewrite_applied=False,
            router_match=False,
            execution_pre_match=False,
            execution_final_match=False,
            recall_fusion_triggered=True,
            over_retrieval=True,
        )
        self.assertTrue(is_over_retrieval(r))

    def test_not_over_retrieval_when_downgraded_to_chat(self) -> None:
        r = RouterEvalResult(
            case_id="x",
            prompt="p",
            expected_route="none",
            category="general_knowledge",
            notes="",
            router_route="hybrid",
            execution_route_pre_retrieval="hybrid",
            execution_route_final="none",
            top_intent="recall",
            top_score=0.8,
            chat_score=0.7,
            confidence_margin=0.1,
            memory_hits=2,
            rag_hits=0,
            web_hits=0,
            downgrade_fired=True,
            rewrite_applied=False,
            router_match=True,
            execution_pre_match=False,
            execution_final_match=True,
        )
        self.assertFalse(is_over_retrieval(r))

    def test_under_retrieval_missed_opportunity(self) -> None:
        r = RouterEvalResult(
            case_id="y",
            prompt="q",
            expected_route="rag",
            category="rag_retrieval",
            notes="",
            router_route="none",
            execution_route_pre_retrieval="none",
            execution_route_final="none",
            top_intent="memory",
            top_score=0.1,
            chat_score=0.5,
            confidence_margin=0.0,
            memory_hits=0,
            rag_hits=0,
            web_hits=0,
            downgrade_fired=False,
            rewrite_applied=False,
            router_match=False,
            execution_pre_match=False,
            execution_final_match=False,
        )
        self.assertTrue(is_under_retrieval(r))

    def test_calibration_summary_in_build_summary(self) -> None:
        results = [
            RouterEvalResult(
                case_id="chat_ok",
                prompt="a",
                expected_route="none",
                category="general_knowledge_retrieval_tempting",
                notes="",
                router_route="none",
                execution_route_pre_retrieval="none",
                execution_route_final="none",
                top_intent="memory",
                top_score=0.1,
                chat_score=0.8,
                confidence_margin=0.2,
                memory_hits=0,
                rag_hits=0,
                web_hits=0,
                downgrade_fired=False,
                rewrite_applied=False,
                router_match=True,
                execution_pre_match=True,
                execution_final_match=True,
                strict_success=True,
                family_success=True,
            ),
            RouterEvalResult(
                case_id="chat_over",
                prompt="b",
                expected_route="none",
                category="general_knowledge_retrieval_tempting",
                notes="",
                router_route="hybrid",
                execution_route_pre_retrieval="hybrid",
                execution_route_final="hybrid",
                top_intent="recall",
                top_score=0.9,
                chat_score=0.75,
                confidence_margin=0.05,
                memory_hits=0,
                rag_hits=3,
                web_hits=0,
                downgrade_fired=False,
                rewrite_applied=False,
                router_match=False,
                execution_pre_match=False,
                execution_final_match=False,
                strict_success=False,
                family_success=False,
                recall_fusion_triggered=True,
                over_retrieval=True,
                retrieval_type="rag",
            ),
        ]
        summary = build_summary(results)
        rc = summary.retrieval_calibration
        self.assertEqual(rc["chat_labeled_total"], 2)
        self.assertEqual(rc["over_retrieval_count"], 1)
        self.assertAlmostEqual(rc["over_retrieval_rate"], 0.5)
        self.assertEqual(rc["recall_fusion_over_retrieval_count"], 1)
        self.assertEqual(len(rc["retrieval_suppression_candidates"]), 1)


class FailureClassificationTests(unittest.TestCase):
    def test_recall_fusion_family_success(self) -> None:
        reason = classify_failure_reason(
            strict_success=False,
            family_success=True,
            expected_route="memory",
            router_route="none",
            execution_pre="hybrid",
            execution_final="hybrid",
            override_reason="recall_fusion",
            downgrade_fired=False,
            memory_hits=1,
            rag_hits=0,
            web_hits=0,
            memory_candidates=1,
            rag_candidates=0,
            relevance_gate_dropped=False,
            rewrite_attempted=False,
            rewrite_applied=False,
            error="",
        )
        self.assertEqual(reason, "recall_fusion_upgrade")

    def test_downgrade_empty_retrieval(self) -> None:
        reason = classify_failure_reason(
            strict_success=False,
            family_success=False,
            expected_route="rag",
            router_route="rag",
            execution_pre="rag",
            execution_final="none",
            override_reason="",
            downgrade_fired=True,
            memory_hits=0,
            rag_hits=0,
            web_hits=0,
            memory_candidates=0,
            rag_candidates=0,
            relevance_gate_dropped=False,
            rewrite_attempted=False,
            rewrite_applied=False,
            error="",
        )
        self.assertEqual(reason, "downgrade_to_none")


class SummaryAndRegressionTests(unittest.TestCase):
    def test_build_summary_and_compare(self) -> None:
        from core.router_evaluation import RouterEvalResult

        results = [
            RouterEvalResult(
                case_id="a",
                prompt="p",
                expected_route="none",
                category="general_knowledge",
                notes="",
                router_route="none",
                execution_route_pre_retrieval="none",
                execution_route_final="none",
                top_intent="memory",
                top_score=0.1,
                chat_score=0.5,
                confidence_margin=0.2,
                memory_hits=0,
                rag_hits=0,
                web_hits=0,
                downgrade_fired=False,
                rewrite_applied=False,
                router_match=True,
                execution_pre_match=True,
                execution_final_match=True,
                strict_success=True,
                family_success=True,
                failure_reason="no_failure",
            ),
            RouterEvalResult(
                case_id="b",
                prompt="q",
                expected_route="rag",
                category="rag_retrieval",
                notes="",
                router_route="none",
                execution_route_pre_retrieval="hybrid",
                execution_route_final="hybrid",
                top_intent="rag",
                top_score=0.2,
                chat_score=0.4,
                confidence_margin=0.1,
                memory_hits=1,
                rag_hits=1,
                web_hits=0,
                downgrade_fired=False,
                rewrite_applied=False,
                router_match=False,
                execution_pre_match=False,
                execution_final_match=False,
                strict_success=False,
                family_success=True,
                failure_reason="recall_fusion_upgrade",
            ),
        ]
        summary = build_summary(results)
        self.assertEqual(summary.total, 2)
        self.assertAlmostEqual(summary.strict_accuracy, 0.5)
        self.assertAlmostEqual(summary.family_accuracy, 1.0)
        self.assertIn("recall_fusion_upgrade", summary.failure_causes)

        baseline = {
            "summary": {"execution_final_accuracy": 1.0},
            "results": [
                {"case_id": "a", "execution_final_match": True, "execution_route_final": "none", "expected_route": "none"},
                {"case_id": "b", "execution_final_match": True, "execution_route_final": "rag", "expected_route": "rag"},
            ],
        }
        current = {
            "summary": {"execution_final_accuracy": 0.5},
            "results": [
                {"case_id": "a", "execution_final_match": True, "execution_route_final": "none", "expected_route": "none"},
                {"case_id": "b", "execution_final_match": False, "execution_route_final": "none", "expected_route": "rag"},
            ],
        }
        cmp = compare_runs(baseline, current)
        self.assertTrue(cmp["regressed"])
        self.assertEqual(len(cmp["new_failures"]), 1)

    def test_write_run_roundtrip(self) -> None:
        from core.router_evaluation import RouterEvalResult

        with tempfile.TemporaryDirectory() as td:
            corpus = Path(td) / "c.json"
            corpus.write_text(
                json.dumps(
                    {
                        "schema": "qube.router_corpus.v1",
                        "version": 1,
                        "description": "t",
                        "cases": [
                            {
                                "id": "x",
                                "prompt": "hi",
                                "expected_route": "none",
                                "category": "t",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            meta, _ = load_corpus(corpus)
            results = [
                RouterEvalResult(
                    case_id="x",
                    prompt="hi",
                    expected_route="none",
                    category="t",
                    notes="",
                    router_route="none",
                    execution_route_pre_retrieval="none",
                    execution_route_final="none",
                    top_intent="",
                    top_score=0.0,
                    chat_score=0.0,
                    confidence_margin=0.0,
                    memory_hits=0,
                    rag_hits=0,
                    web_hits=0,
                    downgrade_fired=False,
                    rewrite_applied=False,
                    router_match=True,
                    execution_pre_match=True,
                    execution_final_match=True,
                    strict_success=True,
                    family_success=True,
                    failure_reason="no_failure",
                )
            ]
            summary = build_summary(results)
            out = Path(td) / "run.json"
            write_run_json(
                out,
                corpus_path=corpus,
                corpus_meta=meta,
                config=RouterEvalConfig(),
                results=results,
                summary=summary,
                run_id="test",
            )
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(data["schema"], "qube.router_eval_run.v1")
            self.assertEqual(corpus_fingerprint(corpus), data["corpus_fingerprint"])


if __name__ == "__main__":
    unittest.main()
