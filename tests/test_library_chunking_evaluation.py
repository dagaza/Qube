"""Unit tests for Library chunking eval metrics (Phase 0 lite)."""

from __future__ import annotations

import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

if "lancedb" not in sys.modules:
    sys.modules["lancedb"] = types.ModuleType("lancedb")
if "pyarrow" not in sys.modules:
    pa = types.ModuleType("pyarrow")

    def _noop(*_args, **_kwargs):
        return None

    pa.schema = _noop
    pa.field = _noop
    pa.list_ = _noop
    pa.float32 = _noop
    pa.utf8 = _noop
    pa.int32 = _noop
    sys.modules["pyarrow"] = pa

from core.library_chunking_evaluation import (
    compare_runs,
    duplicate_pair_rate,
    evaluate_case,
    forbidden_source_hits,
    jaccard_similarity,
    LibraryEvalConfig,
    load_corpus,
    recall_at_k,
    reciprocal_rank,
    substring_check,
)


class MetricTests(unittest.TestCase):
    def test_recall_and_mrr(self) -> None:
        expected = {"eval_notes.md"}
        ranked = ["eval_other.md", "eval_notes.md", "eval_third.md"]
        self.assertEqual(recall_at_k(expected, ranked, 5), 1.0)
        self.assertAlmostEqual(reciprocal_rank(expected, ranked), 0.5)

    def test_jaccard_duplicate_rate(self) -> None:
        a = "API retry behavior with exponential backoff starting at 200 ms"
        b = "API retry behavior exponential backoff 200 ms maximum five attempts"
        self.assertGreater(jaccard_similarity(a, b), 0.3)
        self.assertEqual(duplicate_pair_rate([a, a]), 1.0)
        self.assertEqual(duplicate_pair_rate(["unique alpha", "unique beta"]), 0.0)

    def test_substring_check(self) -> None:
        ok, missing = substring_check("Revenue reached $4.2 million", ["4.2 million"])
        self.assertTrue(ok)
        self.assertEqual(missing, [])

    def test_forbidden_source_prefix(self) -> None:
        hits = forbidden_source_hits(
            [
                "eval_project_notes.md",
                "user_manual.pdf",
                "eval_api_integration_guide.md",
            ],
            forbidden_sources_prefix="eval_",
        )
        self.assertEqual(
            hits,
            ["eval_project_notes.md", "eval_api_integration_guide.md"],
        )
        self.assertEqual(
            forbidden_source_hits(["user_manual.pdf"], forbidden_sources_prefix="eval_"),
            [],
        )

    def test_forbidden_source_top_n_scope(self) -> None:
        hits = forbidden_source_hits(
            ["decoy_world_facts.md", "eval_project_notes.md"],
            forbidden_sources_prefix="eval_",
            top_n=1,
        )
        self.assertEqual(hits, [])

    def test_compare_runs_detects_regression(self) -> None:
        baseline = {"success_rate": 0.9, "recall_at_k_mean": 0.85, "duplicate_pair_rate_mean": 0.1}
        current = {"success_rate": 0.8, "recall_at_k_mean": 0.85, "duplicate_pair_rate_mean": 0.1}
        regressions = compare_runs(current, baseline)
        self.assertTrue(any("success_rate" in r for r in regressions))


class EvaluateCaseTests(unittest.TestCase):
    def test_evaluate_case_with_mock_store(self) -> None:
        class _FakeStore:
            pass

        def _embed(_text: str) -> list[float]:
            return [0.1, 0.2, 0.3, 0.4]

        case = {
            "id": "lib_test",
            "query": "API retry behavior",
            "expected_sources": ["eval_api_integration_guide.md"],
            "expect_contains": ["exponential backoff"],
            "category": "spec",
        }

        def _fake_rag_search(_query, _vector, _store, top_k=5, **_kwargs):
            return {
                "llm_context": "stub",
                "sources": [
                    {
                        "id": 1,
                        "filename": "eval_api_integration_guide.md",
                        "content": "clients must use exponential backoff starting at 200 ms",
                        "type": "rag",
                        "chunk_id": "eval_api_integration_guide.md::0",
                    }
                ],
            }

        with patch("mcp.rag_tool.rag_search", _fake_rag_search):
            result = evaluate_case(
                case,
                embed_fn=_embed,
                store=_FakeStore(),
                config=LibraryEvalConfig(top_k=5),
            )

        self.assertTrue(result.success)
        self.assertEqual(result.recall_at_k, 1.0)
        self.assertEqual(result.reciprocal_rank, 1.0)


class CorpusTests(unittest.TestCase):
    def test_load_baseline_corpus(self) -> None:
        path = Path(ROOT) / "eval" / "library_corpus" / "v1_baseline.json"
        data = load_corpus(path)
        self.assertGreaterEqual(len(data["cases"]), 16)
        negative = [c for c in data["cases"] if c["id"] == "lib_neg_001"]
        self.assertEqual(len(negative), 1)
        self.assertEqual(negative[0].get("forbidden_sources_prefix"), "eval_")


if __name__ == "__main__":
    unittest.main()
