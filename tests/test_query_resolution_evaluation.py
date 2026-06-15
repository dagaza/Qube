"""Tests for query resolution evaluation harness."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.query_resolution_evaluation import (
    DEFAULT_WEB_FIXTURES_DIR,
    build_discourse_resolution,
    build_query_resolution_summary,
    evaluate_query_resolution_case,
    load_query_resolution_corpus,
    run_web_fixture_retrieval,
)
from mcp.internet_tool import parse_ddg_html_results


class LoadQueryResolutionCorpusTests(unittest.TestCase):
    def test_load_v1_corpus(self) -> None:
        path = Path(ROOT) / "eval" / "router_corpus" / "query_resolution_v1.json"
        meta, cases = load_query_resolution_corpus(path)
        self.assertEqual(meta.get("schema"), "qube.query_resolution_corpus.v1")
        self.assertGreaterEqual(len(cases), 5)
        self.assertEqual(cases[0].id, "qr_fu_population")

    def test_reject_bad_schema(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "bad.json"
            bad.write_text(
                json.dumps({"schema": "other", "cases": [{"id": "x", "prompt": "hi", "category": "t", "expect": {}}]}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_query_resolution_corpus(bad)


class WebFixtureParserTests(unittest.TestCase):
    def test_parse_kathmandu_fixture(self) -> None:
        path = DEFAULT_WEB_FIXTURES_DIR / "kathmandu_population.html"
        rows = parse_ddg_html_results(path.read_text(encoding="utf-8"), max_results=10)
        self.assertGreaterEqual(len(rows), 3)
        joined = " ".join(r.get("snippet", "") for r in rows)
        self.assertIn("Kathmandu", joined)


class QueryResolutionEvalTests(unittest.TestCase):
    def test_kathmandu_population_case_passes_without_embedder(self) -> None:
        path = Path(ROOT) / "eval" / "router_corpus" / "query_resolution_v1.json"
        _meta, cases = load_query_resolution_corpus(path)
        case = next(c for c in cases if c.id == "qr_fu_population")
        result = evaluate_query_resolution_case(
            case,
            embed_fn=None,
            fixtures_dir=DEFAULT_WEB_FIXTURES_DIR,
        )
        self.assertTrue(result.resolution_pass, result.failed_checks)
        self.assertIn("Kathmandu", result.inference_text)
        self.assertIn("Kathmandu", result.web_text)
        self.assertGreaterEqual(result.web_fixture_hits, 1)

    def test_build_discourse_resolution_applies_inference_rewrite(self) -> None:
        history = (
            {"role": "user", "content": "What is the capital of Nepal?"},
            {"role": "assistant", "content": "Kathmandu is the capital of Nepal."},
            {"role": "user", "content": "And what is the size of its population?"},
        )
        _fu, _state, resolved = build_discourse_resolution(
            "And what is the size of its population?",
            history,
        )
        self.assertIn("Kathmandu", resolved.inference_text)
        self.assertIn("Kathmandu", resolved.web_text)
        self.assertNotIn(" its ", f" {resolved.web_text.lower()} ")

    def test_web_fixture_retrieval_lexical_gate(self) -> None:
        out = run_web_fixture_retrieval(
            "Kathmandu population",
            "kathmandu_population",
            fixtures_dir=DEFAULT_WEB_FIXTURES_DIR,
            embed_fn=None,
        )
        self.assertGreaterEqual(out["web_hits"], 1)
        self.assertGreaterEqual(out["web_raw_count"], 3)

    def test_summary_counts(self) -> None:
        path = Path(ROOT) / "eval" / "router_corpus" / "query_resolution_v1.json"
        _meta, cases = load_query_resolution_corpus(path)
        results = [
            evaluate_query_resolution_case(c, embed_fn=None, fixtures_dir=DEFAULT_WEB_FIXTURES_DIR)
            for c in cases
        ]
        summary = build_query_resolution_summary(results)
        self.assertEqual(summary.total, len(cases))
        self.assertGreater(summary.passed, 0)


if __name__ == "__main__":
    unittest.main()
