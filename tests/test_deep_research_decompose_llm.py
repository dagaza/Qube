"""Tests for Phase 5 slice 5 — LLM deep-research decomposition."""

from __future__ import annotations

import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.deep_research_decompose import decompose_query  # noqa: E402
from core.knowledge.deep_research_decompose_llm import (  # noqa: E402
    parse_llm_sub_queries,
    validate_llm_sub_queries,
)


class TestDeepResearchDecomposeLlm(unittest.TestCase):
    def test_parse_json_sub_queries(self) -> None:
        raw = (
            '{"sub_queries": ['
            '"ACE inhibitors heart failure mortality trial", '
            '"ACE inhibitors heart failure meta-analysis"]}'
        )
        parsed = parse_llm_sub_queries(raw)
        self.assertEqual(len(parsed), 2)

    def test_validate_inserts_base_query(self) -> None:
        validated = validate_llm_sub_queries(
            "ACE inhibitors heart failure evidence",
            ["ACE inhibitors heart failure meta-analysis"],
        )
        self.assertGreaterEqual(len(validated), 2)
        self.assertIn("ACE inhibitors heart failure evidence", validated)

    def test_decompose_query_falls_back_when_llm_invalid(self) -> None:
        def _bad(_system: str, _user: str) -> str:
            return "not json"

        parts = decompose_query(
            "ACE inhibitors heart failure evidence",
            generate_fn=_bad,
        )
        self.assertGreaterEqual(len(parts), 2)

    def test_decompose_query_uses_llm_when_valid(self) -> None:
        def _good(_system: str, _user: str) -> str:
            return (
                '{"sub_queries": ['
                '"ACE inhibitors heart failure mortality randomized trial", '
                '"ACE inhibitors heart failure systematic review meta-analysis"]}'
            )

        parts = decompose_query(
            "ACE inhibitors heart failure evidence",
            generate_fn=_good,
        )
        self.assertIn("ACE inhibitors heart failure evidence", parts)
        self.assertGreaterEqual(len(parts), 2)

    def test_decompose_accepts_bound_positional_callback(self) -> None:
        """DeepResearchWorker passes _decompose_generate as a positional callback."""

        class WorkerLike:
            def _decompose_generate(self, system: str, user: str) -> str:
                return (
                    '{"sub_queries": ['
                    '"ACE inhibitors heart failure mortality randomized trial", '
                    '"ACE inhibitors heart failure systematic review meta-analysis"]}'
                )

        parts = decompose_query(
            "ACE inhibitors heart failure evidence",
            generate_fn=WorkerLike()._decompose_generate,
        )
        self.assertGreaterEqual(len(parts), 2)


if __name__ == "__main__":
    unittest.main()
