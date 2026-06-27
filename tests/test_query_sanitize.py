"""Tests for API query sanitization."""

from __future__ import annotations

import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.adapters.query_sanitize import sanitize_api_query  # noqa: E402


class TestQuerySanitize(unittest.TestCase):
    def test_strips_trailing_question_mark(self) -> None:
        self.assertEqual(
            sanitize_api_query("semaglutide cardiovascular outcomes?"),
            "semaglutide cardiovascular outcomes",
        )

    def test_strips_multiple_trailing_punctuation(self) -> None:
        self.assertEqual(
            sanitize_api_query("ozempic side effects?!"),
            "ozempic side effects",
        )

    def test_collapses_whitespace(self) -> None:
        self.assertEqual(
            sanitize_api_query("  semaglutide   cardiovascular  "),
            "semaglutide cardiovascular",
        )

    def test_empty_and_whitespace(self) -> None:
        self.assertEqual(sanitize_api_query(""), "")
        self.assertEqual(sanitize_api_query("   "), "")


if __name__ == "__main__":
    unittest.main()
