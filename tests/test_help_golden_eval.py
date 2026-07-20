"""Tests for help golden retrieval eval (Phase 6 §17)."""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.help_golden_eval import (  # noqa: E402
    assert_v1_targets,
    evaluate_golden_questions,
    load_golden_questions,
)


class HelpGoldenEvalTests(unittest.TestCase):
    def test_fixture_has_v1_minimum(self) -> None:
        rows = load_golden_questions()
        self.assertGreaterEqual(len(rows), 55)
        negatives = sum(1 for row in rows if row.get("negative"))
        self.assertGreaterEqual(negatives, 5)

    def test_v1_retrieval_targets(self) -> None:
        summary = evaluate_golden_questions()
        assert_v1_targets(summary)


if __name__ == "__main__":
    unittest.main()
