"""Assistant answer pattern extraction for discourse referent promotion."""
from __future__ import annotations

import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.discourse_answer_patterns import (  # noqa: E402
    extract_referent_from_assistant_answer,
)


class TestDiscourseAnswerPatterns(unittest.TestCase):
    def test_capital_of_city_first(self) -> None:
        match = extract_referent_from_assistant_answer(
            "Kathmandu is the capital of Nepal."
        )
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.referent, "Kathmandu")
        self.assertEqual(match.referent_type, "city")
        self.assertEqual(match.pattern_id, "capital_of")

    def test_capital_of_country_first(self) -> None:
        match = extract_referent_from_assistant_answer(
            "The capital of Nepal is Kathmandu."
        )
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.referent, "Kathmandu")
        self.assertEqual(match.pattern_id, "capital_of_alt")

    def test_ceo_pattern(self) -> None:
        match = extract_referent_from_assistant_answer(
            "Tim Cook is the CEO of Apple."
        )
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.referent, "Tim Cook")
        self.assertEqual(match.referent_type, "person")
        self.assertEqual(match.pattern_id, "ceo_of")

    def test_founded_by_pattern(self) -> None:
        match = extract_referent_from_assistant_answer(
            "Steve Jobs founded Apple."
        )
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.referent, "Steve Jobs")
        self.assertEqual(match.pattern_id, "founded_by")

    def test_thin_answer_no_match(self) -> None:
        self.assertIsNone(
            extract_referent_from_assistant_answer("I don't have that information.")
        )


if __name__ == "__main__":
    unittest.main()
