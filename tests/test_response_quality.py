from __future__ import annotations

import unittest

from core.response_quality import evaluate_response_quality


class TestResponseQuality(unittest.TestCase):
    def test_good_answer_scores_high(self) -> None:
        q = "Explain briefly how gravity works."
        o = "Gravity works by pulling masses toward each other. Bigger mass means stronger gravity."
        res = evaluate_response_quality(q, o)
        self.assertGreaterEqual(res.score, 0.75)
        self.assertIn(res.confidence, ("high", "medium"))

    def test_off_topic_answer_scores_low(self) -> None:
        q = "How do I reverse a Python list?"
        o = "The Roman Empire expanded across Europe and influenced law and architecture."
        res = evaluate_response_quality(q, o)
        self.assertLess(res.score, 0.45)
        self.assertIn("low_relevance", res.issues)

    def test_formally_correct_but_useless_scores_low(self) -> None:
        q = "How can I fix a failing unit test?"
        o = "It depends."
        res = evaluate_response_quality(q, o)
        self.assertLess(res.score, 0.45)
        self.assertIn("low_utility", res.issues)
