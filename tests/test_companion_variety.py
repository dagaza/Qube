"""Tests for variety store anti-repetition."""

from __future__ import annotations

import time
import unittest

from core.companion_cognition.variety import (
    VarietyStore,
    jaccard_similarity,
    normalize_line_fingerprint,
)


class TestCompanionVariety(unittest.TestCase):
    def test_jaccard_detects_similar_lines(self) -> None:
        a = normalize_line_fingerprint("Still here if you need me.")
        b = normalize_line_fingerprint("Still here if you need me today.")
        self.assertGreater(jaccard_similarity(a, b), 0.5)

    def test_semantic_duplicate_detection(self) -> None:
        store = VarietyStore()
        store.record_emission(
            message_id="m1",
            intent="wellbeing",
            mood="calm",
            line="Still here if you need me.",
            now=time.time(),
        )
        self.assertTrue(store.is_semantic_duplicate("Still here if you need me."))

    def test_intent_balance_penalty(self) -> None:
        store = VarietyStore()
        now = time.time()
        for _ in range(3):
            store.record_emission(
                message_id="m",
                intent="wellbeing",
                mood="calm",
                line="Line one.",
                now=now,
            )
        snap = store.snapshot(now=now)
        self.assertLess(store.intent_balance_penalty("wellbeing", window=3), 1.0)


if __name__ == "__main__":
    unittest.main()
