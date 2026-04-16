"""T3.2 — narrative / recap intent detector.

``detect_narrative_intent`` is the single source of truth for the
"what have we been working on?" style override used by ``LLMWorker`` to
force MEMORY + ``prefer_episode=True``. The detector must:

- match a handful of canonical recap phrasings
- reject ordinary questions that happen to contain the word "we"
- be case-insensitive
- return False for empty / whitespace input
"""
from __future__ import annotations

import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.memory_filters import (  # noqa: E402
    NARRATIVE_RECALL_SYSTEM_SUFFIX,
    detect_narrative_intent,
)


class TestDetectNarrativeIntent(unittest.TestCase):
    def test_recap_phrasings_match(self):
        positives = [
            "What have we been working on?",
            "what were we discussing?",
            "Can you recap what we did today?",
            "Catch me up on where we left off.",
            "where did we leave off?",
            "what's the status of this project?",
            "Summarize our conversation so far.",
            "Please summarize the chat.",
            "What have we decided so far?",
        ]
        for s in positives:
            self.assertTrue(
                detect_narrative_intent(s),
                f"expected narrative intent to fire for: {s!r}",
            )

    def test_non_recap_questions_do_not_match(self):
        negatives = [
            "",
            "   ",
            "What is the capital of France?",
            "Remember that my license expires in July.",
            "Can you read the PDF in my docs?",
            "Hello, how are you?",
            "Tell me about Dr. Evelyn.",
            "I prefer dark roast coffee.",
        ]
        for s in negatives:
            self.assertFalse(
                detect_narrative_intent(s),
                f"expected narrative intent to NOT fire for: {s!r}",
            )

    def test_suffix_constant_is_nonempty_and_mentions_episode(self):
        # The suffix is what steers the LLM to prefer EPISODE sources on
        # recap turns; if the word "EPISODE" ever disappears from it,
        # memory_tool's inline [EPISODE] label would no longer match the
        # instruction the LLM sees.
        self.assertIsInstance(NARRATIVE_RECALL_SYSTEM_SUFFIX, str)
        self.assertTrue(NARRATIVE_RECALL_SYSTEM_SUFFIX.strip())
        self.assertIn("EPISODE", NARRATIVE_RECALL_SYSTEM_SUFFIX)


if __name__ == "__main__":
    unittest.main()
