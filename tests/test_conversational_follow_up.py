"""Tests for conversational follow-up preservation after format retry."""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.conversational_follow_up import (
    extract_follow_up_candidate,
    is_lively_conversational_follow_up,
    is_safe_follow_up,
    preserve_streamed_follow_up,
)


class ConversationalFollowUpTests(unittest.TestCase):
    def test_lively_question_detected(self) -> None:
        self.assertTrue(is_lively_conversational_follow_up("Would you like another?"))

    def test_meta_tail_rejected(self) -> None:
        self.assertFalse(
            is_lively_conversational_follow_up("We need to produce final answer.")
        )

    def test_prefix_merge_keeps_tail(self) -> None:
        base = "Why did the scarecrow win an award? Because he was outstanding in his field."
        streamed = f"{base} Would you like another?"
        merged = preserve_streamed_follow_up(base, streamed)
        self.assertIn("Would you like another?", merged)

    def test_different_body_still_keeps_follow_up(self) -> None:
        streamed = (
            "Why did the chicken cross the road? To get to the other side! "
            "Would you like another?"
        )
        replacement = "Because he was outstanding in his field."
        merged = preserve_streamed_follow_up(replacement, streamed)
        self.assertIn("Would you like another?", merged)
        self.assertIn("outstanding in his field", merged)

    def test_paragraph_follow_up(self) -> None:
        streamed = (
            "Here is a joke for you.\n\n"
            "Want to hear another one?"
        )
        replacement = "Knock knock. Who's there? Lettuce."
        merged = preserve_streamed_follow_up(replacement, streamed)
        self.assertIn("Want to hear another one?", merged)

    def test_unsafe_planning_tail_not_preserved(self) -> None:
        streamed = (
            "The sky is blue. "
            "We need to answer why the sky is blue? Provide concise."
        )
        replacement = "Rayleigh scattering makes the sky look blue."
        merged = preserve_streamed_follow_up(replacement, streamed)
        self.assertEqual(merged, replacement)

    def test_duplicate_follow_up_not_appended(self) -> None:
        replacement = "Funny joke.\n\nWould you like another?"
        streamed = "Different joke.\n\nWould you like another?"
        merged = preserve_streamed_follow_up(replacement, streamed)
        self.assertEqual(merged.count("Would you like another?"), 1)

    def test_extract_follow_up_candidate(self) -> None:
        tail = extract_follow_up_candidate(
            "Joke body here.\n\nShall I tell you another?"
        )
        self.assertEqual(tail, "Shall I tell you another?")

    def test_safe_follow_up_passes_validation(self) -> None:
        self.assertTrue(is_safe_follow_up("Would you like another joke?"))


if __name__ == "__main__":
    unittest.main()
