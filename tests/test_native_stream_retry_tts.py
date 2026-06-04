"""Native streaming retry must not speak internal format-fallback markers."""

from __future__ import annotations

import re
import unittest


class TestNativeStreamRetryTts(unittest.TestCase):
    def test_tts_strip_removes_format_fallback_marker(self) -> None:
        raw = "[format fallback applied]\n\nHere's a joke."
        text = re.sub(r'[*_]{1,3}', '', raw)
        text = re.sub(r'\[(\d+|W)\]', '', text)
        text = re.sub(
            r"\[\s*format\s+fallback\s+applied\s*\]",
            "",
            text,
            flags=re.IGNORECASE,
        )
        cleaned = text.strip()
        self.assertNotIn("format fallback", cleaned.lower())
        self.assertIn("Here's a joke", cleaned)

    def test_tts_dedupe_key_normalizes_citations(self) -> None:
        raw_a = "Here's a joke from the list in [2]:"
        raw_b = "Here's a joke from the list in :"
        normalize = lambda s: re.sub(r"\s+", " ", re.sub(r"\[(\d+|W)\]", "", s)).strip().lower()
        self.assertEqual(normalize(raw_a), normalize(raw_b))

    def test_merge_user_visible_tail_keeps_short_follow_up(self) -> None:
        from core.output_artifact_strip import merge_user_visible_stream_tail

        replacement = "Why did the scarecrow win an award? Because he was outstanding in his field."
        streamed = (
            f"{replacement} Would you like another?"
        )
        merged = merge_user_visible_stream_tail(replacement, streamed)
        self.assertIn("Would you like another?", merged)

    def test_merge_user_visible_tail_rejects_long_tail(self) -> None:
        from core.output_artifact_strip import merge_user_visible_stream_tail

        replacement = "Short answer."
        streamed = replacement + " " + ("x" * 200)
        merged = merge_user_visible_stream_tail(replacement, streamed)
        self.assertEqual(merged, replacement)


if __name__ == "__main__":
    unittest.main()
