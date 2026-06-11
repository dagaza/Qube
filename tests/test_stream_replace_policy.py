from __future__ import annotations

import unittest

from core.stream_replace_policy import resolve_stream_replacement


class TestStreamReplacePolicy(unittest.TestCase):
    def test_keeps_stream_when_retry_is_much_shorter(self) -> None:
        streamed = "A" * 1000
        retry = "B" * 100
        resolved, reason = resolve_stream_replacement(retry, streamed)
        self.assertEqual(resolved, streamed)
        self.assertEqual(reason, "retry_shorter_than_streamed")

    def test_allows_retry_when_longer(self) -> None:
        streamed = "Short streamed body."
        retry = streamed + " Extra section with more detail."
        resolved, reason = resolve_stream_replacement(retry, streamed)
        self.assertEqual(reason, None)
        self.assertEqual(resolved, retry)

    def test_keeps_stream_when_retry_is_prefix(self) -> None:
        streamed = "Intro section.\n\n# Music\n\n* **Item** — detail."
        retry = "Intro section."
        resolved, reason = resolve_stream_replacement(retry, streamed)
        self.assertEqual(resolved, streamed)
        self.assertEqual(reason, "retry_prefix_of_streamed")

    def test_preserves_follow_up_when_retry_slightly_shorter(self) -> None:
        streamed = (
            "Why did the scarecrow win an award? "
            "Because he was outstanding in his field. "
            "Would you like another?"
        )
        retry = "Because he was outstanding in his field."
        resolved, reason = resolve_stream_replacement(retry, streamed, min_ratio=0.5)
        self.assertIsNone(reason)
        self.assertIn("Would you like another?", resolved)


if __name__ == "__main__":
    unittest.main()
