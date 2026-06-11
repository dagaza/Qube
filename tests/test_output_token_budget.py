"""Tests for output token budget and truncation notice heuristics."""
from __future__ import annotations

import unittest

from core.output_token_budget import (
    clamp_max_tokens_to_context,
    describe_output_token_budget,
    probable_max_tokens_truncation,
    resolve_output_token_budget,
)


class TestOutputTokenBudget(unittest.TestCase):
    def test_resolve_limit_enabled_caps_at_user_limit(self) -> None:
        self.assertEqual(
            resolve_output_token_budget(
                context_window=15000,
                limit_enabled=True,
                user_limit=4096,
            ),
            4096,
        )

    def test_resolve_limit_disabled_uses_remaining_context(self) -> None:
        self.assertEqual(
            resolve_output_token_budget(
                context_window=15000,
                limit_enabled=False,
                user_limit=4096,
            ),
            14488,
        )

    def test_clamp_after_prompt_known(self) -> None:
        self.assertEqual(
            clamp_max_tokens_to_context(
                n_ctx=8000,
                prompt_token_count=6000,
                requested_max_tokens=4096,
                limit_enabled=True,
            ),
            1936,
        )

    def test_clamp_unlimited_uses_all_remaining(self) -> None:
        self.assertEqual(
            clamp_max_tokens_to_context(
                n_ctx=8000,
                prompt_token_count=6000,
                requested_max_tokens=4096,
                limit_enabled=False,
            ),
            1936,
        )

    def test_describe_limit_enabled(self) -> None:
        text = describe_output_token_budget(
            context_window=15000,
            limit_enabled=True,
            user_limit=4096,
            chat_history_messages=10,
        )
        self.assertIn("4,096", text)
        self.assertIn("15,000", text)
        self.assertIn("10 messages", text)

    def test_describe_limit_disabled_mentions_shared_window(self) -> None:
        text = describe_output_token_budget(
            context_window=15000,
            limit_enabled=False,
            user_limit=4096,
            chat_history_messages=20,
        )
        self.assertIn("not capped separately", text)
        self.assertIn("20 messages", text)

    def test_probable_truncation_finish_reason_length(self) -> None:
        self.assertEqual(
            probable_max_tokens_truncation(
                "Hello world.",
                stream_finish_reason="length",
                max_tokens=512,
                limit_enabled=True,
            ),
            "finish_reason_length",
        )

    def test_probable_truncation_heuristic_unfinished_list(self) -> None:
        body = (
            "* **Item 1** — First.\n"
            "* **Item 2** — Second.\n"
            "* **Post-Impressionism** — A reaction to Impressionism, these"
        )
        reason = probable_max_tokens_truncation(
            body,
            stream_finish_reason="",
            max_tokens=4096,
            limit_enabled=True,
            completion_token_count=4000,
        )
        self.assertEqual(reason, "heuristic_truncated_output")

    def test_probable_truncation_skips_when_not_near_cap(self) -> None:
        body = "* **Post-Impressionism** — A reaction to Impressionism, these"
        self.assertIsNone(
            probable_max_tokens_truncation(
                body,
                stream_finish_reason="",
                max_tokens=4096,
                limit_enabled=True,
                completion_token_count=500,
            )
        )


if __name__ == "__main__":
    unittest.main()
