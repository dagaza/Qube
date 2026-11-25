"""Tests for output token budget and truncation notice heuristics."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from core.output_token_budget import (
    clamp_max_tokens_to_context,
    count_prompt_tokens_for_budget,
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

    def test_count_prompt_tokens_for_budget_empty(self) -> None:
        self.assertEqual(count_prompt_tokens_for_budget(None, "hello"), 0)
        self.assertEqual(count_prompt_tokens_for_budget(object(), ""), 0)


def test_is_native_empty_visible_output_notice() -> None:
    from workers.llm_worker import (
        NATIVE_EMPTY_VISIBLE_OUTPUT_MSG,
        is_native_empty_visible_output_notice,
    )

    assert is_native_empty_visible_output_notice(NATIVE_EMPTY_VISIBLE_OUTPUT_MSG)
    assert is_native_empty_visible_output_notice(
        NATIVE_EMPTY_VISIBLE_OUTPUT_MSG + NATIVE_EMPTY_VISIBLE_OUTPUT_MSG
    )
    assert not is_native_empty_visible_output_notice("Kathmandu is the capital.")


class _FakeLlama:
    """llama-cpp-python 0.3.x shape: ``n_ctx`` is a method, ``tokenize`` returns ids."""

    def __init__(self, n_ctx: int, prompt_tokens: int) -> None:
        self._n = n_ctx
        self._prompt_tokens = prompt_tokens

    def n_ctx(self) -> int:
        return self._n

    def tokenize(self, text: bytes, add_bos: bool = False, special: bool = True) -> list[int]:
        return [0] * self._prompt_tokens


def _engine(llama: object | None, *, loaded_n_ctx: int = 4096, prompt: str = "<prompt>"):
    from workers.native_llama_engine import NativeLlamaEngine

    engine = NativeLlamaEngine.__new__(NativeLlamaEngine)
    engine._llama = llama
    engine._n_ctx = loaded_n_ctx
    engine._last_formatted_prompt = prompt
    return engine


class EngineMaxTokensClampTests(unittest.TestCase):
    """Plan S16 (V21): the native engine's clamp runs with a callable ``n_ctx``."""

    def test_old_expression_raised_on_callable_n_ctx(self) -> None:
        # The pre-S16 expression, kept here as the reason for the fix.
        llama = _FakeLlama(4096, 3000)
        with self.assertRaises(TypeError):
            int(getattr(llama, "n_ctx", 0) or 4096)

    def test_clamp_fits_reply_into_remaining_window(self) -> None:
        engine = _engine(_FakeLlama(4096, 3000))
        with self.assertNoLogs("Qube.NativeLLM.Debug", level="WARNING"):
            out = engine._clamp_max_tokens_for_prompt(4096, context_window=8192, limit_enabled=True)
        # 4096 loaded (not the 8192 requested) - 3000 prompt - 64 margin.
        self.assertEqual(out, 1032)

    def test_clamp_keeps_user_limit_when_it_fits(self) -> None:
        engine = _engine(_FakeLlama(32768, 500))
        self.assertEqual(engine._clamp_max_tokens_for_prompt(2048, context_window=32768, limit_enabled=True), 2048)

    def test_unlimited_uses_remaining_context(self) -> None:
        engine = _engine(_FakeLlama(8192, 1000))
        self.assertEqual(
            engine._clamp_max_tokens_for_prompt(7680, context_window=8192, limit_enabled=False), 8192 - 1000 - 64
        )

    def test_no_formatted_prompt_leaves_max_tokens(self) -> None:
        engine = _engine(_FakeLlama(4096, 3000), prompt="")
        self.assertEqual(engine._clamp_max_tokens_for_prompt(4096, context_window=4096, limit_enabled=True), 4096)

    def test_tokenize_failure_still_clamps_with_zero_prompt(self) -> None:
        class _NoTokenize(_FakeLlama):
            def tokenize(self, *a, **k):  # count_prompt_tokens_for_budget swallows this
                raise RuntimeError("tokenizer broke")

        engine = _engine(_NoTokenize(4096, 0))
        self.assertEqual(engine._clamp_max_tokens_for_prompt(8000, context_window=4096, limit_enabled=True), 3840)

    def test_unexpected_error_skips_clamp(self) -> None:
        engine = _engine(_FakeLlama(4096, 3000))
        with (
            patch("core.output_token_budget.clamp_max_tokens_to_context", side_effect=ValueError("boom")),
            self.assertLogs("Qube.NativeLLM.Debug", level="WARNING") as logs,
        ):
            self.assertEqual(
                engine._clamp_max_tokens_for_prompt(4096, context_window=4096, limit_enabled=True), 4096
            )
        self.assertIn("clamp skipped", logs.output[0])


if __name__ == "__main__":
    unittest.main()
