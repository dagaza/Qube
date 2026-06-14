"""Unit tests for Qwen3 sidecar inference helpers (no model load)."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from core.qwen3_sidecar_inference import (
    attach_chat_template_kwargs,
    _diagnostics_from_completion_output,
)
from core.title_think_trace import analyze_think_trace, split_think_and_answer

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


class TestCompletionDiagnostics(unittest.TestCase):
    def test_diagnostics_from_chat_output(self) -> None:
        output = {
            "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 3, "prompt_tokens": 10, "total_tokens": 13},
        }
        diag = _diagnostics_from_completion_output(
            output,
            path="chat",
            stop_sequences=[],
            chat_template_kwargs={"enable_thinking": False},
        )
        self.assertEqual(diag.raw_output, "Hello")
        self.assertEqual(diag.finish_reason, "stop")
        self.assertEqual(diag.completion_tokens, 3)
        self.assertEqual(diag.chat_template_kwargs, {"enable_thinking": False})


class TestAttachChatTemplateKwargs(unittest.TestCase):
    def test_handler_wrap_merges_kwargs(self) -> None:
        model = MagicMock()
        base = MagicMock(return_value="ok")
        model.chat_handler = base
        attach_chat_template_kwargs(model, {"enable_thinking": False})
        wrapped = model.chat_handler
        wrapped("a", b=1)
        base.assert_called_once_with("a", enable_thinking=False, b=1)


class TestThinkTrace(unittest.TestCase):
    def test_split_think_and_answer(self) -> None:
        raw = (
            f"{_THINK_OPEN}Need a title{_THINK_CLOSE}\n"
            "Nginx Reverse Proxy"
        )
        think, answer, had = split_think_and_answer(raw)
        self.assertTrue(had)
        self.assertIn("Need", think)
        self.assertIn("Nginx", answer)

    def test_analyze_finds_reasoning_candidate(self) -> None:
        raw = (
            f"{_THINK_OPEN}The best title is Nginx Reverse Proxy for this setup."
            f"{_THINK_CLOSE}\nStart By Installing"
        )
        analysis = analyze_think_trace(
            raw,
            user_prompt="setup nginx reverse proxy on ubuntu",
            assistant_reply="Start by installing nginx",
            final_title="Nginx Reverse Proxy",
        )
        self.assertTrue(analysis.candidate_in_reasoning)
        self.assertTrue(analysis.reasoning_has_best_title)
        self.assertFalse(analysis.answer_has_best_title)


if __name__ == "__main__":
    unittest.main()
