"""Tests for Qwen3 thinking policy helpers (no llama_cpp)."""
from __future__ import annotations

import unittest

from core.execution_policy import resolve_execution_policy
from core.model_reasoning_profile import ModelReasoningProfile
from core.qwen3_thinking_policy import is_qwen3_model, template_kwargs_for_thinking_policy


def _qwen_profile() -> ModelReasoningProfile:
    return ModelReasoningProfile(
        model_name="Qwen3.5-9B",
        supports_thinking_tokens=True,
        thinking_token_patterns=["</think>"],
        default_mode="thinking",
        reasoning_confidence=1.0,
        detection_method="tokenizer_scan",
    )


class TestQwen3ThinkingPolicy(unittest.TestCase):
    def test_is_qwen3_model_detects_path(self) -> None:
        self.assertTrue(
            is_qwen3_model(model_path="/models/Qwen3-1.7B-Q6_K.gguf", model_name="")
        )
        self.assertFalse(is_qwen3_model(model_path="/models/llama.gguf", model_name=""))

    def test_template_kwargs_when_think_off(self) -> None:
        pol = resolve_execution_policy(_qwen_profile(), False, "internal")
        kw = template_kwargs_for_thinking_policy(
            pol,
            model_path="/models/Qwen3.5-9B-Q6_K.gguf",
            model_name="Qwen3.5-9B",
        )
        self.assertEqual(kw, {"enable_thinking": False})

    def test_template_kwargs_when_think_on(self) -> None:
        pol = resolve_execution_policy(_qwen_profile(), True, "internal")
        kw = template_kwargs_for_thinking_policy(
            pol,
            model_path="/models/Qwen3.5-9B-Q6_K.gguf",
            model_name="Qwen3.5-9B",
        )
        self.assertEqual(kw, {"enable_thinking": True})

    def test_template_kwargs_empty_for_non_qwen(self) -> None:
        from core.execution_policy import _policy_non_thinking

        kw = template_kwargs_for_thinking_policy(
            _policy_non_thinking(),
            model_path="/models/gpt-oss.gguf",
            model_name="Gpt-Oss-20B",
        )
        self.assertEqual(kw, {})


if __name__ == "__main__":
    unittest.main()
