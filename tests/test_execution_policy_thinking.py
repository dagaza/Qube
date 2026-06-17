"""Execution policy + Think toggle resolver tests."""
from __future__ import annotations

import unittest

from core.execution_policy import (
    ExecutionPolicy,
    execution_policy_debug_fields,
    resolve_execution_policy,
    resolve_user_think_enabled,
)
from core.model_reasoning_profile import ModelReasoningProfile
from core.prompt_template_router import resolve_reasoning_mode as router_resolve_reasoning_mode


def _qwen_profile() -> ModelReasoningProfile:
    return ModelReasoningProfile(
        model_name="Qwen3.5-9B",
        supports_thinking_tokens=True,
        thinking_token_patterns=["</think>"],
        default_mode="thinking",
        reasoning_confidence=1.0,
        detection_method="tokenizer_scan",
    )


class TestResolveUserThinkEnabled(unittest.TestCase):
    def test_unset_override_defaults_off(self) -> None:
        self.assertFalse(resolve_user_think_enabled(_qwen_profile(), None))

    def test_explicit_override_wins(self) -> None:
        self.assertTrue(resolve_user_think_enabled(_qwen_profile(), True))
        self.assertFalse(resolve_user_think_enabled(_qwen_profile(), False))


class TestResolveReasoningMode(unittest.TestCase):
    def test_think_off_uses_disabled_even_with_hard_enforcement(self) -> None:
        pol = ExecutionPolicy(
            execution_mode="direct",
            allow_thinking_tokens=False,
            strip_thinking_output=True,
            ui_display_thinking=False,
            tts_strip_thinking=True,
            enforcement_mode="hard",
        )
        self.assertEqual(router_resolve_reasoning_mode(pol), "disabled")

    def test_think_on_soft_uses_soft(self) -> None:
        pol = ExecutionPolicy(
            execution_mode="thinking",
            allow_thinking_tokens=True,
            strip_thinking_output=False,
            ui_display_thinking=True,
            tts_strip_thinking=False,
            enforcement_mode="soft",
        )
        self.assertEqual(router_resolve_reasoning_mode(pol), "soft")


class TestResolveExecutionPolicy(unittest.TestCase):
    def test_internal_qwen_defaults_to_direct_when_override_unset(self) -> None:
        pol = resolve_execution_policy(_qwen_profile(), None, "internal")
        self.assertFalse(pol.allow_thinking_tokens)
        self.assertTrue(pol.strip_thinking_output)
        self.assertEqual(pol.execution_mode, "direct")

    def test_internal_qwen_user_on_enables_thinking(self) -> None:
        pol = resolve_execution_policy(_qwen_profile(), True, "internal")
        self.assertTrue(pol.allow_thinking_tokens)
        self.assertFalse(pol.strip_thinking_output)
        self.assertEqual(pol.execution_mode, "thinking")


class TestExecutionPolicyDebugFields(unittest.TestCase):
    def test_includes_reasoning_and_template_kwargs(self) -> None:
        pol = resolve_execution_policy(_qwen_profile(), False, "internal")
        fields = execution_policy_debug_fields(
            pol,
            reasoning_mode="disabled",
            chat_template_kwargs={"enable_thinking": False},
        )
        self.assertFalse(fields["allow_thinking_tokens"])
        self.assertTrue(fields["strip_thinking_output"])
        self.assertEqual(fields["reasoning_mode"], "disabled")
        self.assertEqual(fields["chat_template_kwargs"], {"enable_thinking": False})


if __name__ == "__main__":
    unittest.main()
