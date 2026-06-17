from __future__ import annotations

import unittest

from core.execution_policy import ExecutionPolicy
from core.output_validation import validate_output
from core.output_validation_sanitize import sanitize_output_for_validation
from core.prompt_contract import PromptContract


def _contract() -> PromptContract:
    return PromptContract(
        mode="messages",
        chat_format="chat_template.default",
        prompt=None,
        messages=[{"role": "user", "content": "Hi"}],
        stop=[],
        template_source="gguf",
        confidence="high",
    )


class TestOutputValidationSanitize(unittest.TestCase):
    def test_gemma_thought_prefix_sanitized_passes_validation(self) -> None:
        body = "Kathmandu is the capital of Nepal."
        raw = f"<|channel>thought\nPlanning the answer.\n\n{body}"
        self.assertFalse(validate_output(raw, _contract()).is_valid)
        sanitized = sanitize_output_for_validation(raw, harmony_active=False)
        result = validate_output(sanitized, _contract())
        self.assertTrue(result.is_valid)
        self.assertIn(body, sanitized)

    def test_clean_body_unchanged(self) -> None:
        body = "Hello world."
        self.assertEqual(sanitize_output_for_validation(body), body)

    def test_preserves_thinking_when_policy_allows(self) -> None:
        raw = "<think>plan</think>Answer."
        pol = ExecutionPolicy(
            execution_mode="thinking",
            allow_thinking_tokens=True,
            strip_thinking_output=False,
            ui_display_thinking=True,
            tts_strip_thinking=False,
            enforcement_mode="soft",
        )
        out = sanitize_output_for_validation(raw, policy=pol)
        self.assertIn("<think>", out)
        self.assertIn("Answer.", out)

    def test_strips_thinking_when_policy_requires(self) -> None:
        raw = "<think>plan</think>Answer."
        pol = ExecutionPolicy(
            execution_mode="direct",
            allow_thinking_tokens=False,
            strip_thinking_output=True,
            ui_display_thinking=False,
            tts_strip_thinking=True,
            enforcement_mode="hard",
        )
        out = sanitize_output_for_validation(raw, policy=pol)
        self.assertNotIn("<think>", out)
        self.assertIn("Answer.", out)


if __name__ == "__main__":
    unittest.main()
