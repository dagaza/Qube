from __future__ import annotations

import unittest

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
        sanitized = sanitize_output_for_validation(raw)
        result = validate_output(sanitized, _contract())
        self.assertTrue(result.is_valid)
        self.assertIn(body, sanitized)

    def test_clean_body_unchanged(self) -> None:
        body = "Hello world."
        self.assertEqual(sanitize_output_for_validation(body), body)


if __name__ == "__main__":
    unittest.main()
