from __future__ import annotations

import unittest

from core.output_validation import validate_output
from core.prompt_contract import PromptContract


def _contract() -> PromptContract:
    return PromptContract(
        mode="messages",
        chat_format="chatml",
        prompt=None,
        messages=[{"role": "user", "content": "Hi"}],
        stop=["<|im_end|>"],
        template_source="fallback",
        confidence="medium",
    )


class TestOutputValidation(unittest.TestCase):
    def test_template_leakage_is_invalid_high(self) -> None:
        res = validate_output("[INST] Hello [/INST]", _contract())
        self.assertFalse(res.is_valid)
        self.assertIn("template_leakage", res.issues)
        self.assertEqual(res.severity, "high")

    def test_role_confusion_is_invalid_high(self) -> None:
        res = validate_output("User: Hello\nAssistant: Hi", _contract())
        self.assertFalse(res.is_valid)
        self.assertIn("role_confusion", res.issues)
        self.assertEqual(res.severity, "high")

    def test_valid_output_passes(self) -> None:
        text = "Gravity is a force that attracts objects with mass."
        res = validate_output(text, _contract())
        self.assertTrue(res.is_valid)
        self.assertEqual(res.issues, [])
        self.assertEqual(res.severity, "low")

    def test_harmony_channel_token_is_template_leakage_high(self) -> None:
        res = validate_output("Answer <|channel|> tail", _contract())
        self.assertFalse(res.is_valid)
        self.assertIn("template_leakage", res.issues)
        self.assertEqual(res.severity, "high")

    def test_bracketed_meta_only_is_meta_preamble_high(self) -> None:
        res = validate_output(
            "[The user refers to a file mentioning Dr. Evelyn Vance.]",
            _contract(),
        )
        self.assertFalse(res.is_valid)
        self.assertIn("meta_preamble", res.issues)
        self.assertEqual(res.severity, "high")
