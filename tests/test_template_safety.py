"""Tests for core/template_safety.py."""

from __future__ import annotations

import unittest

from core.template_safety import is_unsafe_chat_template


class TemplateSafetyTests(unittest.TestCase):
    def test_safe_minimal_template(self) -> None:
        ok, reasons = is_unsafe_chat_template("{{ bos_token }}{% for message in messages %}{{ message['role'] }}{% endfor %}")
        self.assertFalse(ok)
        self.assertEqual(reasons, [])

    def test_channel_token_unsafe(self) -> None:
        ok, reasons = is_unsafe_chat_template("prefix <|channel|> suffix")
        self.assertTrue(ok)
        self.assertTrue(any("channel" in r for r in reasons))

    def test_start_assistant_unsafe(self) -> None:
        ok, reasons = is_unsafe_chat_template("<|start|>assistant\n")
        self.assertTrue(ok)
        self.assertTrue(any("start" in r and "assistant" in r for r in reasons))

    def test_empty_not_unsafe(self) -> None:
        self.assertEqual(is_unsafe_chat_template(""), (False, []))
        self.assertEqual(is_unsafe_chat_template("   "), (False, []))


if __name__ == "__main__":
    unittest.main()
