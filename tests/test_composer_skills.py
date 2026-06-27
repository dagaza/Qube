"""Tests for composer skill tokens (Phase A)."""

from __future__ import annotations

import unittest

from core.composer_attachments import ComposerAttachment
from core.composer_skills import (
    format_skill_token,
    list_skill_mentions_for_palette,
    parse_composer_input,
    strip_all_composer_tokens_for_display,
    strip_skill_tokens,
    substantive_composer_prompt,
)


class ComposerSkillTokenTests(unittest.TestCase):
    def test_format_and_strip_skill_token(self) -> None:
        token = format_skill_token("decision_analysis")
        self.assertEqual(token, "@[skill:decision_analysis]")
        enforced, remainder = strip_skill_tokens(f"{token} Which option is better?")
        self.assertEqual(enforced, ("decision_analysis",))
        self.assertEqual(remainder, "Which option is better?")

    def test_parse_composer_input_splits_routing_and_skills(self) -> None:
        text = (
            "@[skill:problem_solving] @[tool:internet] "
            "@[file:doc.pdf] explain this"
        )
        clean, attachments, enforced = parse_composer_input(text)
        self.assertEqual(enforced, ("problem_solving",))
        self.assertEqual(clean, "explain this")
        self.assertEqual(len(attachments), 2)
        kinds = {a.kind for a in attachments}
        self.assertEqual(kinds, {"tool", "file"})

    def test_duplicate_skill_tokens_deduped(self) -> None:
        text = "@[skill:writing_assistance] @[skill:writing_assistance] hi"
        clean, _attachments, enforced = parse_composer_input(text)
        self.assertEqual(enforced, ("writing_assistance",))
        self.assertEqual(clean, "hi")

    def test_strip_all_tokens_for_display(self) -> None:
        raw = "@[skill:prompt_engineering] @[tool:memory] hello"
        self.assertEqual(strip_all_composer_tokens_for_display(raw), "hello")

    def test_list_skill_mentions_for_palette(self) -> None:
        rows = list_skill_mentions_for_palette(query="decision")
        ids = [r.id for r in rows]
        self.assertIn("decision_analysis", ids)

    def test_skill_tokens_never_become_routing_attachments(self) -> None:
        _clean, attachments, enforced = parse_composer_input(
            "@[skill:software_engineering] fix bug"
        )
        self.assertEqual(enforced, ("software_engineering",))
        self.assertEqual(attachments, [])

    def test_substantive_composer_prompt_strips_tokens(self) -> None:
        self.assertIsNone(substantive_composer_prompt("@[tool:internet]"))
        self.assertEqual(
            substantive_composer_prompt("@[tool:internet] What is the capital?"),
            "What is the capital?",
        )


if __name__ == "__main__":
    unittest.main()
