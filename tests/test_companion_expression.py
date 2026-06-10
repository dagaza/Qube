"""Tests for L0/L1 expression renderers."""

from __future__ import annotations

import unittest

from core.companion_cognition.expression import render_template, safe_slots, sanitize_slot_value
from core.companion_cognition.message_library import MessageTemplate
from core.companion_cognition.types import CompanionThought


class TestCompanionExpression(unittest.TestCase):
    def test_sanitize_basename_strips_path(self) -> None:
        self.assertEqual(
            sanitize_slot_value("basename", "../../etc/passwd"),
            "passwd",
        )

    def test_template_expansion(self) -> None:
        tpl = MessageTemplate(
            id="t1",
            intent="celebration",
            pattern="{basename} is ready.",
            placeholders=("basename",),
            contexts=("model_download_completed",),
            cooldown_hours=1,
        )
        thought = CompanionThought(
            intent="celebration",
            mood="warm",
            energy="low",
            slots={"basename": "model.gguf"},
        )
        line = render_template(tpl, safe_slots(thought))
        self.assertIn("model.gguf", line)


if __name__ == "__main__":
    unittest.main()
