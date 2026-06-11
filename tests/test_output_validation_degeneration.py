from __future__ import annotations

import unittest

from core.output_validation import _degeneration, validate_output
from core.prompt_contract import PromptContract


def _contract() -> PromptContract:
    return PromptContract(
        mode="messages",
        chat_format="chatml",
        prompt=None,
        messages=[{"role": "user", "content": "Hi"}],
        stop=[],
        template_source="fallback",
        confidence="medium",
    )


def _kathmandu_bullets() -> str:
    items = [
        ("Newari Folk Music (Dhimayika)", "Traditional music deeply rooted in Newar rituals."),
        ("Chants and Hymns", "Devotional songs used extensively in Buddhist monasteries."),
        ("Sarangi Playing", "Traditional bowed string instrument narratives."),
        ("Madal Percussion", "Complex drumming patterns during religious processions."),
        ("Tibetan Influence Music", "Musical elements from Himalayan cultural exchange."),
        ("Contemporary Fusion Bands", "Modern groups blending Nepali and Western instruments."),
        ("Festival Orchestras", "Large ensembles for major religious festivals."),
        ("Kathak Influences", "Intricate rhythmic footwork in local dance forms."),
        ("Acoustic Storytelling Music", "Simple accompaniment for bardic narratives."),
        ("Modern Pop Scene", "Younger musicians using Nepali lyrics in pop formats."),
    ]
    return "\n".join(f"- **{title}** — {desc}" for title, desc in items)


class TestOutputValidationDegeneration(unittest.TestCase):
    def test_structured_bullet_lists_are_not_degeneration(self) -> None:
        text = _kathmandu_bullets()
        self.assertFalse(_degeneration(text))
        res = validate_output(text, _contract())
        self.assertNotIn("degeneration", res.issues)

    def test_obvious_token_loop_still_flags(self) -> None:
        text = "loop loop loop loop loop loop loop loop"
        self.assertTrue(_degeneration(text))
        res = validate_output(text, _contract())
        self.assertIn("degeneration", res.issues)


if __name__ == "__main__":
    unittest.main()
