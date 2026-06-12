from __future__ import annotations

import unittest

from core.output_validation import (
    _degeneration,
    analyze_degeneration,
    validate_output,
)
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


def _art_essay_overview() -> str:
    """Synthetic art-history overview resembling the Jun 12 telemetry case."""
    sections = [
        (
            "Prehistoric Art (30,000 BCE - 3000 BCE)",
            "The roots of human artistic expression can be traced back to the prehistoric era. "
            "Cave paintings, petroglyphs, and sculptures from this period provide a glimpse into "
            "early human societies' spiritual beliefs and daily lives.",
        ),
        (
            "Ancient Civilizations (3000 BCE - 500 CE)",
            "Artistic expression flourished during the ancient civilizations, including Egypt, "
            "Mesopotamia, India, China, and Greece. Egyptian art is characterized by rigidity and "
            "symbolism, while Greek art pursued idealized beauty and harmony.",
        ),
        (
            "Classical Antiquity (500 BCE - 476 CE)",
            "The art of classical antiquity is often associated with the Greek and Roman periods. "
            "Greek sculpture emphasized balance, while Roman art reflected greater diversity of "
            "subjects and architectural mastery.",
        ),
        (
            "Medieval Art (500 CE - 1400 CE)",
            "The Middle Ages saw a shift toward religious themes. Illuminated manuscripts, stained "
            "glass windows, and church murals became common during this period across Europe.",
        ),
        (
            "Renaissance Art (1400 CE - 1600 CE)",
            "The Renaissance marked a revival of interest in classical culture, leading to advances "
            "in perspective, naturalism, and individualism in painting and sculpture.",
        ),
        (
            "Baroque and Rococo (1600 CE - 1750 CE)",
            "The Baroque period was characterized by dramatic contrasts and emotional intensity, "
            "while Rococo favored lighter ornamentation and delicate subjects.",
        ),
        (
            "Modernism (1850 CE - 1970 CE)",
            "Modernism rejected traditional realism, embracing abstract forms, innovation, and "
            "experimentation across painting, sculpture, and design.",
        ),
        (
            "Contemporary Art (1970 CE - Present)",
            "Contemporary art continues to challenge boundaries through installation, performance, "
            "and conceptual work that questions authorship and meaning.",
        ),
        (
            "Digital and AI Art (Present)",
            "Digital tools and artificial intelligence have opened new frontiers for human artistic "
            "expression, blending computation with traditional creative practice.",
        ),
    ]
    parts = ["**Human Artistic Expression: A Historical Overview**"]
    for title, body in sections:
        parts.append(f"**{title}**")
        parts.append(body)
        parts.append(
            "Scholars study how artistic expression in each era reflects broader social change."
        )
    return "\n\n".join(parts)


def _long_essay_with_dispersed_bigrams() -> str:
    paragraphs = []
    for i in range(30):
        paragraphs.append(
            f"Section {i}: Societies across region {i} developed distinct cultural practices over centuries. "
            f"Artists drew inspiration from local landscapes, trade routes, and ritual traditions "
            f"to create works reflecting the values of their era."
        )
    return "\n\n".join(paragraphs)


def _clustered_sentence_repeat() -> str:
    sentence = "The Renaissance was important because it changed art forever."
    return " ".join([sentence] * 4)


def _dispersed_sentence_repeat() -> str:
    sentence = "The Renaissance was important because it changed art forever."
    filler = " ".join(f"token{i}" for i in range(120))
    return f"{sentence} {filler} {sentence} {filler} {sentence} {filler} {sentence}"


class TestOutputValidationDegeneration(unittest.TestCase):
    def test_structured_bullet_lists_are_not_degeneration(self) -> None:
        text = _kathmandu_bullets()
        self.assertFalse(_degeneration(text))
        res = validate_output(text, _contract())
        self.assertNotIn("degeneration", res.issues)

    def test_obvious_token_loop_still_flags(self) -> None:
        text = "loop loop loop loop loop loop loop loop"
        analysis = analyze_degeneration(text)
        self.assertTrue(analysis.flagged)
        self.assertTrue(analysis.retry_eligible)
        res = validate_output(text, _contract())
        self.assertIn("degeneration", res.issues)
        self.assertEqual(res.severity, "high")
        self.assertTrue(res.degeneration_retry_eligible)

    def test_art_essay_regression_not_retry_eligible(self) -> None:
        text = _art_essay_overview()
        analysis = analyze_degeneration(text)
        self.assertFalse(analysis.retry_eligible)
        res = validate_output(text, _contract())
        self.assertNotIn("degeneration", res.issues)

    def test_long_essay_dispersed_bigrams_not_flagged(self) -> None:
        text = _long_essay_with_dispersed_bigrams()
        self.assertFalse(_degeneration(text))
        res = validate_output(text, _contract())
        self.assertNotIn("degeneration", res.issues)

    def test_clustered_sentence_repeat_is_retry_eligible(self) -> None:
        text = _clustered_sentence_repeat()
        analysis = analyze_degeneration(text)
        self.assertTrue(analysis.flagged)
        self.assertTrue(analysis.retry_eligible)
        self.assertTrue(analysis.clustered)
        res = validate_output(text, _contract())
        self.assertEqual(res.severity, "high")

    def test_dispersed_sentence_repeat_not_flagged(self) -> None:
        text = _dispersed_sentence_repeat()
        self.assertFalse(_degeneration(text))
        res = validate_output(text, _contract())
        self.assertNotIn("degeneration", res.issues)

    def test_stopword_only_bigrams_not_flagged(self) -> None:
        segments = [
            "Scholars examine pottery, textiles, and architecture across many regions.",
            "Museums preserve objects that reveal daily life in earlier societies.",
            "Curators compare stylistic choices between coastal and inland communities.",
            "Historians trace how trade influenced decorative motifs over time.",
            "Archaeologists document tools used for carving stone and casting metal.",
            "Archivists maintain records describing festivals, markets, and workshops.",
            "Students study how patronage shaped monumental building projects.",
            "Researchers analyze pigments found in murals inside ancient temples.",
        ]
        text = " ".join(segments + ["from the"] * 6)
        self.assertFalse(_degeneration(text))
        res = validate_output(text, _contract())
        self.assertNotIn("degeneration", res.issues)


if __name__ == "__main__":
    unittest.main()
