"""Acronym definitions from conversation history (plan S5)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.discourse_acronyms import (  # noqa: E402
    expand_acronyms,
    extract_acronym_definitions,
)


def _assistant(text: str) -> dict[str, str]:
    return {"role": "assistant", "content": text}


def _user(text: str) -> dict[str, str]:
    return {"role": "user", "content": text}


class ExtractAcronymDefinitionsTests(unittest.TestCase):
    def test_definition_table(self) -> None:
        cases = [
            # (text, expected definitions)
            ("Actor-network theory (ANT) is a sociological approach.", {"ANT": "Actor-network theory"}),
            ("Multiple sclerosis (MS) is an autoimmune disease.", {"MS": "Multiple sclerosis"}),
            ("The World Health Organization (WHO) said so.", {"WHO": "World Health Organization"}),
            ("the Food and Drug Administration (FDA) approved it", {"FDA": "Food and Drug Administration"}),
            ("the Department of Defense (DoD) budget", {"DoD": "Department of Defense"}),
            ("Crohn's disease (CD) affects the gut", {"CD": "Crohn's disease"}),
            ("Approved by the FDA (Food and Drug Administration).", {"FDA": "Food and Drug Administration"}),
            ("messenger RNA (mRNA) vaccines", {}),  # initials m-R don't spell M-R-N-A
            ("a theory (ANT) of everything", {}),  # initials don't match
            ("deoxyribonucleic acid (DNA)", {}),  # not word initials: not guessed
            ("see the table (ID) below", {}),
            ("the result (p) was significant", {}),  # one letter is not an acronym
            ("Ants (ant) are insects", {}),  # lowercase acronym rejected
            ("Microsoft (MS) shares", {}),  # one word cannot supply two letters
            ("No acronyms here at all.", {}),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(extract_acronym_definitions([_assistant(text)]), expected)

    def test_user_and_assistant_messages_both_count(self) -> None:
        history = [
            _user("Tell me about Generative adversarial networks (GANs)"),
            _assistant("Sure. Reinforcement learning (RL) is different."),
        ]
        self.assertEqual(
            extract_acronym_definitions(history),
            {"GANs": "Generative adversarial networks", "RL": "Reinforcement learning"},
        )

    def test_conflicting_definitions_are_dropped(self) -> None:
        history = [
            _assistant("Multiple sclerosis (MS) is a disease."),
            _user("Is that the same MS as mass spectrometry (MS)?"),
        ]
        self.assertEqual(extract_acronym_definitions(history), {})

    def test_same_definition_repeated_is_kept(self) -> None:
        history = [
            _assistant("Multiple sclerosis (MS) is a disease."),
            _assistant("multiple sclerosis (MS) has relapsing forms."),
        ]
        self.assertEqual(extract_acronym_definitions(history), {"MS": "Multiple sclerosis"})

    def test_only_recent_turns_are_read(self) -> None:
        history = [_assistant("Actor-network theory (ANT) is …")] + [
            _user(f"filler {i}") for i in range(6)
        ]
        self.assertEqual(extract_acronym_definitions(history), {})
        self.assertEqual(
            extract_acronym_definitions(history, max_turns=7),
            {"ANT": "Actor-network theory"},
        )

    def test_empty_and_malformed_history(self) -> None:
        self.assertEqual(extract_acronym_definitions(None), {})
        self.assertEqual(extract_acronym_definitions([]), {})
        self.assertEqual(extract_acronym_definitions([{"role": "user"}, "x", {"content": 3}]), {})


class ExpandAcronymsTests(unittest.TestCase):
    DEFS = {
        "ANT": "actor-network theory",
        "MS": "multiple sclerosis",
        "US": "United States",
    }

    def test_expansion_table(self) -> None:
        cases = [
            ("how ANT analyzes AI", "how actor-network theory (ANT) analyzes AI", [("ANT", "actor-network theory")]),
            ("how MS is treated", "how multiple sclerosis (MS) is treated", [("MS", "multiple sclerosis")]),
            # Undefined acronyms stay as written.
            ("what does the WHO recommend", "what does the WHO recommend", []),
            # Case-sensitive, whole-word: the pronoun "us", the insect "ant", "ANTs", "MSc".
            ("tell us about ant colonies", "tell us about ant colonies", []),
            ("ANTs and an MSc degree", "ANTs and an MSc degree", []),
            ("ANT-based methods", "ANT-based methods", []),
            # Long form already present: no double expansion.
            ("actor-network theory (ANT) and AI", "actor-network theory (ANT) and AI", []),
            ("Actor-Network Theory and ANT", "Actor-Network Theory and ANT", []),
            # Only the first use expands; multiple acronyms expand in text order.
            ("ANT vs ANT", "actor-network theory (ANT) vs ANT", [("ANT", "actor-network theory")]),
            (
                "MS rates in the US",
                "multiple sclerosis (MS) rates in the United States (US)",
                [("MS", "multiple sclerosis"), ("US", "United States")],
            ),
            ("", "", []),
        ]
        for text, expected, applied in cases:
            with self.subTest(text=text):
                self.assertEqual(expand_acronyms(text, self.DEFS), (expected, applied))

    def test_no_definitions_is_identity(self) -> None:
        self.assertEqual(expand_acronyms("how ANT works", {}), ("how ANT works", []))


if __name__ == "__main__":
    unittest.main()
