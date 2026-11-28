"""Plan S27: follow-up subjects on the held-out sets, fixed cause by cause (D63).

Cause 1: a sentence-initial auxiliary, modal, wh-word or connective is not part of
a name. ``_PROPER_NAME`` accepts a lowercase word after "the" / "of" / "and" (for
"Slay the Spire"), so "Is the second one stable?" gave the topic "Is the second",
and both patterns glued the first word onto a real name ("Does Apple").
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.discourse_state import _extract_topic_candidate, extract_assistant_referent  # noqa: E402
from core.query_resolution_evaluation import build_discourse_resolution  # noqa: E402


def _u(text: str) -> dict[str, str]:
    return {"role": "user", "content": text}


def _a(text: str) -> dict[str, str]:
    return {"role": "assistant", "content": text}


class TestLeadFunctionWords(unittest.TestCase):
    """Cause 1: the turn's own capitalised first word is not a name."""

    def test_no_name_is_left(self):
        for turn in (
            "Is the second one stable?",
            "Does the heat pump make a big difference in winter?",
            "Is the second dose worse than the first?",
            "And the second point?",
            "Can the landlord come into the flat without telling me?",
            "Did the cover crops change grain yield?",
        ):
            with self.subTest(turn=turn):
                self.assertEqual(_extract_topic_candidate(turn), (None, "unknown"))

    def test_the_name_after_the_word_is_kept(self):
        for turn, name in (
            ("Does Apple make good laptops?", "Apple"),
            ("Is Python fast enough for games?", "Python"),
            ("Will the Fed cut rates?", "Fed"),
            ("Was the Treaty of Versailles fair?", "Treaty of Versailles"),
            ("Did Project Osprey stay within budget?", "Project Osprey"),
        ):
            with self.subTest(turn=turn):
                self.assertEqual(_extract_topic_candidate(turn)[0], name)

    def test_names_with_connectors_are_unchanged(self):
        for turn, name in (
            ("Lord of the Rings is long.", "Lord of the Rings"),
            ("Slay the Spire is fun.", "Slay the Spire"),
            ("Marie Curie discovered radium.", "Marie Curie"),
            ("Tell me about the Legend of Zelda.", "the Legend of Zelda"),
        ):
            with self.subTest(turn=turn):
                self.assertEqual(_extract_topic_candidate(turn)[0], name)

    def test_an_assistant_answer_opening_with_an_auxiliary(self):
        self.assertIsNone(extract_assistant_referent("Is the second option cheaper? Usually, yes."))

    def test_the_follow_up_keeps_no_junk_subject(self):
        history = [_u("We're rewriting our backend in Go."), _a("Go is a good fit for network services.")]
        _fu, state, _r = build_discourse_resolution("Is the garbage collector a problem for latency?", history)
        self.assertNotEqual(state.subject, "Is the garbage")
        self.assertNotEqual(state.active_topic, "Is the garbage")


class TestInTurnReference(unittest.TestCase):
    """V90: a reference that points inside the turn does not carry the earlier subject.

    "How much is the deposit and how is it protected?": "it" is the deposit. The
    web path counted it as a back-reference; S36's ``refers_inside_turn`` was
    used only for library and memory search.
    """

    def _web(self, prior: str, answer: str, turn: str) -> str:
        _fu, _st, r = build_discourse_resolution(turn, [_u(prior), _a(answer)])
        return r.web_text

    def test_the_turn_is_searched_as_it_is(self):
        for prior, answer, turn in (
            ("Tell me about Kestrel.", "Kestrel is a software company.",
             "How much is the deposit and how is it protected?"),
            ("What is the singular value decomposition?", "It factors a matrix into three matrices.",
             "Is my bike covered if it's stolen from the shared bike store?"),
            ("What's the rent on my flat?", "It is 950 a month.",
             "When is the Sintra day trip and how do we get there?"),
        ):
            with self.subTest(turn=turn):
                self.assertEqual(self._web(prior, answer, turn), turn)

    def test_a_reference_out_of_the_turn_still_carries_the_subject(self):
        for prior, answer, turn, expected in (
            ("What is the singular value decomposition?", "It factors a matrix into three matrices.",
             "How does it relate to PCA?", "singular value decomposition How does it relate to PCA?"),
            ("Tell me about Kestrel.", "Kestrel is a software company.",
             "Is it profitable and how big is it?", "Kestrel Is it profitable and how big is it?"),
            ("What is a heat pump?", "A heat pump moves heat from outside air into a home.",
             "How efficient is it in winter?", "heat pump How efficient is it in winter?"),
        ):
            with self.subTest(turn=turn):
                self.assertEqual(self._web(prior, answer, turn), expected)


if __name__ == "__main__":
    unittest.main()
