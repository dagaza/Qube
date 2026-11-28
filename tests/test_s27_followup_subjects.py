"""Plan S27: follow-up subjects on the held-out sets, fixed cause by cause (D63).

Cause 1: a sentence-initial auxiliary, modal, wh-word or connective is not part of
a name. ``_PROPER_NAME`` accepts a lowercase word after "the" / "of" / "and" (for
"Slay the Spire"), so "Is the second one stable?" gave the topic "Is the second",
and both patterns glued the first word onto a real name ("Does Apple").

Cause 2: an earlier statement or instruction whose subject no rule found ("My
father has type 2 diabetes.", "Explain how a heat pump works."). The fallbacks run
only on turns already answered (``earlier_turn``), and never on a question: an
earlier question reaches the search as itself (S36's ``prior_question``).

Cause 3: the follow-up's own noun replaced the subject. "What's the contribution
limit this year?" is a facet question like "What is the contribution limit?"
(V91, D64), and "What about in winter?" is a facet, not the new subject "in winter".
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.discourse_state import _extract_topic_candidate, extract_assistant_referent  # noqa: E402
from core.discourse_subject import extract_turn_subject  # noqa: E402
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


class TestEarlierStatementSubject(unittest.TestCase):
    """Cause 2: the subject of an earlier statement or instruction."""

    def test_an_earlier_statement_names_its_subject(self):
        for turn, subject in (
            ("My father has type 2 diabetes.", "type 2 diabetes"),
            ("Our cat was diagnosed with kidney disease.", "kidney disease"),
            ("My daughter was just diagnosed with asthma.", "asthma"),
            ("My doctor prescribed sertraline for anxiety.", "sertraline"),
            ("My dentist recommended a water flosser.", "water flosser"),
            ("Our main database is PostgreSQL 16.", "PostgreSQL 16"),
            ("We just moved to Rotterdam.", "Rotterdam"),
            ("Explain how a heat pump works.", "heat pump"),
        ):
            with self.subTest(turn=turn):
                self.assertEqual(extract_turn_subject(turn, earlier_turn=True), subject)

    def test_the_current_turn_is_unchanged(self):
        for turn in ("My father has type 2 diabetes.", "We just moved to Rotterdam.", "Explain how a heat pump works."):
            with self.subTest(turn=turn):
                self.assertIsNone(extract_turn_subject(turn))

    def test_an_earlier_question_gets_no_fallback_subject(self):
        for turn in (
            "When is the Heron app launching?",
            "Where was Mozart born?",
            "Which company makes the Model Y?",
            "How many days of annual leave do I get at Kestrel?",
            "How long does it take to fly from London to Tokyo?",
        ):
            with self.subTest(turn=turn):
                self.assertIsNone(extract_turn_subject(turn, earlier_turn=True))

    def test_the_follow_up_carries_the_subject(self):
        for prior, answer, turn, expected in (
            ("My father has type 2 diabetes.", "That is common; diet and exercise help.",
             "Is it hereditary?", "type 2 diabetes Is it hereditary?"),
            ("We just moved to Rotterdam.", "Welcome! It is a lively port city.",
             "What's the weather like there?", "Rotterdam What's the weather like there?"),
            ("Explain how a heat pump works.", "It moves heat from the outside air into the house.",
             "Is it noisy?", "heat pump Is it noisy?"),
            ("My dentist recommended a water flosser.", "They clean between teeth with a jet of water.",
             "How often should I use it?", "water flosser How often should I use it?"),
        ):
            with self.subTest(turn=turn):
                _fu, state, r = build_discourse_resolution(turn, [_u(prior), _a(answer)])
                self.assertEqual(r.web_text, expected)
                self.assertEqual(r.retrieval_text, expected)



class TestFacetFollowUps(unittest.TestCase):
    """Cause 3: a facet of the subject is not a new subject."""

    ROTH = [_u("What is a Roth IRA?"), _a("A retirement account funded with after-tax money.")]
    PUMP = [_u("What is a heat pump?"), _a("It moves heat from outside air into a home.")]

    def _web(self, history, turn: str) -> str:
        return build_discourse_resolution(turn, history)[2].web_text

    def test_a_facet_question_with_what_s_or_a_time_phrase(self):
        for turn in (
            "What's the contribution limit this year?",
            "What is the contribution limit this year?",
            "What's the income limit now?",
        ):
            with self.subTest(turn=turn):
                self.assertEqual(self._web(self.ROTH, turn), f"Roth IRA {turn}")

    def test_a_question_that_names_its_own_subject_is_not_a_facet(self):
        for turn in ("What's the population of Iceland?", "What is the capital of Peru?", "What's the rent on my flat?"):
            with self.subTest(turn=turn):
                self.assertEqual(self._web(self.ROTH, turn), turn)

    def test_what_about_a_preposition_is_a_facet(self):
        for turn, expected in (
            ("What about in winter?", "heat pump in winter"),
            ("And what about for a small flat?", "heat pump for a small flat"),
        ):
            with self.subTest(turn=turn):
                self.assertEqual(self._web(self.PUMP, turn), expected)

    def test_what_about_a_name_is_still_a_new_subject(self):
        _fu, state, r = build_discourse_resolution("What about Rust?", self.PUMP)
        self.assertEqual(r.web_text, "What about Rust?")
        self.assertEqual(state.subject, "Rust")


if __name__ == "__main__":
    unittest.main()
