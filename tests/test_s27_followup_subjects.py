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

Cause 4: references the tracker missed. "he" / "her" for a person named only by a
role ("My father has type 2 diabetes." -> "Which foods should he avoid?"), on the
web; library and memory keep S36's previous question for these (D51). "that" as
the subject of a question ("Is that better than normal floss?") and a locative
"there" inside the sentence ("What's the weather like there in April?").

Cause 5: the referent is in the assistant's answer. A who / where / which-N
question's answer names a person, place or entity, used for the next turn only when
its reference fits ("there" a place, "he" a person, "it" the answer to "which
company ..."). An ordinal ("the second one", "the last option") picks an item of a
numbered, bulleted or inline list in the previous answer.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.discourse_state import _extract_topic_candidate, extract_assistant_referent  # noqa: E402
from core.discourse_subject import (  # noqa: E402
    REASON_LIST_ITEM,
    extract_turn_subject,
    has_back_reference,
    normalize_subject,
    track_subject,
)
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



class TestMissedReferences(unittest.TestCase):
    """Cause 4: back-references the tracker missed."""

    def _resolve(self, prior: str, answer: str, turn: str):
        return build_discourse_resolution(turn, [_u(prior), _a(answer)])[2]

    def test_a_people_pronoun_carries_the_subject_on_the_web(self):
        for prior, answer, turn, web in (
            ("My father has type 2 diabetes.", "Diet and exercise help a lot.",
             "Which foods should he avoid?", "type 2 diabetes Which foods should he avoid?"),
            ("Our cat was diagnosed with kidney disease.", "Many cats live for years with it.",
             "What should we feed her?", "kidney disease What should we feed her?"),
        ):
            with self.subTest(turn=turn):
                r = self._resolve(prior, answer, turn)
                self.assertEqual(r.web_text, web)
                # Library and memory: S36's previous question, unchanged (D51).
                self.assertEqual(r.retrieval_text, f"{prior} {turn}")
                self.assertEqual(r.retrieval_rewrite_reason, "prior_question")

    def test_that_as_the_subject_of_a_question(self):
        r = self._resolve("My dentist recommended a water flosser.", "They clean between teeth with a jet of water.",
                          "Is that better than normal floss?")
        self.assertEqual(r.web_text, "water flosser Is that better than normal floss?")
        for turn in ("What does that cost?", "How much does that cost?", "Is that true?"):
            with self.subTest(turn=turn):
                self.assertTrue(has_back_reference(turn))

    def test_there_inside_the_sentence(self):
        r = self._resolve("Which city is the capital of Australia?", "The capital of Australia is Canberra.",
                          "What's the weather like there in April?")
        self.assertEqual(r.web_text, "Canberra What's the weather like there in April?")
        for turn in ("How many people live there?", "Can I get there by train?"):
            with self.subTest(turn=turn):
                self.assertTrue(has_back_reference(turn))

    def test_existential_there_is_not_a_reference(self):
        for turn in ("Are there good international schools?", "Is there a cure?", "Were there any survivors?",
                     "Why are there so many?", "I think there is a problem"):
            with self.subTest(turn=turn):
                self.assertFalse(has_back_reference(turn))

    def test_a_closing_called_is_not_part_of_the_subject(self):
        self.assertEqual(normalize_subject("my cat called"), "my cat")
        self.assertEqual(normalize_subject("the dog named"), "dog")



class TestAnswerReferents(unittest.TestCase):
    """Cause 5: the next turn refers to what the answer named."""

    def _web(self, prior: str, answer: str, turn: str) -> str:
        return build_discourse_resolution(turn, [_u(prior), _a(answer)])[2].web_text

    def test_the_answer_names_the_referent(self):
        for prior, answer, turn, web in (
            ("Where was Beethoven born?", "Beethoven was born in Bonn in 1770.",
             "How do I get there from Cologne?", "Bonn How do I get there from Cologne?"),
            ("Which company makes the Galaxy S24?", "The Galaxy S24 is made by Samsung.",
             "Where is it based?", "Samsung Where is it based?"),
            ("Which country has the tallest building in the world?",
             "The tallest building, the Burj Khalifa, is in the United Arab Emirates, in Dubai.",
             "Do I need a visa to travel there?", "United Arab Emirates Do I need a visa to travel there?"),
            ("Who wrote Thinking, Fast and Slow?", "Daniel Kahneman wrote it; it was published in 2011.",
             "Did he win a Nobel Prize?", "Daniel Kahneman Did he win a Nobel Prize?"),
        ):
            with self.subTest(turn=turn):
                self.assertEqual(self._web(prior, answer, turn), web)

    def test_a_reference_that_does_not_fit_keeps_the_question_subject(self):
        # "it" after a where-question is the thing asked about, not the place.
        web = self._web("Where is the Eiffel Tower?", "The Eiffel Tower is in Paris, France.", "How tall is it?")
        self.assertIn("Eiffel Tower", web)
        self.assertNotIn("Paris", web)
        # "his" is a person; the answer named a place.
        web = self._web("Where was Beethoven born?", "Beethoven was born in Bonn in 1770.", "What was his first symphony?")
        self.assertNotIn("Bonn", web)

    def test_an_ordinal_picks_the_list_item(self):
        for answer, turn, web in (
            ("Popular choices are:\n1. Python, for its readable syntax.\n2. JavaScript, for the web.\n3. Scratch, for children.",
             "Is the first one free?", "Is Python free?"),
            ("A few options:\n- lower the thermostat\n- insulate the loft\n- draught-proof the windows",
             "How much does the second option cost?", "How much does insulate the loft cost?"),
            ("Good sources include oranges, kiwis and strawberries.", "Are the last ones easy to grow?",
             "Are strawberries easy to grow?"),
        ):
            with self.subTest(turn=turn):
                self.assertEqual(self._web("Which ones would you suggest?", answer, turn), web)

    def test_an_ordinal_without_a_list_is_not_resolved(self):
        for prior, answer, turn in (
            ("What is the shingles vaccine?", "Shingrix is given as two doses, two to six months apart.",
             "Is the second dose worse than the first?"),
            ("Tell me about Rust.", "Rust is a systems programming language.", "Is the second one faster?"),
        ):
            with self.subTest(turn=turn):
                track = track_subject([_u(prior), _a(answer)], turn)
                self.assertNotEqual(track.reasons, (REASON_LIST_ITEM,))
                self.assertNotIn("Shingrix", self._web(prior, answer, turn))


if __name__ == "__main__":
    unittest.main()
