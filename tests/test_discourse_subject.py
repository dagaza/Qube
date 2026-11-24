"""Conversation subject / facet tracking and follow-up retrieval text (plan S6)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.discourse_intent import FOLLOW_UP_SUPPRESS_THRESHOLD, classify_follow_up  # noqa: E402
from core.discourse_state import DiscourseState, update_discourse_state  # noqa: E402
from core.discourse_subject import (  # noqa: E402
    RELATION_CONTINUATION,
    RELATION_NEW_SUBJECT,
    RELATION_NO_PRIOR,
    RELATION_NONE,
    extract_turn_subject,
    mentions_subject,
    normalize_subject,
    track_subject,
)
from core.knowledge.scientific_query_planner import plan_scientific_query  # noqa: E402
from core.query_resolution_evaluation import build_discourse_resolution  # noqa: E402

_FIXTURES = ROOT / "eval" / "fixtures" / "knowledge_quality"


def _u(text: str) -> dict[str, str]:
    return {"role": "user", "content": text}


def _a(text: str) -> dict[str, str]:
    return {"role": "assistant", "content": text}


_OAUTH = [
    _u("What is OAuth?"),
    _a("OAuth is an authorization framework. Would you like an example of the authorization code flow?"),
]
_MS = [
    _u("What is multiple sclerosis?"),
    _a("Multiple sclerosis (MS) is a chronic autoimmune disease. Would you like to know how MS is treated?"),
]
_ANT = [
    _u("What is actor-network theory?>"),
    _a("Actor-network theory (ANT) is a sociological approach that treats human and non-human entities as actors."),
]
_PYTHON = [
    _u("What's new in Python 3.13?"),
    _a("Python 3.13 added an experimental free-threaded build and a new REPL."),
]
_HEAT_PUMP = [
    _u("How does a heat pump work?"),
    _a("A heat pump moves heat from outside air into your home using a refrigerant cycle."),
]
_STATINS = [
    _u("Are statins effective for primary prevention in older adults?"),
    _a("Evidence is mixed for people over 75; trials like JUPITER included few very old participants."),
]
_NEPAL = [
    _u("What is the capital of Nepal?"),
    _a("Kathmandu is the capital of Nepal."),
]

# (history, turn, expected subject that the web query must contain)
FOLLOW_UPS: list[tuple[list[dict[str, str]], str, str]] = [
    (_ANT, "Yes, please tell me how ANT analyzes Artificial Inteligence for example.", "actor-network theory"),
    (_OAUTH, "Yes, please.", "oauth"),
    (_OAUTH, "Sure, go on", "oauth"),
    (_OAUTH, "Tell me more", "oauth"),
    (_OAUTH, "what about the security risks?", "oauth"),
    (_OAUTH, "How is it different from SAML?", "oauth"),
    (_OAUTH, "And in mobile apps?", "oauth"),
    (_OAUTH, "Why is that?", "oauth"),
    (_MS, "Yes, tell me how MS is treated", "multiple sclerosis"),
    (_MS, "What are the symptoms?", "multiple sclerosis"),
    (_MS, "Is there a cure?", "multiple sclerosis"),
    (_MS, "What are its early signs?", "multiple sclerosis"),
    (_PYTHON, "and what about 3.14?", "python 3.14"),
    (_HEAT_PUMP, "And how much does it cost to install one?", "heat pump"),
    (_STATINS, "what about side effects in that group?", "statins"),
    (_NEPAL, "How many people live there?", "kathmandu"),
    (_NEPAL, "What about its music?", "kathmandu"),
]

# (history, turn, prior-subject word that must NOT be added)
TOPIC_SWITCHES: list[tuple[list[dict[str, str]], str, str]] = [
    (_ANT, "What is the boiling point of ethanol?", "actor"),
    (_ANT, "How do vaccines work?", "actor"),
    (_OAUTH, "What is the capital of France?", "oauth"),
    (_OAUTH, "Tell me about the Roman Empire", "oauth"),
    (_OAUTH, "Yes, and how do vaccines work?", "oauth"),
    (_OAUTH, "Thanks! What is quantum computing?", "oauth"),
    (_OAUTH, "What about Rust?", "oauth"),
    (_OAUTH, "Can you recommend a good laptop?", "oauth"),
    (_OAUTH, "Is it true that vaccines cause autism?", "oauth"),
    (_MS, "What are the symptoms of diabetes?", "sclerosis"),
    (_MS, "Who was Ada Lovelace?", "sclerosis"),
    (_MS, "Is coffee bad for your heart?", "sclerosis"),
    (_HEAT_PUMP, "Let's talk about black holes", "heat"),
    (_STATINS, "Does intermittent fasting improve insulin sensitivity?", "statin"),
]


class FollowUpRetrievalTableTests(unittest.TestCase):
    def test_follow_ups_keep_the_subject(self) -> None:
        for history, turn, subject in FOLLOW_UPS:
            with self.subTest(turn=turn):
                follow_up, state, resolved = build_discourse_resolution(turn, history)
                self.assertEqual(state.subject_relation, RELATION_CONTINUATION, state)
                self.assertTrue(follow_up.active, follow_up)
                self.assertIn(subject, resolved.web_text.lower())
                self.assertNotIn("Regarding", resolved.web_text)

    def test_topic_switches_never_anchor(self) -> None:
        for history, turn, old_word in TOPIC_SWITCHES:
            with self.subTest(turn=turn):
                _fu, state, resolved = build_discourse_resolution(turn, history)
                self.assertNotEqual(state.subject_relation, RELATION_CONTINUATION, state)
                self.assertNotIn(old_word, resolved.web_text.lower())
                self.assertEqual(resolved.web_text, resolved.inference_text)

    def test_history_with_or_without_current_turn_is_the_same(self) -> None:
        # The worker persists the user message before reading history.
        for history, turn, _subject in FOLLOW_UPS[:6]:
            with self.subTest(turn=turn):
                self.assertEqual(track_subject(history, turn), track_subject(history + [_u(turn)], turn))


class SubjectFacetFixtureAcceptanceTests(unittest.TestCase):
    """S6 acceptance on the S2 multi-turn fixtures."""

    TOPIC_SWITCH_IDS = {"mt_topic_switch_ethanol"}

    def _fixtures(self) -> list[dict]:
        out = []
        for path in sorted(_FIXTURES.glob("*.json")):
            fixture = json.loads(path.read_text(encoding="utf-8"))
            if fixture.get("history") and fixture.get("gold_subject"):
                out.append(fixture)
        return out

    def test_acceptance(self) -> None:
        fixtures = self._fixtures()
        self.assertGreaterEqual(len(fixtures), 8)
        follow_ups = [f for f in fixtures if f["id"] not in self.TOPIC_SWITCH_IDS]
        detected = preserved = 0
        for fixture in follow_ups:
            follow_up, _state, resolved = build_discourse_resolution(
                fixture["query_clean"], fixture["history"]
            )
            detected += follow_up.active
            preserved += fixture["gold_subject"].lower() in resolved.web_text.lower()
        self.assertGreaterEqual(detected / len(follow_ups), 0.9)
        self.assertGreaterEqual(preserved / len(follow_ups), 0.9)
        for fixture in fixtures:
            if fixture["id"] not in self.TOPIC_SWITCH_IDS:
                continue
            _fu, state, resolved = build_discourse_resolution(fixture["query_clean"], fixture["history"])
            self.assertEqual(state.subject_relation, RELATION_NEW_SUBJECT)
            self.assertEqual(resolved.web_text, fixture["query_clean"])

    def test_expected_retrieval_texts(self) -> None:
        expected = {
            "mt_ant_followup_live": "actor-network theory how ANT analyzes Artificial Inteligence",
            "kq_affirmation_followup": "OAuth example of the authorization code flow",
            "mt_acronym_ms_treatment": "multiple sclerosis how MS is treated",
            "mt_ellipsis_python": "What's new in Python 3.14?",
            "mt_facet_shift_heat_pump": "heat pump how much does it cost to install one?",
            "mt_pronoun_statins": "statins side effects in that group",
        }
        by_id = {f["id"]: f for f in self._fixtures()}
        for fid, text in expected.items():
            with self.subTest(fixture=fid):
                fixture = by_id[fid]
                _fu, _state, resolved = build_discourse_resolution(fixture["query_clean"], fixture["history"])
                self.assertEqual(resolved.web_text, text)

    def test_planner_entity_comes_from_resolved_query(self) -> None:
        # B3: the medical entity is found in the resolved query, not the raw turn.
        _fu, _state, resolved = build_discourse_resolution("what about side effects in that group?", _STATINS)
        self.assertEqual(plan_scientific_query(resolved.web_text).entity_keywords, ("Statins",))
        self.assertEqual(plan_scientific_query("what about side effects in that group?").entity_keywords, ())


class SubjectStateTests(unittest.TestCase):
    def test_facet_never_replaces_subject(self) -> None:
        history = _ANT + [_u("Yes, please tell me how ANT analyzes Artificial Intelligence.")]
        state = update_discourse_state(history, None, history[-1]["content"])
        self.assertEqual(state.subject, "actor-network theory")
        self.assertEqual(state.facet, "Artificial Intelligence")
        # The legacy slot is unchanged (routing / prompt hints read it).
        self.assertEqual(state.active_topic, "Artificial Intelligence")

    def test_subject_survives_several_facets(self) -> None:
        history = _ANT + [
            _u("How does ANT analyze Artificial Intelligence?"),
            _a("ANT treats AI systems as actors in a network."),
            _u("And what about its critics?"),
        ]
        track = track_subject(history, "And what about its critics?")
        self.assertEqual(track.subject, "actor-network theory")
        self.assertEqual(track.relation, RELATION_CONTINUATION)

    def test_new_subject_replaces_and_unrelated_turn_clears(self) -> None:
        history = _ANT + [_u("How do vaccines work?"), _a("Vaccines train the immune system.")]
        self.assertEqual(track_subject(history, "Why is that?").subject, "vaccines")
        history = _OAUTH + [_u("Can you recommend a good laptop?"), _a("Sure, here are three.")]
        track = track_subject(history, "Why is that?")
        self.assertEqual(track.relation, RELATION_NO_PRIOR)
        self.assertIsNone(track.subject)

    def test_answer_entity_becomes_subject(self) -> None:
        track = track_subject(_NEPAL, "What about its music?")
        self.assertEqual(track.subject, "Kathmandu")

    def test_state_fields_and_dict(self) -> None:
        state = update_discourse_state(_MS + [_u("What are the symptoms?")], None, "What are the symptoms?")
        self.assertEqual(state.subject, "multiple sclerosis")
        self.assertEqual(state.facet, "symptoms")
        d = state.to_dict()
        self.assertEqual(d["discourse_subject"], "multiple sclerosis")
        self.assertEqual(d["discourse_subject_relation"], RELATION_CONTINUATION)
        self.assertEqual(DiscourseState().subject_relation, RELATION_NO_PRIOR)

    def test_stray_punctuation_is_stripped(self) -> None:
        state = update_discourse_state([_u("What is actor-network theory?>")], None, "What is actor-network theory?>")
        self.assertEqual(state.active_topic, "actor-network theory")
        self.assertEqual(state.subject, "actor-network theory")

    def test_composer_tokens_ignored(self) -> None:
        history = [_u("@[tool:evidence] What is OAuth?"), _a("OAuth is an authorization framework.")]
        self.assertEqual(track_subject(history, "@[tool:evidence] Why is that?").subject, "OAuth")


class StatementSubjectTests(unittest.TestCase):
    """Plan S9: subjects from statement openers and framed noun phrases."""

    def test_activity_statements_name_the_subject(self) -> None:
        cases = {
            "I'm learning Python decorators.": "Python decorators",
            "I'm planning a trip to Lisbon.": "trip to Lisbon",
            "I'm studying photosynthesis because of an exam.": "photosynthesis",
            "I'm looking for a laptop under $1000.": "laptop under $1000",
            "We've been reading about intermittent fasting.": "intermittent fasting",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(extract_turn_subject(text), expected)

    def test_activity_statements_without_a_subject(self) -> None:
        for text in (
            "I'm trying to fix my bike.",
            "I'm wondering whether to buy a heat pump.",
            "I'm still thinking about it.",
            "I'm going to Paris next week.",
            "I'm doing that now.",
            "I'm confused.",
        ):
            with self.subTest(text=text):
                self.assertIsNone(extract_turn_subject(text))

    def test_framed_noun_phrase_is_the_subject(self) -> None:
        cases = {
            "Tell me about metformin.": "metformin",
            "Tell me about photosynthesis": "photosynthesis",
            "Can you tell me about the French Revolution?": "French Revolution",
            # Not a leading frame, a reference, an acceptance or about the speaker.
            "Explain quantum tunnelling please": None,
            "Tell me about it.": None,
            "Tell me more": None,
            "Tell me about how vaccines work": None,
            "Sure, show me.": None,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(extract_turn_subject(text), expected)

    def test_statement_subjects_anchor_follow_ups(self) -> None:
        cases = [
            ([_u("I'm learning Python decorators."), _a("Decorators wrap functions to add behavior.")],
             "Tips for this?", "python decorators"),
            ([_u("I'm planning a trip to Lisbon."), _a("Lisbon is Portugal's coastal capital.")],
             "How about hotels there?", "lisbon"),
            ([_u("Tell me about metformin."), _a("Metformin is a first-line medication for type 2 diabetes.")],
             "What are the side effects?", "metformin"),
        ]
        for history, turn, subject in cases:
            with self.subTest(turn=turn):
                _fu, state, resolved = build_discourse_resolution(turn, history)
                self.assertEqual(state.subject_relation, RELATION_CONTINUATION, state)
                self.assertIn(subject, resolved.web_text.lower())

    def test_statement_subject_does_not_capture_a_topic_switch(self) -> None:
        history = [_u("I'm learning Python decorators."), _a("Decorators wrap functions to add behavior.")]
        for turn in ("How do volcanoes form?", "I'm planning a trip to Rome."):
            with self.subTest(turn=turn):
                _fu, state, resolved = build_discourse_resolution(turn, history)
                self.assertNotEqual(state.subject_relation, RELATION_CONTINUATION, state)
                self.assertNotIn("decorator", resolved.web_text.lower())

    def test_affirmation_after_offer_still_continues(self) -> None:
        history = [
            _u("What is the borrow checker in Rust?"),
            _a("The borrow checker enforces Rust's ownership rules. Would you like an example of a borrow error?"),
        ]
        _fu, state, resolved = build_discourse_resolution("Sure, show me.", history)
        self.assertEqual(state.subject_relation, RELATION_CONTINUATION, state)
        self.assertIn("borrow checker", resolved.web_text.lower())


class SubjectHelperTests(unittest.TestCase):
    def test_extract_turn_subject(self) -> None:
        cases = {
            "What is actor-network theory?>": "actor-network theory",
            "What's new in Python 3.13?": "Python 3.13",
            "How does a heat pump work?": "heat pump",
            "Are statins effective for primary prevention in older adults?": "statins",
            "Is coffee bad for your heart?": "coffee",
            "Does intermittent fasting improve insulin sensitivity?": "intermittent fasting",
            "What is space made of?": "space",
            "What are black holes?": "black holes",
            "Why is that?": None,
            "Is it true?": None,
            "Yes, please.": None,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(extract_turn_subject(text), expected)

    def test_mentions_subject(self) -> None:
        self.assertTrue(mentions_subject("how ANT analyzes AI", "actor-network theory", {"ANT": "Actor-network theory"}))
        self.assertTrue(mentions_subject("the Heat-Pump costs", "heat pump"))
        self.assertTrue(mentions_subject("statin side effects", "statins"))
        self.assertFalse(mentions_subject("ant colonies", "actor-network theory", {"ANT": "Actor-network theory"}))
        self.assertFalse(mentions_subject("pumpkin pie", "heat pump"))

    def test_normalize_subject(self) -> None:
        self.assertEqual(normalize_subject("the boiling point of ethanol"), "boiling point of ethanol")
        self.assertEqual(normalize_subject("new in Python 3.13"), "Python 3.13")
        self.assertIsNone(normalize_subject("3.14"))
        self.assertIsNone(normalize_subject("it"))

    def test_relation_constants(self) -> None:
        self.assertEqual(
            {RELATION_CONTINUATION, RELATION_NEW_SUBJECT, RELATION_NONE, RELATION_NO_PRIOR},
            {"continuation", "new_subject", "none", "no_prior"},
        )


class FollowUpClassifierS6Tests(unittest.TestCase):
    def _classify(self, history, turn):
        full = history + [_u(turn)]
        state = update_discourse_state(full, None, turn)
        return classify_follow_up(turn, full, state)

    def test_affirmation_continuations_reach_threshold(self) -> None:
        for turn in ("Sure, go on", "Yes, please.", "OK, continue"):
            with self.subTest(turn=turn):
                fu = self._classify(_OAUTH, turn)
                self.assertTrue(fu.active, fu)
                self.assertGreaterEqual(fu.confidence, FOLLOW_UP_SUPPRESS_THRESHOLD)

    def test_how_after_affirmation(self) -> None:
        fu = self._classify(_MS, "Yes, tell me how MS is treated")
        self.assertEqual(fu.kind.value, "why_how")
        self.assertIn("why_how_after_affirmation", fu.signals)

    def test_capitalised_facet_is_not_penalised(self) -> None:
        fu = self._classify(_ANT, "Yes, please tell me how ANT analyzes Artificial Inteligence for example.")
        self.assertTrue(fu.active)
        self.assertIn("facet_of_subject", fu.signals)
        self.assertNotIn("explicit_entity_penalty", fu.signals)

    def test_explicit_new_entity_is_still_penalised(self) -> None:
        fu = self._classify(_OAUTH, "Tell me about the Roman Empire")
        self.assertFalse(fu.active)

    def test_topic_change_phrase_stays_none(self) -> None:
        fu = self._classify(_OAUTH, "Let's talk about black holes")
        self.assertEqual(fu.kind.value, "none")


if __name__ == "__main__":
    unittest.main()
