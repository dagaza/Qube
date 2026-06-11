"""Referent stability policy and Kathmandu regression tests."""
from __future__ import annotations

import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.discourse_intent import (  # noqa: E402
    FollowUpKind,
    build_entity_aspect_grounding_suffix,
    classify_follow_up,
)
from core.discourse_prompt_rewrite import select_salience_anchor  # noqa: E402
from core.discourse_query_rewrite import resolve_ambiguous_user_query  # noqa: E402
from core.discourse_referent_policy import (  # noqa: E402
    extract_entity_and_aspect,
    should_replace_referent,
    validate_referent_candidate,
    validate_resolved_query,
)
from core.discourse_state import (  # noqa: E402
    DiscourseState,
    extract_assistant_referent,
    promote_referent_after_assistant,
    update_discourse_state,
)


FLORA_USER = "What about Kathmandu's flora and fauna?"
FLORA_ASST = (
    "Kathmandu is dominated by urban vegetation, featuring species like Peepal (Ficus religiosa), "
    "Banyan trees, and various ornamental flowers such as Jasmine and Marigold in public parks "
    "and temple courtyards. The city supports diverse fauna including pigeons, mynas, sparrows, "
    "and occasionally squirrels or jackals within green belts, though large-scale wildlife is "
    "primarily found in the nearby Chitwan National Park rather than within the city limits itself."
)
MUSIC_USER = "Ok, how about its music and arts scene?"


class TestEntityAspectParse(unittest.TestCase):
    def test_possessive_about_phrase(self) -> None:
        parsed = extract_entity_and_aspect(FLORA_USER)
        self.assertEqual(parsed.entity, "Kathmandu")
        self.assertEqual(parsed.aspect, "flora and fauna")


class TestReferentValidation(unittest.TestCase):
    def test_rejects_list_fragment(self) -> None:
        usable, reason = validate_referent_candidate(
            "Jasmine and Marigold in public",
            source="assistant_answer",
            user_prompt=FLORA_USER,
            assistant_text=FLORA_ASST,
        )
        self.assertFalse(usable)
        self.assertIn(reason, ("not_in_user_text", "enumeration_fragment"))

    def test_accepts_kathmandu_from_assistant_when_in_user_prompt(self) -> None:
        usable, _ = validate_referent_candidate(
            "Kathmandu",
            source="assistant_answer",
            user_prompt=FLORA_USER,
            assistant_text=FLORA_ASST,
        )
        self.assertTrue(usable)


class TestStickyPromotion(unittest.TestCase):
    def test_assistant_list_does_not_replace_user_entity(self) -> None:
        prior = update_discourse_state(
            [{"role": "user", "content": FLORA_USER}],
            None,
            FLORA_USER,
        )
        self.assertEqual(prior.active_referent, "Kathmandu")
        promoted = promote_referent_after_assistant(
            user_prompt=FLORA_USER,
            assistant_text=FLORA_ASST,
            prior=prior,
        )
        self.assertEqual(promoted.active_referent, "Kathmandu")
        self.assertNotIn("Jasmine", promoted.active_referent or "")

    def test_capital_of_pattern_still_promotes(self) -> None:
        prior = update_discourse_state(
            [{"role": "user", "content": "What is the capital of Nepal?"}],
            None,
            "What is the capital of Nepal?",
        )
        promoted = promote_referent_after_assistant(
            user_prompt="What is the capital of Nepal?",
            assistant_text="Kathmandu is the capital of Nepal.",
            prior=prior,
        )
        self.assertEqual(promoted.active_referent, "Kathmandu")
        self.assertEqual(promoted.referent_source, "assistant_pattern")


class TestKathmanduFloraToMusicRegression(unittest.TestCase):
    def _music_chain_state(self) -> DiscourseState:
        prior = update_discourse_state(
            [
                {"role": "user", "content": "What is the capital of Nepal?"},
                {"role": "assistant", "content": "Kathmandu is the capital of Nepal."},
            ],
            None,
            "What is the capital of Nepal?",
        )
        prior = promote_referent_after_assistant(
            user_prompt="What is the capital of Nepal?",
            assistant_text="Kathmandu is the capital of Nepal.",
            prior=prior,
        )
        prior = update_discourse_state(
            [
                {"role": "user", "content": "What is the capital of Nepal?"},
                {"role": "assistant", "content": "Kathmandu is the capital of Nepal."},
                {"role": "user", "content": FLORA_USER},
            ],
            prior,
            FLORA_USER,
        )
        prior = promote_referent_after_assistant(
            user_prompt=FLORA_USER,
            assistant_text=FLORA_ASST,
            prior=prior,
        )
        history = [
            {"role": "user", "content": "What is the capital of Nepal?"},
            {"role": "assistant", "content": "Kathmandu is the capital of Nepal."},
            {"role": "user", "content": FLORA_USER},
            {"role": "assistant", "content": FLORA_ASST},
            {"role": "user", "content": MUSIC_USER},
        ]
        return update_discourse_state(history, prior, MUSIC_USER)

    def test_music_follow_up_keeps_kathmandu_referent(self) -> None:
        state = self._music_chain_state()
        self.assertEqual(state.active_referent, "Kathmandu")
        self.assertIn("music", (state.active_aspect or "").lower())

    def test_music_follow_up_rewrites_to_kathmandu(self) -> None:
        state = self._music_chain_state()
        history = [
            {"role": "user", "content": FLORA_USER},
            {"role": "assistant", "content": FLORA_ASST},
            {"role": "user", "content": MUSIC_USER},
        ]
        follow_up = classify_follow_up(MUSIC_USER, history, state)
        self.assertEqual(follow_up.kind, FollowUpKind.ANAPHORIC)
        resolved = resolve_ambiguous_user_query(MUSIC_USER, state, follow_up)
        self.assertTrue(resolved.succeeded)
        self.assertIn("Kathmandu", resolved.resolved)
        self.assertNotIn("Jasmine", resolved.resolved)

    def test_salience_anchor_persists_after_rewrite(self) -> None:
        state = self._music_chain_state()
        follow_up = classify_follow_up(MUSIC_USER, [], state)
        resolved = resolve_ambiguous_user_query(MUSIC_USER, state, follow_up)
        anchor, _, reason = select_salience_anchor(
            discourse=state,
            user_message=MUSIC_USER,
            resolved_query=resolved,
        )
        self.assertEqual(anchor, "Kathmandu")
        self.assertTrue(reason.startswith("referent_salience"))

    def test_grounding_suffix_mentions_entity_and_aspect(self) -> None:
        suffix = build_entity_aspect_grounding_suffix(
            "Kathmandu",
            aspect="music and arts scene",
            entity_type="city",
        )
        self.assertIn("Kathmandu", suffix)
        self.assertIn("music and arts scene", suffix)
        self.assertIn("previous answer", suffix.lower())


class TestExtractionHardening(unittest.TestCase):
    def test_assistant_referent_prefers_subject_not_list(self) -> None:
        ref = extract_assistant_referent(FLORA_ASST)
        self.assertEqual(ref, "Kathmandu")


class TestShouldReplaceReferent(unittest.TestCase):
    def test_sticky_user_blocks_assistant_answer(self) -> None:
        prior = DiscourseState(
            active_referent="Kathmandu",
            referent_type="city",
            referent_source="user_question",
            referent_confidence=0.85,
        )
        allow, reason = should_replace_referent(
            prior,
            "Jasmine and Marigold in public",
            "assistant_answer",
            0.80,
        )
        self.assertFalse(allow)
        self.assertEqual(reason, "sticky_user_referent")


class TestResolvedQueryValidation(unittest.TestCase):
    def test_rejects_jasmine_rewrite(self) -> None:
        state = DiscourseState(
            active_referent="Kathmandu",
            referent_type="city",
            referent_source="user_question",
            referent_confidence=0.85,
        )
        ok, reason = validate_resolved_query(
            "Ok, how about Jasmine and Marigold in public's music and arts scene?",
            state,
        )
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
