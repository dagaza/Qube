"""Discourse follow-up classification."""
from __future__ import annotations

import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.discourse_intent import (  # noqa: E402
    FOLLOW_UP_SUPPRESS_THRESHOLD,
    FollowUpKind,
    build_referent_salience_suffix,
    classify_follow_up,
)
from core.discourse_query import (  # noqa: E402
    is_deictic_meta_web_request,
    prior_substantive_user_query,
    resolve_retrieval_query,
    resolve_search_target,
    resolve_web_query,
    should_veto_ungrounded_web_follow_up,
)
from core.discourse_patterns import is_deictic_prompt  # noqa: E402
from core.discourse_query_rewrite import resolve_ambiguous_user_query  # noqa: E402
from core.discourse_state import (  # noqa: E402
    DiscourseState,
    extract_assistant_referent,
    is_deictic_topic_phrase,
    promote_referent_after_assistant,
    update_discourse_state,
)


class TestClassifyFollowUp(unittest.TestCase):
    def test_tips_for_this_is_follow_up(self) -> None:
        history = [
            {"role": "user", "content": "Tell me about Slay the Spire"},
            {"role": "assistant", "content": "Slay the Spire is a roguelike deckbuilder."},
            {"role": "user", "content": "Can you give me tips to be successful at this?"},
        ]
        state = DiscourseState(active_topic="Slay the Spire", topic_type="game", confidence=0.8)
        result = classify_follow_up(history[-1]["content"], history, state)
        self.assertIn(result.kind, (FollowUpKind.TIPS_FOR_THIS, FollowUpKind.ANAPHORIC))
        self.assertGreaterEqual(result.confidence, FOLLOW_UP_SUPPRESS_THRESHOLD)

    def test_standalone_question_not_follow_up(self) -> None:
        result = classify_follow_up(
            "What is the capital of France?",
            [{"role": "user", "content": "Hi"}],
            None,
        )
        self.assertEqual(result.kind, FollowUpKind.NONE)
        self.assertLess(result.confidence, FOLLOW_UP_SUPPRESS_THRESHOLD)

    def test_no_history_not_follow_up(self) -> None:
        result = classify_follow_up("tips for this", [], None)
        self.assertEqual(result.kind, FollowUpKind.NONE)

    def test_why_follow_up(self) -> None:
        history = [
            {"role": "user", "content": "Explain quantum tunneling"},
            {"role": "assistant", "content": "Quantum tunneling is ..."},
            {"role": "user", "content": "Why?"},
        ]
        result = classify_follow_up("Why?", history, None)
        self.assertEqual(result.kind, FollowUpKind.WHY_HOW)
        self.assertGreaterEqual(result.confidence, 0.55)


class TestDiscourseState(unittest.TestCase):
    def test_what_is_slays_the_spire(self) -> None:
        history = [
            {"role": "user", "content": "What is Slay the Spire?"},
            {"role": "assistant", "content": "Slay the Spire is a roguelike deckbuilder."},
            {"role": "user", "content": "Can you give me tips for this?"},
        ]
        state = update_discourse_state(history, None, history[-1]["content"])
        self.assertEqual(state.active_topic, "Slay the Spire")
        self.assertEqual(state.topic_type, "game")

    def test_extracts_topic_from_prior_user_turn(self) -> None:
        history = [
            {"role": "user", "content": "What do you know about Slay the Spire?"},
            {"role": "assistant", "content": "It is a deckbuilding roguelike game."},
            {"role": "user", "content": "Any tips for this?"},
        ]
        state = update_discourse_state(history, None, "Any tips for this?")
        self.assertEqual(state.active_topic, "Slay the Spire")
        self.assertEqual(state.topic_type, "game")

    def test_concept_why_question_extracts_subject(self) -> None:
        history = [
            {"role": "user", "content": "Why do birds take dust baths?"},
            {"role": "assistant", "content": "Birds dust-bathe to remove parasites."},
            {"role": "user", "content": "Can you search online for that?"},
        ]
        state = update_discourse_state(history, None, history[-1]["content"])
        self.assertEqual(state.active_topic, "birds take dust baths")
        self.assertEqual(state.topic_type, "concept")

    def test_concept_how_works_question(self) -> None:
        state = update_discourse_state(
            [{"role": "user", "content": "How do refrigerators work?"}],
            None,
            "How do refrigerators work?",
        )
        self.assertEqual(state.active_topic, "refrigerators")
        self.assertEqual(state.topic_type, "concept")

    def test_concept_why_sky_blue(self) -> None:
        state = update_discourse_state(
            [{"role": "user", "content": "Why is the sky blue?"}],
            None,
            "Why is the sky blue?",
        )
        self.assertEqual(state.active_topic, "the sky blue")
        self.assertEqual(state.topic_type, "concept")

    def test_concept_how_recursion(self) -> None:
        state = update_discourse_state(
            [{"role": "user", "content": "How does recursion work?"}],
            None,
            "How does recursion work?",
        )
        self.assertEqual(state.active_topic, "recursion")
        self.assertEqual(state.topic_type, "concept")

    def test_explicit_topic_in_current_turn(self) -> None:
        history = [{"role": "user", "content": "Let's talk about Dark Souls"}]
        state = update_discourse_state(history, None, "Let's talk about Dark Souls")
        self.assertIn("Dark Souls", state.active_topic or "")

    def test_nepal_kathmandu_population_follow_up(self) -> None:
        prior = update_discourse_state(
            [
                {"role": "user", "content": "What is the capital of Nepal?"},
                {"role": "assistant", "content": "Kathmandu."},
            ],
            None,
            "What is the capital of Nepal?",
        )
        history = [
            {"role": "user", "content": "What is the capital of Nepal?"},
            {"role": "assistant", "content": "Kathmandu."},
            {"role": "user", "content": "What is the population of this city?"},
        ]
        state = update_discourse_state(
            history,
            prior,
            "What is the population of this city?",
        )
        self.assertEqual(state.active_referent, "Kathmandu")
        self.assertEqual(state.referent_type, "city")
        self.assertNotEqual(state.active_topic, "the population of this city")

    def test_deictic_what_is_does_not_set_topic(self) -> None:
        state = update_discourse_state(
            [],
            None,
            "What is the population of this city?",
        )
        self.assertNotEqual(state.active_topic, "the population of this city")
        self.assertTrue(is_deictic_topic_phrase("the population of this city"))

    def test_single_word_assistant_entity(self) -> None:
        self.assertEqual(extract_assistant_referent("Kathmandu."), "Kathmandu")


class TestReferentSalience(unittest.TestCase):
    def test_referent_salience_suffix_wording(self) -> None:
        suffix = build_referent_salience_suffix("Kathmandu", referent_type="city")
        self.assertIn("Primary referent: Kathmandu (city)", suffix)
        self.assertNotIn("Active conversation topic", suffix)
        self.assertIn("Resolve follow-up references", suffix)

    def test_nepal_follow_up_prompt_blocks_use_referent_not_deictic_topic(self) -> None:
        from core.prompt_blocks import build_prompt_blocks
        from core.prompt_renderers import render_system_ok_messages

        prior = update_discourse_state(
            [
                {"role": "user", "content": "What is the capital of Nepal?"},
                {"role": "assistant", "content": "Kathmandu."},
            ],
            None,
            "What is the capital of Nepal?",
        )
        history = [
            {"role": "user", "content": "What is the capital of Nepal?"},
            {"role": "assistant", "content": "Kathmandu."},
            {
                "role": "user",
                "content": (
                    "[Referring to Kathmandu]\n\n"
                    "What is the population of this city?"
                ),
            },
        ]
        state = update_discourse_state(
            history,
            prior,
            "What is the population of this city?",
        )
        salience = build_referent_salience_suffix(
            state.active_referent or "",
            referent_type=state.referent_type,
        )
        blocks = build_prompt_blocks(
            execution_route="NONE",
            explicit_remember_active=False,
            has_retrieval_sources=False,
            conversation_history=history,
            topic_salience_hint=salience,
            follow_up_active=True,
        )
        messages = render_system_ok_messages(blocks)
        system = messages[0]["content"]
        self.assertIn("Primary referent: Kathmandu (city)", system)
        self.assertNotIn("Active conversation topic: the population of this city", system)
        self.assertIn("[Referring to Kathmandu]", messages[-1]["content"])


class TestQueryExpansion(unittest.TestCase):
    def test_expands_with_active_topic(self) -> None:
        from core.discourse_intent import FollowUpClassification

        follow_up = FollowUpClassification(FollowUpKind.TIPS_FOR_THIS, 0.72)
        state = DiscourseState(active_topic="Slay the Spire", topic_type="game", confidence=0.8)
        expanded = resolve_retrieval_query(
            "tips to be successful at this",
            follow_up,
            state,
        )
        self.assertIn("Slay the Spire", expanded)
        self.assertIn("tips to be successful at this", expanded)

    def test_no_expansion_without_topic(self) -> None:
        from core.discourse_intent import FollowUpClassification

        follow_up = FollowUpClassification(FollowUpKind.ANAPHORIC, 0.65)
        expanded = resolve_retrieval_query("why?", follow_up, DiscourseState())
        self.assertEqual(expanded, "why?")

    def test_web_query_matches_retrieval_expansion(self) -> None:
        from core.discourse_intent import FollowUpClassification

        follow_up = FollowUpClassification(FollowUpKind.TIPS_FOR_THIS, 0.72)
        state = DiscourseState(active_topic="Slay the Spire", topic_type="game", confidence=0.8)
        web_q = resolve_web_query("tips for this", follow_up, state)
        self.assertIn("Slay the Spire", web_q)

    def test_veto_web_only_when_ungrounded(self) -> None:
        from core.discourse_intent import FollowUpClassification

        fu = FollowUpClassification(FollowUpKind.TIPS_FOR_THIS, 0.72)
        self.assertFalse(
            should_veto_ungrounded_web_follow_up(
                fu, DiscourseState(active_topic="Slay the Spire")
            )
        )
        self.assertTrue(should_veto_ungrounded_web_follow_up(fu, DiscourseState()))

    def test_resolve_search_target_uses_referent(self) -> None:
        from core.discourse_intent import FollowUpClassification

        follow_up = FollowUpClassification(FollowUpKind.ANAPHORIC, 0.72)
        state = DiscourseState(
            active_referent="Kathmandu",
            referent_type="city",
            active_topic="the capital of Nepal",
            confidence=0.8,
        )
        target = resolve_search_target(
            "What is the population of this city?",
            follow_up,
            state,
        )
        self.assertEqual(target.rewrite_reason, "referent_expansion")
        self.assertIn("Kathmandu", target.query)
        self.assertFalse(target.query.startswith("Regarding the population of this city"))


class TestMetaWebQueryRewrite(unittest.TestCase):
    def test_deictic_online_search_for_the_answer(self) -> None:
        prompt = (
            "Yes that would be nice. Can you also do an online search "
            "for the answer?"
        )
        self.assertTrue(is_deictic_meta_web_request(prompt))

    def test_substantive_online_request_not_meta(self) -> None:
        self.assertFalse(is_deictic_meta_web_request("Look online for a recipe."))
        self.assertFalse(
            is_deictic_meta_web_request("Can you search online for a joke?")
        )
        self.assertFalse(
            is_deictic_meta_web_request("Find reviews online for this phone.")
        )

    def test_rewrites_meta_web_request_to_prior_user_turn(self) -> None:
        from core.discourse_intent import FollowUpClassification

        history = [
            {"role": "user", "content": "Why do birds take dust baths?"},
            {
                "role": "assistant",
                "content": "Birds take dust baths to remove parasites.",
            },
            {
                "role": "user",
                "content": (
                    "Yes that would be nice. Can you also do an online search "
                    "for the answer?"
                ),
            },
        ]
        follow_up = classify_follow_up(history[-1]["content"], history, None)
        state = update_discourse_state(history, None, history[-1]["content"])
        target = resolve_search_target(
            history[-1]["content"],
            follow_up,
            state,
            history,
        )
        self.assertEqual(target.query, "Why do birds take dust baths?")
        self.assertEqual(target.rewrite_reason, "meta_prior_turn")

    def test_topic_expansion_beats_meta_rewrite(self) -> None:
        from core.discourse_intent import FollowUpClassification

        follow_up = FollowUpClassification(FollowUpKind.TIPS_FOR_THIS, 0.72)
        state = DiscourseState(active_topic="Slay the Spire", topic_type="game", confidence=0.8)
        target = resolve_search_target(
            "Search the web for deckbuilding tips",
            follow_up,
            state,
            [
                {"role": "user", "content": "What is Slay the Spire?"},
                {"role": "assistant", "content": "A roguelike deckbuilder."},
            ],
        )
        self.assertEqual(target.rewrite_reason, "topic_expansion")
        self.assertIn("Slay the Spire", target.query)
        self.assertIn("deckbuilding", target.query.lower())

    def test_prior_substantive_user_query_skips_meta_chain(self) -> None:
        history = [
            {"role": "user", "content": "Why do birds take dust baths?"},
            {"role": "assistant", "content": "Birds dust-bathe to stay clean."},
            {"role": "user", "content": "Can you search online for the answer?"},
            {
                "role": "user",
                "content": "Please do an online search for the answer too.",
            },
        ]
        prior = prior_substantive_user_query(history, history[-1]["content"])
        self.assertEqual(prior, "Why do birds take dust baths?")


class TestDiscoursePhase15(unittest.TestCase):
    def test_its_is_deictic_prompt(self) -> None:
        self.assertTrue(is_deictic_prompt("And what is the size of its population?"))

    def test_what_is_its_topic_not_stored_as_explicit(self) -> None:
        state = update_discourse_state(
            [],
            None,
            "And what is the size of its population?",
        )
        self.assertNotEqual(state.active_topic, "the size of its population")

    def test_promote_after_assistant_capital_of_sentence(self) -> None:
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
        self.assertEqual(promoted.referent_type, "city")
        self.assertEqual(promoted.referent_source, "assistant_pattern")
        self.assertGreaterEqual(promoted.referent_confidence, 0.85)

    def test_its_population_follow_up_resolves_kathmandu(self) -> None:
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
        history = [
            {"role": "user", "content": "What is the capital of Nepal?"},
            {"role": "assistant", "content": "Kathmandu is the capital of Nepal."},
            {"role": "user", "content": "And what is the size of its population?"},
        ]
        state = update_discourse_state(
            history,
            promoted,
            "And what is the size of its population?",
        )
        self.assertEqual(state.active_referent, "Kathmandu")
        self.assertNotEqual(state.active_topic, "the size of its population")
        follow_up = classify_follow_up(
            "And what is the size of its population?",
            history,
            state,
        )
        self.assertTrue(follow_up.active)
        resolved = resolve_ambiguous_user_query(
            "And what is the size of its population?",
            state,
            follow_up,
        )
        self.assertTrue(resolved.succeeded)
        self.assertIn("Kathmandu", resolved.resolved)


if __name__ == "__main__":
    unittest.main()
