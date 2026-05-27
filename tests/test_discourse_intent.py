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
    classify_follow_up,
)
from core.discourse_query import (  # noqa: E402
    resolve_retrieval_query,
    resolve_web_query,
    should_veto_ungrounded_web_follow_up,
)
from core.discourse_state import DiscourseState, update_discourse_state  # noqa: E402


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

    def test_explicit_topic_in_current_turn(self) -> None:
        history = [{"role": "user", "content": "Let's talk about Dark Souls"}]
        state = update_discourse_state(history, None, "Let's talk about Dark Souls")
        self.assertIn("Dark Souls", state.active_topic or "")


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


if __name__ == "__main__":
    unittest.main()
