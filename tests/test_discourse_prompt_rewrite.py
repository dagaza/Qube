"""Validated follow-up prompt grounding anchors."""
from __future__ import annotations

import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.discourse_intent import FollowUpClassification, FollowUpKind, classify_follow_up  # noqa: E402
from core.discourse_prompt_rewrite import (  # noqa: E402
    resolve_discourse_prompt_rewrite,
    score_rewrite_anchor,
    validate_stored_discourse_topic,
)
from core.discourse_query import resolve_search_target  # noqa: E402
from core.discourse_query_rewrite import resolve_ambiguous_user_query  # noqa: E402
from core.discourse_state import (  # noqa: E402
    DiscourseState,
    promote_referent_after_assistant,
    update_discourse_state,
)


class TestAnchorScoring(unittest.TestCase):
    def test_rejects_pure_number(self) -> None:
        score = score_rewrite_anchor("18", user_message="What about its elevation?")
        self.assertFalse(score.usable)
        self.assertEqual(score.reject_reason, "pure_number")

    def test_rejects_measurement(self) -> None:
        score = score_rewrite_anchor(
            "1,400 metres (≈4,600 feet) above sea level",
            user_message="How high is it?",
        )
        self.assertFalse(score.usable)
        self.assertEqual(score.reject_reason, "measurement")

    def test_rejects_short_noun_phrase(self) -> None:
        score = score_rewrite_anchor("square miles", user_message="What is its area?")
        self.assertFalse(score.usable)
        self.assertEqual(score.reject_reason, "short_noun_phrase")

    def test_accepts_named_entity(self) -> None:
        score = score_rewrite_anchor("Kathmandu", user_message="What is its population?")
        self.assertTrue(score.usable)
        self.assertEqual(score.accept_reason, "named_entity")
        self.assertGreaterEqual(score.confidence, 0.85)

    def test_accepts_relevant_noun_phrase(self) -> None:
        score = score_rewrite_anchor(
            "the capital of Nepal",
            user_message="What is the population of the capital of Nepal?",
        )
        self.assertTrue(score.usable)
        self.assertEqual(score.accept_reason, "relevant_noun_phrase")

    def test_rejects_irrelevant_noun_phrase(self) -> None:
        score = score_rewrite_anchor(
            "the capital of Nepal",
            user_message="What is its population?",
        )
        self.assertFalse(score.usable)
        self.assertEqual(score.reject_reason, "low_relevance")

    def test_validate_stored_topic_rejects_measurements(self) -> None:
        self.assertFalse(validate_stored_discourse_topic("18"))
        self.assertFalse(validate_stored_discourse_topic("48 square kilometres"))
        self.assertTrue(validate_stored_discourse_topic("Kathmandu"))
        self.assertTrue(validate_stored_discourse_topic("the capital of Nepal"))


class TestPromptRewrite(unittest.TestCase):
    def _kathmandu_state(self) -> DiscourseState:
        prior = update_discourse_state(
            [{"role": "user", "content": "What is the capital of Nepal?"}],
            None,
            "What is the capital of Nepal?",
        )
        return promote_referent_after_assistant(
            user_prompt="What is the capital of Nepal?",
            assistant_text="Kathmandu is the capital of Nepal.",
            prior=prior,
        )

    def test_referent_anchor_for_deictic_follow_up(self) -> None:
        state = self._kathmandu_state()
        prompt = "What is its population?"
        follow_up = classify_follow_up(prompt, [], state)
        result = resolve_discourse_prompt_rewrite(
            user_message=prompt,
            resolved_query=resolve_ambiguous_user_query(prompt, state, follow_up),
            follow_up=follow_up,
            discourse=state,
        )
        self.assertTrue(result.applied)
        self.assertIn("Kathmandu", result.grounded)
        self.assertEqual(result.rewrite_anchor, "Kathmandu")
        self.assertTrue(result.rewrite_reason.startswith("query_"))

    def test_no_rewrite_for_bad_topic_anchor(self) -> None:
        state = DiscourseState(
            active_topic="18",
            topic_type="unknown",
            confidence=0.55,
        )
        prompt = "What about its elevation?"
        follow_up = FollowUpClassification(FollowUpKind.ANAPHORIC, 0.85)
        result = resolve_discourse_prompt_rewrite(
            user_message=prompt,
            resolved_query=None,
            follow_up=follow_up,
            discourse=state,
        )
        self.assertFalse(result.applied)
        self.assertEqual(result.grounded, prompt)
        self.assertIn("anchor_rejected", result.rewrite_reason)

    def test_no_generic_fallback_prefix(self) -> None:
        state = DiscourseState(active_topic="18", confidence=0.5)
        prompt = "Tell me more"
        follow_up = FollowUpClassification(FollowUpKind.ELLIPSIS, 0.8)
        result = resolve_discourse_prompt_rewrite(
            user_message=prompt,
            resolved_query=None,
            follow_up=follow_up,
            discourse=state,
        )
        self.assertFalse(result.applied)
        self.assertNotIn("Continuing from the conversation above", result.grounded)

    def test_trace_fields_present(self) -> None:
        state = self._kathmandu_state()
        prompt = "What is its area?"
        follow_up = classify_follow_up(prompt, [], state)
        resolved = resolve_ambiguous_user_query(prompt, state, follow_up)
        result = resolve_discourse_prompt_rewrite(
            user_message=prompt,
            resolved_query=resolved,
            follow_up=follow_up,
            discourse=state,
        )
        fields = result.trace_fields()
        self.assertIn("rewrite_anchor", fields)
        self.assertIn("rewrite_confidence", fields)
        self.assertIn("rewrite_reason", fields)


class TestSearchTargetAnchorValidation(unittest.TestCase):
    def test_search_target_skips_measurement_topic(self) -> None:
        state = DiscourseState(active_topic="1,400 metres", confidence=0.55)
        follow_up = FollowUpClassification(FollowUpKind.ANAPHORIC, 0.9)
        result = resolve_search_target(
            "How high is it?",
            follow_up,
            state,
            None,
        )
        self.assertEqual(result.rewrite_reason, "none")
        self.assertEqual(result.query, "How high is it?")

    def test_search_target_expands_with_valid_referent(self) -> None:
        state = DiscourseState(
            active_referent="Kathmandu",
            referent_type="city",
            confidence=0.9,
        )
        follow_up = FollowUpClassification(FollowUpKind.ANAPHORIC, 0.9)
        result = resolve_search_target(
            "population",
            follow_up,
            state,
            None,
        )
        self.assertTrue(result.rewritten)
        self.assertIn("Kathmandu", result.query)


class TestConversationHealthRewriteGate(unittest.TestCase):
    def test_rewrite_disabled_when_health_recovery(self) -> None:
        state = DiscourseState(active_referent="Nepal", referent_type="country", confidence=0.9)
        follow_up = FollowUpClassification(FollowUpKind.ANAPHORIC, 0.8)
        result = resolve_discourse_prompt_rewrite(
            user_message="And its religious minorities?",
            resolved_query=None,
            follow_up=follow_up,
            discourse=state,
            allow_rewrite=False,
        )
        self.assertFalse(result.applied)
        self.assertEqual(result.rewrite_reason, "conversation_health")


if __name__ == "__main__":
    unittest.main()
