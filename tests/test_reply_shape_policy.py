"""Reply-shape policy: intent-aware formatting and instruction conflict resolution."""
from __future__ import annotations

import unittest

from core.discourse_intent import FollowUpClassification, FollowUpKind
from core.reply_shape_policy import (
    detect_enumeration_intent,
    resolve_reply_shape_policy,
)


class TestEnumerationIntent(unittest.TestCase):
    def test_list_query_detected(self) -> None:
        self.assertTrue(detect_enumeration_intent("List the major ethnic groups in Nepal"))

    def test_factual_lookup_not_enumeration(self) -> None:
        self.assertFalse(detect_enumeration_intent("What is the capital of France?"))


class TestReplyShapePolicy(unittest.TestCase):
    def test_brief_lookup_stays_brief(self) -> None:
        policy = resolve_reply_shape_policy(
            execution_route="NONE",
            user_query="What is the capital of France?",
        )
        self.assertEqual(policy.chat_format_mode, "brief")
        self.assertEqual(policy.format_intent, "brief")

    def test_enumeration_upgrades_brief_conflict(self) -> None:
        policy = resolve_reply_shape_policy(
            execution_route="NONE",
            user_query="What are the major ethnic groups in Nepal?",
        )
        self.assertEqual(policy.chat_format_mode, "structured")
        self.assertEqual(policy.format_intent, "enumeration")
        self.assertTrue(policy.require_list_format)
        self.assertIn("brief_vs_enumeration", policy.instruction_conflicts)

    def test_compare_follow_up_structured(self) -> None:
        fu = FollowUpClassification(FollowUpKind.COMPARE, 0.7, ("compare",))
        policy = resolve_reply_shape_policy(
            execution_route="NONE",
            user_query="Compare Hinduism and Buddhism in Nepal",
            follow_up=fu,
        )
        self.assertEqual(policy.format_intent, "structured")

    def test_structured_intent_includes_markdown_heading_guidance(self) -> None:
        policy = resolve_reply_shape_policy(
            execution_route="NONE",
            user_query="Explain how photosynthesis works.",
        )
        self.assertEqual(policy.format_intent, "structured")
        hint = policy.system_reply_hint.lower()
        self.assertIn("substantial multi-section", hint)
        self.assertIn("##", policy.system_reply_hint)
        self.assertIn("bold-only", hint)
        self.assertIn("skip headings", hint)

    def test_brief_lookup_omits_structured_heading_guidance(self) -> None:
        policy = resolve_reply_shape_policy(
            execution_route="NONE",
            user_query="What is the capital of France?",
        )
        self.assertEqual(policy.format_intent, "brief")
        self.assertNotIn("substantial multi-section", policy.system_reply_hint.lower())
        self.assertNotIn("bold-only", policy.system_reply_hint.lower())
        self.assertNotIn("##", policy.system_reply_hint)

    def test_anaphoric_follow_up_mixed_or_follow_up(self) -> None:
        fu = FollowUpClassification(FollowUpKind.ANAPHORIC, 0.7, ("anaphoric",))
        policy = resolve_reply_shape_policy(
            execution_route="NONE",
            user_query="And its religious minorities?",
            follow_up=fu,
        )
        self.assertIn(policy.format_intent, ("follow_up", "mixed", "brief"))
        self.assertIn("follow-up", policy.system_reply_hint.lower())

    def test_prior_turn_unreliable_adds_uncertainty_hint(self) -> None:
        policy = resolve_reply_shape_policy(
            execution_route="NONE",
            user_query="What is the capital of France?",
            prior_turn_unreliable=True,
        )
        self.assertIn("confident", policy.system_reply_hint.lower())


if __name__ == "__main__":
    unittest.main()
