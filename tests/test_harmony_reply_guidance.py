"""Harmony system guidance merged into rendered prompts."""
from __future__ import annotations

import unittest

from core.harmony_reply_guidance import (
    HARMONY_BRIEF_REPLY_GUIDANCE,
    HARMONY_ENUMERATION_REPLY_GUIDANCE,
    HARMONY_FINAL_REPLY_GUIDANCE,
    HARMONY_MIXED_REPLY_GUIDANCE,
    merge_harmony_system_content,
)
from core.harmony_renderer import render_harmony_final_prompt
from core.reply_shape_policy import resolve_reply_shape_policy


class TestHarmonyReplyGuidance(unittest.TestCase):
    def test_single_turn_includes_guidance_without_user_system(self) -> None:
        prompt = render_harmony_final_prompt(
            [{"role": "user", "content": "Why do birds bathe?"}]
        )
        self.assertIn(HARMONY_FINAL_REPLY_GUIDANCE, prompt)
        self.assertIn("2–4 short sections", prompt)

    def test_merges_with_existing_system(self) -> None:
        prompt = render_harmony_final_prompt(
            [
                {"role": "system", "content": "You are Qube."},
                {"role": "user", "content": "Why do birds bathe?"},
            ]
        )
        self.assertIn("You are Qube.", prompt)
        self.assertIn("Cleaning, Temperature", prompt)
        self.assertEqual(prompt.count(HARMONY_FINAL_REPLY_GUIDANCE), 1)

    def test_structured_task_omits_guidance(self) -> None:
        prompt = render_harmony_final_prompt(
            [
                {"role": "system", "content": "Return JSON ONLY."},
                {"role": "user", "content": "Conversation:\nuser: hi"},
            ],
            include_reply_guidance=False,
        )
        self.assertIn("Return JSON ONLY.", prompt)
        self.assertNotIn(HARMONY_FINAL_REPLY_GUIDANCE, prompt)
        self.assertNotIn("2–4 short sections", prompt)

    def test_brief_mode_guidance_text(self) -> None:
        merged = merge_harmony_system_content(
            ["You are Qube."],
            chat_format_mode="brief",
        )
        self.assertIn(HARMONY_BRIEF_REPLY_GUIDANCE, merged)
        self.assertNotIn(HARMONY_FINAL_REPLY_GUIDANCE, merged)

    def test_mixed_mode_guidance_text(self) -> None:
        merged = merge_harmony_system_content(
            ["You are Qube."],
            chat_format_mode="mixed",
        )
        self.assertIn(HARMONY_MIXED_REPLY_GUIDANCE, merged)
        self.assertNotIn("2–4 short sections", merged)

    def test_enumeration_policy_uses_list_guidance(self) -> None:
        policy = resolve_reply_shape_policy(
            execution_route="NONE",
            user_query="List the major ethnic groups in Nepal",
        )
        merged = merge_harmony_system_content(
            ["You are Qube."],
            reply_shape_policy=policy,
        )
        self.assertIn(HARMONY_ENUMERATION_REPLY_GUIDANCE, merged)
        self.assertNotIn(HARMONY_BRIEF_REPLY_GUIDANCE, merged)
