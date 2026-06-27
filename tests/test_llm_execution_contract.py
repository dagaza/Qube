"""Primary-engine task contract boundary (Phase 1)."""
from __future__ import annotations

import unittest

from core.harmony_reply_guidance import HARMONY_FINAL_REPLY_GUIDANCE
from core.llm_execution_contract import (
    PrimaryEngineTask,
    check_task_prompt_policy,
    normalize_messages_for_task,
    policy_for_task,
)


class TestPolicyForTask(unittest.TestCase):
    def test_chat_skips_harmony_layers_by_default(self) -> None:
        p = policy_for_task(PrimaryEngineTask.chat)
        self.assertFalse(p.include_harmony_reply_guidance)
        self.assertFalse(p.include_harmony_phrase_stops)
        self.assertFalse(p.require_role_separated_messages)

    def test_chat_allows_harmony_layers_when_model_active(self) -> None:
        p = policy_for_task(PrimaryEngineTask.chat, harmony_model_active=True)
        self.assertTrue(p.include_harmony_reply_guidance)
        self.assertTrue(p.include_harmony_phrase_stops)
        self.assertFalse(p.require_role_separated_messages)

    def test_memory_extraction_forbids_harmony_layers(self) -> None:
        p = policy_for_task(PrimaryEngineTask.memory_extraction)
        self.assertFalse(p.include_harmony_reply_guidance)
        self.assertFalse(p.include_harmony_phrase_stops)
        self.assertTrue(p.require_role_separated_messages)

    def test_deep_research_synthesis_matches_extraction_policy(self) -> None:
        p = policy_for_task(PrimaryEngineTask.deep_research_synthesis)
        self.assertFalse(p.include_harmony_reply_guidance)
        self.assertTrue(p.require_role_separated_messages)


class TestNormalizeMessagesForTask(unittest.TestCase):
    def test_chat_accepts_multi_turn(self) -> None:
        msgs = normalize_messages_for_task(
            [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hi"},
            ],
            PrimaryEngineTask.chat,
        )
        self.assertEqual(len(msgs), 2)

    def test_extraction_requires_system_user_pair(self) -> None:
        with self.assertRaises(ValueError):
            normalize_messages_for_task(
                [{"role": "user", "content": "only user"}],
                PrimaryEngineTask.memory_extraction,
            )

    def test_extraction_accepts_system_user(self) -> None:
        msgs = normalize_messages_for_task(
            [
                {"role": "system", "content": "Return JSON."},
                {"role": "user", "content": "Conversation:\nuser: hi"},
            ],
            PrimaryEngineTask.memory_extraction,
        )
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])


class TestCheckTaskPromptPolicy(unittest.TestCase):
    def test_flags_harmony_guidance_on_extraction(self) -> None:
        flags = check_task_prompt_policy(
            task=PrimaryEngineTask.memory_extraction,
            rendered_prompt=f"<|start|>system<|message|>{HARMONY_FINAL_REPLY_GUIDANCE}<|end|>",
        )
        self.assertIn("forbidden_harmony_chat_guidance_present", flags)

    def test_chat_does_not_flag_guidance_when_harmony_active(self) -> None:
        flags = check_task_prompt_policy(
            task=PrimaryEngineTask.chat,
            rendered_prompt=f"system {HARMONY_FINAL_REPLY_GUIDANCE}",
            policy=policy_for_task(PrimaryEngineTask.chat, harmony_model_active=True),
        )
        self.assertNotIn("forbidden_harmony_chat_guidance_present", flags)


if __name__ == "__main__":
    unittest.main()
