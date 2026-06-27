"""Settings-gated CHAT_PERSONALITY_SUFFIX on plain NONE turns."""
from __future__ import annotations

import unittest

from core.memory_filters import (
    CHAT_PERSONALITY_SUFFIX,
    FILE_SEARCH_SYSTEM_SUFFIX,
    NARRATIVE_RECALL_SYSTEM_SUFFIX,
    NO_SOURCES_SYSTEM_SUFFIX,
    RECALL_FUSION_SYSTEM_SUFFIX,
)
from core.prompt_blocks import build_prompt_blocks, compose_system_prompt


class TestChatPersonalitySuffix(unittest.TestCase):
    def test_present_none_toggle_on_no_retrieval(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="NONE",
            explicit_remember_active=False,
            has_retrieval_sources=False,
            chat_personality_enabled=True,
            engine_mode="internal",
        )
        sys_p = compose_system_prompt(blocks)
        self.assertIn(CHAT_PERSONALITY_SUFFIX, sys_p)

    def test_absent_when_toggle_off(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="NONE",
            explicit_remember_active=False,
            chat_personality_enabled=False,
        )
        self.assertNotIn(CHAT_PERSONALITY_SUFFIX, compose_system_prompt(blocks))

    def test_absent_with_retrieval_sources(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="NONE",
            explicit_remember_active=False,
            has_retrieval_sources=True,
            chat_personality_enabled=True,
        )
        self.assertNotIn(CHAT_PERSONALITY_SUFFIX, compose_system_prompt(blocks))

    def test_absent_rag_hybrid_memory_with_sources(self) -> None:
        for route in ("RAG", "HYBRID", "MEMORY"):
            blocks = build_prompt_blocks(
                execution_route=route,
                explicit_remember_active=False,
                has_retrieval_sources=True,
                chat_personality_enabled=True,
            )
            sys_p = compose_system_prompt(blocks)
            self.assertNotIn(CHAT_PERSONALITY_SUFFIX, sys_p)
            self.assertIn(RECALL_FUSION_SYSTEM_SUFFIX, sys_p)

    def test_absent_web_route(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="WEB",
            explicit_remember_active=False,
            chat_personality_enabled=True,
        )
        self.assertNotIn(CHAT_PERSONALITY_SUFFIX, compose_system_prompt(blocks))

    def test_absent_explicit_remember(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="NONE",
            explicit_remember_active=True,
            explicit_remember_body="My cat is Luna",
            chat_personality_enabled=True,
        )
        self.assertNotIn(CHAT_PERSONALITY_SUFFIX, compose_system_prompt(blocks))

    def test_absent_file_search_and_narrative(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="RAG",
            explicit_remember_active=False,
            has_retrieval_sources=True,
            file_search_active=True,
            chat_personality_enabled=True,
        )
        self.assertIn(FILE_SEARCH_SYSTEM_SUFFIX, compose_system_prompt(blocks))
        self.assertNotIn(CHAT_PERSONALITY_SUFFIX, compose_system_prompt(blocks))

        blocks = build_prompt_blocks(
            execution_route="MEMORY",
            explicit_remember_active=False,
            has_retrieval_sources=True,
            narrative_active=True,
            chat_personality_enabled=True,
        )
        sys_p = compose_system_prompt(blocks)
        self.assertIn(NARRATIVE_RECALL_SYSTEM_SUFFIX, sys_p)
        self.assertNotIn(CHAT_PERSONALITY_SUFFIX, sys_p)

    def test_absent_no_sources_mode(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="MEMORY",
            explicit_remember_active=False,
            has_retrieval_sources=False,
            chat_personality_enabled=True,
        )
        sys_p = compose_system_prompt(blocks)
        self.assertTrue(blocks.no_sources_mode)
        self.assertIn(NO_SOURCES_SYSTEM_SUFFIX, sys_p)
        self.assertNotIn(CHAT_PERSONALITY_SUFFIX, sys_p)

    def test_absent_web_capability_blocked(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="NONE",
            explicit_remember_active=False,
            web_capability_blocked=True,
            chat_personality_enabled=True,
        )
        self.assertNotIn(CHAT_PERSONALITY_SUFFIX, compose_system_prompt(blocks))

    def test_follow_up_none_turn_gets_no_cite_suffix(self) -> None:
        from core.memory_filters import CHAT_FOLLOW_UP_NO_SOURCES_SUFFIX

        blocks = build_prompt_blocks(
            execution_route="NONE",
            explicit_remember_active=False,
            has_retrieval_sources=False,
            follow_up_active=True,
        )
        sys_p = compose_system_prompt(blocks)
        self.assertIn(CHAT_FOLLOW_UP_NO_SOURCES_SUFFIX, sys_p)


if __name__ == "__main__":
    unittest.main()
