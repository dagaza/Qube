"""Tests for RAG capability veto prompt wiring and library intent detection."""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.memory_filters import (
    RAG_CAPABILITY_DISABLED_SUFFIX,
    STRICT_ISOLATION_SYSTEM_SUFFIX,
    detect_recall_intent,
    library_lane_allowed,
    query_implies_library_intent,
)
from core.prompt_blocks import build_prompt_blocks, compose_system_prompt


class LibraryLaneAllowedTests(unittest.TestCase):
    def test_master_on(self) -> None:
        self.assertTrue(
            library_lane_allowed(
                mcp_rag_enabled=True,
                force_rag_via_trigger=False,
                scoped_library_active=False,
            )
        )

    def test_bypass_trigger(self) -> None:
        self.assertTrue(
            library_lane_allowed(
                mcp_rag_enabled=False,
                force_rag_via_trigger=True,
                scoped_library_active=False,
            )
        )

    def test_blocked_when_off_and_no_bypass(self) -> None:
        self.assertFalse(
            library_lane_allowed(
                mcp_rag_enabled=False,
                force_rag_via_trigger=False,
                scoped_library_active=False,
            )
        )


class QueryImpliesLibraryIntentTests(unittest.TestCase):
    def test_recall_intent(self) -> None:
        self.assertTrue(query_implies_library_intent("Tell me about Dr. Evelyn."))

    def test_document_substring(self) -> None:
        self.assertTrue(
            query_implies_library_intent("What does the onboarding pdf say?")
        )

    def test_plain_chat_false(self) -> None:
        self.assertFalse(query_implies_library_intent("Tell me a joke."))

    def test_operational_library_prompt_false(self) -> None:
        self.assertFalse(
            query_implies_library_intent(
                "how can i remove entries from my library"
            )
        )

    def test_recall_fusion_decision_flag(self) -> None:
        self.assertTrue(
            query_implies_library_intent(
                "hello",
                decision={"recall_fusion": True},
            )
        )

    def test_detect_recall_intent_helper(self) -> None:
        self.assertTrue(detect_recall_intent("Remind me about metric units"))


class RagVetoPromptTests(unittest.TestCase):
    def test_rag_capability_blocked_suffix(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="NONE",
            explicit_remember_active=False,
            rag_capability_blocked=True,
        )
        system = compose_system_prompt(blocks)
        self.assertIn(RAG_CAPABILITY_DISABLED_SUFFIX.strip()[:40], system)

    def test_rag_capability_blocked_forbids_doc_citations(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="MEMORY",
            explicit_remember_active=False,
            rag_capability_blocked=True,
            has_retrieval_sources=True,
        )
        system = compose_system_prompt(blocks)
        self.assertIn("[1]", system)
        self.assertIn("Do NOT emit bracket citation tokens", system)

    def test_strict_isolation_suffix_on_retrieval_route(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="RAG",
            explicit_remember_active=False,
            has_retrieval_sources=True,
            strict_isolation_enabled=True,
        )
        system = compose_system_prompt(blocks)
        self.assertIn(STRICT_ISOLATION_SYSTEM_SUFFIX.strip()[:40], system)

    def test_strict_not_applied_when_capability_blocked(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="NONE",
            explicit_remember_active=False,
            rag_capability_blocked=True,
            strict_isolation_enabled=True,
        )
        system = compose_system_prompt(blocks)
        self.assertIn(RAG_CAPABILITY_DISABLED_SUFFIX.strip()[:40], system)
        self.assertNotIn(STRICT_ISOLATION_SYSTEM_SUFFIX.strip()[:40], system)


if __name__ == "__main__":
    unittest.main()
