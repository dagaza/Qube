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
    detect_file_search_intent,
    library_lane_allowed,
    query_explicitly_requests_library_search,
    query_has_lexical_library_signal,
    query_implies_library_intent,
    should_apply_recall_fusion,
    should_downgrade_embedding_rag_on_continuation,
    should_downgrade_short_vague_retrieval_on_first_turn,
    is_conversational_continuation_turn,
    _router_embedding_implies_library_intent,
)
from core.prompt_blocks import build_prompt_blocks, compose_system_prompt

USER_ACRONYM_PROMPT = (
    "It's actually not, it's something like BDO or BOD. "
    "It sounds like a 3 letter acronym"
)


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


class QueryExplicitlyRequestsLibrarySearchTests(unittest.TestCase):
    def test_recall_intent_personal_phrase_without_decision(self) -> None:
        self.assertFalse(
            query_explicitly_requests_library_search("Tell me about Dr. Evelyn.")
        )

    def test_document_substring(self) -> None:
        self.assertTrue(
            query_explicitly_requests_library_search("What does the onboarding pdf say?")
        )

    def test_plain_chat_false(self) -> None:
        self.assertFalse(query_explicitly_requests_library_search("Tell me a joke."))

    def test_general_knowledge_false(self) -> None:
        self.assertFalse(
            query_explicitly_requests_library_search("What is the capital of Nepal?")
        )

    def test_operational_library_prompt_false(self) -> None:
        self.assertFalse(
            query_explicitly_requests_library_search(
                "how can i remove entries from my library"
            )
        )

    def test_user_acronym_correction_false(self) -> None:
        self.assertFalse(query_explicitly_requests_library_search(USER_ACRONYM_PROMPT))

    def test_recall_fusion_decision_flag_does_not_imply_explicit_request(self) -> None:
        self.assertFalse(
            query_explicitly_requests_library_search(
                "hello",
                decision={"recall_fusion": True},
            )
        )

    def test_embedding_router_signal_does_not_imply_explicit_request(self) -> None:
        decision = {
            "top_intent": "rag",
            "top_intent_source": "embedding",
            "top_score": 0.42,
            "rag_score_source": "embedding",
            "rag_score_final": 0.42,
        }
        self.assertTrue(_router_embedding_implies_library_intent(decision))
        self.assertFalse(
            query_explicitly_requests_library_search(USER_ACRONYM_PROMPT, decision=decision)
        )

    def test_alias_matches_explicit_helper(self) -> None:
        self.assertEqual(
            query_implies_library_intent("What does the onboarding pdf say?"),
            query_explicitly_requests_library_search("What does the onboarding pdf say?"),
        )

    def test_lexical_helper(self) -> None:
        self.assertTrue(query_has_lexical_library_signal("What does the onboarding pdf say?"))
        self.assertFalse(query_has_lexical_library_signal(USER_ACRONYM_PROMPT))


class ShouldApplyRecallFusionTests(unittest.TestCase):
    def test_personal_recall_phrase(self) -> None:
        self.assertTrue(
            should_apply_recall_fusion("Remind me about metric units")
        )

    def test_general_knowledge_blocked(self) -> None:
        self.assertFalse(
            should_apply_recall_fusion("What is the capital of Nepal?")
        )

    def test_honors_router_recall_active(self) -> None:
        self.assertTrue(
            should_apply_recall_fusion(
                "Tell me about Dr. Evelyn.",
                decision={"recall_active": True},
            )
        )

    def test_honors_router_chat_margin_block(self) -> None:
        self.assertFalse(
            should_apply_recall_fusion(
                "What is the capital of Nepal?",
                decision={
                    "recall_active": False,
                    "recall_score": 1.0,
                    "recall_threshold": 0.62,
                },
            )
        )


class ContinuationRoutingTests(unittest.TestCase):
    def test_is_conversational_continuation_on_correction(self) -> None:
        self.assertTrue(
            is_conversational_continuation_turn(
                USER_ACRONYM_PROMPT,
                follow_up_active=False,
                prior_execution_route="NONE",
                has_chat_history=True,
            )
        )

    def test_not_continuation_without_history(self) -> None:
        self.assertFalse(
            is_conversational_continuation_turn(
                USER_ACRONYM_PROMPT,
                follow_up_active=False,
                prior_execution_route="NONE",
                has_chat_history=False,
            )
        )

    def test_not_continuation_when_prior_route_was_rag(self) -> None:
        self.assertFalse(
            is_conversational_continuation_turn(
                USER_ACRONYM_PROMPT,
                follow_up_active=False,
                prior_execution_route="RAG",
                has_chat_history=True,
            )
        )

    def test_downgrade_embedding_rag_on_user_session_prompt(self) -> None:
        decision = {
            "top_intent": "rag",
            "top_intent_source": "embedding",
            "top_score": 0.42,
            "rag_score_source": "embedding",
            "rag_score_final": 0.42,
        }
        self.assertTrue(
            should_downgrade_embedding_rag_on_continuation(
                USER_ACRONYM_PROMPT,
                decision=decision,
                execution_route="RAG",
                prior_execution_route="NONE",
                follow_up_active=False,
                has_chat_history=True,
            )
        )

    def test_no_downgrade_when_explicit_library_request(self) -> None:
        decision = {
            "top_intent": "rag",
            "top_intent_source": "embedding",
            "top_score": 0.42,
            "rag_score_source": "embedding",
        }
        self.assertFalse(
            should_downgrade_embedding_rag_on_continuation(
                "What does the onboarding pdf say?",
                decision=decision,
                execution_route="RAG",
                prior_execution_route="NONE",
                follow_up_active=False,
                has_chat_history=True,
            )
        )

    def test_no_downgrade_when_recall_active(self) -> None:
        decision = {
            "recall_active": True,
            "top_intent": "recall",
            "rag_score_source": "embedding",
        }
        self.assertFalse(
            should_downgrade_embedding_rag_on_continuation(
                USER_ACRONYM_PROMPT,
                decision=decision,
                execution_route="HYBRID",
                prior_execution_route="NONE",
                follow_up_active=False,
                has_chat_history=True,
            )
        )


class ShortVagueFirstTurnRetrievalTests(unittest.TestCase):
    def test_downgrades_embedding_hybrid_on_test(self) -> None:
        decision = {
            "memory_score_source": "embedding",
            "rag_score_source": "embedding",
            "memory_score_final": 0.841,
            "rag_score_final": 0.839,
            "chat_score": 0.857,
        }
        self.assertTrue(
            should_downgrade_short_vague_retrieval_on_first_turn(
                "Test",
                decision=decision,
                execution_route="HYBRID",
                has_chat_history=False,
            )
        )

    def test_no_downgrade_when_substring_library_signal(self) -> None:
        decision = {
            "memory_score_source": "embedding",
            "rag_score_source": "substring",
            "rag_score_final": 0.40,
            "chat_score": 0.50,
        }
        self.assertFalse(
            should_downgrade_short_vague_retrieval_on_first_turn(
                "according to my notes",
                decision=decision,
                execution_route="RAG",
                has_chat_history=False,
            )
        )

    def test_no_downgrade_on_follow_up_turn(self) -> None:
        decision = {
            "memory_score_source": "embedding",
            "rag_score_source": "embedding",
            "memory_score_final": 0.841,
            "rag_score_final": 0.839,
            "chat_score": 0.857,
        }
        self.assertFalse(
            should_downgrade_short_vague_retrieval_on_first_turn(
                "Test",
                decision=decision,
                execution_route="HYBRID",
                has_chat_history=True,
            )
        )


class RagCapabilityBlockedLogicTests(unittest.TestCase):
    """Mirror LLMWorker rag_capability_blocked gate (options 1 + 2)."""

    def _blocked(
        self,
        *,
        clean_prompt: str,
        execution_route: str = "NONE",
        rag_vetoed: bool = True,
        library_blocked: bool = True,
        decision: dict | None = None,
    ) -> bool:
        explicit = query_explicitly_requests_library_search(
            clean_prompt, decision=decision
        )
        return bool(
            library_blocked
            and explicit
            and execution_route in ("NONE", "MEMORY")
            and (
                rag_vetoed
                or detect_file_search_intent(clean_prompt)
                or query_has_lexical_library_signal(clean_prompt)
            )
        )

    def test_embedding_only_veto_does_not_block(self) -> None:
        decision = {
            "top_intent": "rag",
            "top_intent_source": "embedding",
            "top_score": 0.42,
        }
        self.assertFalse(
            self._blocked(
                clean_prompt=USER_ACRONYM_PROMPT,
                rag_vetoed=True,
                decision=decision,
            )
        )

    def test_lexical_pdf_with_veto_blocks(self) -> None:
        self.assertTrue(
            self._blocked(
                clean_prompt="What does the onboarding pdf say?",
                rag_vetoed=True,
            )
        )


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
