"""PR2: PromptBlocks builder + system_ok renderer parity with legacy LLMWorker assembly."""
from __future__ import annotations

import unittest

from core.memory_filters import (
    CITATION_DISCIPLINE_SUFFIX,
    COMPOSER_WEB_EMPTY_SUFFIX,
    CHAT_FOLLOW_UP_WEB_EMPTY_SUFFIX,
    EXPLICIT_WEB_EMPTY_SUFFIX,
    FILE_SEARCH_SYSTEM_SUFFIX,
    FINANCE_SOURCES_EMPTY_SUFFIX,
    GROUNDED_ANSWER_SYSTEM_SUFFIX,
    LEGAL_SOURCES_EMPTY_SUFFIX,
    NARRATIVE_RECALL_SYSTEM_SUFFIX,
    NO_SOURCES_SYSTEM_SUFFIX,
    RECALL_FUSION_SYSTEM_SUFFIX,
)
from core.prompt_blocks import (
    build_prompt_blocks,
    compose_system_prompt,
)
from core.prompt_renderers import render_system_ok_messages


def _legacy_compose(
    *,
    execution_route: str,
    explicit_remember_active: bool,
    explicit_remember_body: str = "",
    file_search_active: bool = False,
    narrative_active: bool = False,
    has_retrieval_sources: bool = False,
    engine_mode: str = "external",
    internal_nvidia: bool = False,
) -> str:
    """Inline replica of pre-PR2 LLMWorker system_prompt assembly."""
    system_prompt = (
        "You are Qube, a highly capable offline AI assistant. "
        "Answer naturally and accurately."
    )
    route = execution_route.upper()
    if explicit_remember_active:
        quoted = (explicit_remember_body or "").strip()
        system_prompt = (
            "The user has just asked you to remember a fact for future reference. "
            "Acknowledge briefly — one short sentence — that you've made a note of it, "
            "and optionally paraphrase the fact naturally. "
            "Do NOT use bracket tokens like [1], [2], or [W]. "
            "Do NOT cite sources. "
            "Do NOT say you cannot remember things; durable facts are persisted "
            "automatically for future turns."
        )
        if quoted:
            system_prompt += f' The fact to acknowledge is: "{quoted}".'
    elif route in ("RAG", "HYBRID", "MEMORY"):
        if not has_retrieval_sources:
            system_prompt = (
                "You are Qube, a highly capable offline AI assistant. "
                "Answer naturally and accurately."
            )
            system_prompt += NO_SOURCES_SYSTEM_SUFFIX
        else:
            system_prompt += (
                " You MUST cite your sources inline using brackets and the ID, like [1] or [2]. "
                "Write citations as plain bracket tokens only—one id per bracket (e.g. [1] and [2], "
                "never [1, 2, 3] in a single bracket)—do not wrap them in Markdown links, "
                "do not add URLs in parentheses after the token, and do not put them inside code fences or backticks."
            )
            system_prompt += RECALL_FUSION_SYSTEM_SUFFIX
            system_prompt += CITATION_DISCIPLINE_SUFFIX
            system_prompt += GROUNDED_ANSWER_SYSTEM_SUFFIX
            if file_search_active:
                system_prompt += FILE_SEARCH_SYSTEM_SUFFIX
            if narrative_active:
                system_prompt += NARRATIVE_RECALL_SYSTEM_SUFFIX
    elif route in ("WEB", "INTERNET"):
        system_prompt = (
            "Real-time live web search results have been provided for this turn. "
            "You MUST use the TOOLS context provided below to answer the user's query. "
            "Do not state that you are offline or cannot browse the internet. "
            "CRITICAL: Respond directly to the user in a natural, conversational tone. "
            "Do NOT output your internal reasoning, 'Step 1' thoughts, or search metadata. "
            "Write only the user-facing response. "
            "Cite using the numbered bracket ids from context ([1], [2], etc.)—"
            "never echo SOURCE headers, never use Markdown links or URLs after citations."
        )
        system_prompt += CITATION_DISCIPLINE_SUFFIX
    if engine_mode == "internal":
        if internal_nvidia:
            system_prompt += (
                " Start directly with the answer content in natural language. "
                "Do not narrate instructions, planning notes, request analysis, or hidden reasoning. "
                "Write only what the user should see. "
                "Prioritize clarity and completeness. "
                "Use short answers for simple questions, but give fuller explanations when the user asks to explain, compare, or summarize."
            )
        else:
            system_prompt += (
                " Start directly with the answer content in natural language. "
                "Do not include preamble, planning, or meta commentary. "
                "Do not restate or analyze the user's request. "
                "Write only what the user should see. "
                "Keep the response natural and focused."
            )
    return system_prompt


class TestPromptBlocks(unittest.TestCase):
    def test_parity_chat_none_internal(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="NONE",
            explicit_remember_active=False,
            engine_mode="internal",
            internal_nvidia_family=False,
        )
        self.assertEqual(compose_system_prompt(blocks), _legacy_compose(
            execution_route="NONE",
            explicit_remember_active=False,
            engine_mode="internal",
            internal_nvidia=False,
        ))

    def test_parity_no_sources_memory(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="MEMORY",
            explicit_remember_active=False,
            has_retrieval_sources=False,
            engine_mode="external",
        )
        self.assertTrue(blocks.no_sources_mode)
        self.assertIn(NO_SOURCES_SYSTEM_SUFFIX, compose_system_prompt(blocks))
        self.assertEqual(
            compose_system_prompt(blocks),
            _legacy_compose(
                execution_route="MEMORY",
                explicit_remember_active=False,
                has_retrieval_sources=False,
                engine_mode="external",
            ),
        )

    def test_parity_rag_with_sources_and_narrative(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="RAG",
            explicit_remember_active=False,
            has_retrieval_sources=True,
            narrative_active=True,
            file_search_active=True,
            engine_mode="external",
        )
        sys_p = compose_system_prompt(blocks)
        self.assertIn(NARRATIVE_RECALL_SYSTEM_SUFFIX, sys_p)
        self.assertIn(FILE_SEARCH_SYSTEM_SUFFIX, sys_p)
        self.assertIn(GROUNDED_ANSWER_SYSTEM_SUFFIX, sys_p)
        self.assertEqual(
            sys_p,
            _legacy_compose(
                execution_route="RAG",
                explicit_remember_active=False,
                has_retrieval_sources=True,
                narrative_active=True,
                file_search_active=True,
                engine_mode="external",
            ),
        )

    def test_parity_web_route(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="WEB",
            explicit_remember_active=False,
            has_retrieval_sources=True,
            retrieval_source_count=1,
            web_hit_count=1,
            engine_mode="external",
        )
        self.assertIn(CITATION_DISCIPLINE_SUFFIX, compose_system_prompt(blocks))
        self.assertEqual(
            compose_system_prompt(blocks),
            _legacy_compose(
                execution_route="WEB",
                explicit_remember_active=False,
                has_retrieval_sources=True,
                engine_mode="external",
            ),
        )

    def test_parity_explicit_remember(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="MEMORY",
            explicit_remember_active=True,
            explicit_remember_body="My dog is named Rex",
            engine_mode="external",
        )
        self.assertIn("Rex", compose_system_prompt(blocks))
        self.assertEqual(
            compose_system_prompt(blocks),
            _legacy_compose(
                execution_route="MEMORY",
                explicit_remember_active=True,
                explicit_remember_body="My dog is named Rex",
                engine_mode="external",
            ),
        )

    def test_render_injects_retrieval_on_last_user(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="RAG",
            explicit_remember_active=False,
            has_retrieval_sources=True,
            retrieval_context="[1] SOURCE: doc\nSnippet.",
            conversation_history=[
                {"role": "user", "content": "What is X?"},
            ],
        )
        messages = render_system_ok_messages(blocks)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn("=== SYSTEM RETRIEVED CONTEXT ===", messages[-1]["content"])
        self.assertIn("[1] SOURCE: doc", messages[-1]["content"])
        self.assertIn("USER QUERY:\nWhat is X?", messages[-1]["content"])

    def test_explicit_remember_avoids_brand_persona(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="NONE",
            explicit_remember_active=True,
            explicit_remember_body="Favorite color is green",
        )
        sys_p = compose_system_prompt(blocks)
        self.assertNotIn("You are Qube", sys_p)
        self.assertIn("persisted automatically", sys_p)

    def test_web_route_avoids_brand_persona(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="WEB",
            explicit_remember_active=False,
            has_retrieval_sources=True,
            retrieval_source_count=1,
            web_hit_count=1,
        )
        sys_p = compose_system_prompt(blocks)
        self.assertNotIn("You are Qube", sys_p)
        self.assertIn("Real-time live web search results", sys_p)

    def test_web_route_empty_results_uses_explicit_empty_suffix(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="WEB",
            explicit_remember_active=False,
            has_retrieval_sources=False,
            explicit_web_empty_results=True,
        )
        sys_p = compose_system_prompt(blocks)
        self.assertNotIn("Real-time live web search results", sys_p)
        self.assertIn(EXPLICIT_WEB_EMPTY_SUFFIX.strip()[:40], sys_p)

    def test_web_route_without_sources_defaults_to_empty_suffix(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="WEB",
            explicit_remember_active=False,
            has_retrieval_sources=False,
        )
        sys_p = compose_system_prompt(blocks)
        self.assertNotIn("Real-time live web search results", sys_p)
        self.assertIn(EXPLICIT_WEB_EMPTY_SUFFIX.strip()[:40], sys_p)
        self.assertTrue(blocks.no_sources_mode)

    def test_multi_web_suffix_forbids_w(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="WEB",
            explicit_remember_active=False,
            has_retrieval_sources=True,
            retrieval_source_count=2,
            web_hit_count=2,
        )
        sys_p = compose_system_prompt(blocks)
        self.assertIn("do NOT use [W] on this turn", sys_p)

    def test_web_retrieval_injects_citation_exemplar(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="WEB",
            explicit_remember_active=False,
            has_retrieval_sources=True,
            retrieval_context="--- [1]: Example ---\nSnippet.",
            retrieval_source_count=1,
            web_hit_count=1,
            conversation_history=[{"role": "user", "content": "What happened?"}],
        )
        messages = render_system_ok_messages(blocks)
        last = messages[-1]["content"]
        self.assertIn("=== CITATION FORMAT (follow exactly) ===", last)
        self.assertIn("Every factual sentence using the source above must end with [1].", last)
        self.assertIn("--- [1]: Example ---", last)
        q_idx = last.index("USER QUERY:")
        ex_idx = last.index("=== CITATION FORMAT (follow exactly) ===")
        src_idx = last.index("--- [1]: Example ---")
        self.assertLess(src_idx, ex_idx)
        self.assertLess(ex_idx, q_idx)

    def test_web_multi_retrieval_exemplar_avoids_numeric_examples(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="WEB",
            explicit_remember_active=False,
            has_retrieval_sources=True,
            retrieval_context="--- [1]: A ---\nOne.\n\n--- [2]: B ---\nTwo.",
            retrieval_source_count=2,
            web_hit_count=2,
            conversation_history=[{"role": "user", "content": "Compare them."}],
        )
        messages = render_system_ok_messages(blocks)
        last = messages[-1]["content"]
        self.assertIn("not sentence order or list numbering", last)
        self.assertIn("One factual sentence. [n]", last)
        self.assertNotIn("Tuesday [1]", last)
        self.assertNotIn("the same day [2]", last)

    def test_rag_retrieval_omits_web_citation_exemplar(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="RAG",
            explicit_remember_active=False,
            has_retrieval_sources=True,
            retrieval_context="--- [1]: Doc ---\nSnippet.",
            retrieval_source_count=1,
            conversation_history=[{"role": "user", "content": "What is X?"}],
        )
        messages = render_system_ok_messages(blocks)
        self.assertNotIn(
            "=== CITATION FORMAT (follow exactly) ===",
            messages[-1]["content"],
        )

    def test_base_chat_keeps_brand_persona(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="NONE",
            explicit_remember_active=False,
        )
        sys_p = compose_system_prompt(blocks)
        self.assertIn("You are Qube, a highly capable offline AI assistant.", sys_p)

        blocks = build_prompt_blocks(
            execution_route="NONE",
            explicit_remember_active=False,
            conversation_history=[{"role": "user", "content": "Hi"}],
        )
        messages = render_system_ok_messages(blocks)
        self.assertEqual(messages[-1]["content"], "Hi")

    def test_skill_guidance_injected_after_route_suffixes(self) -> None:
        guidance = (
            "=== REASONING GUIDANCE (non-authoritative) ===\n"
            "[Task decomposition] Break it down.\n"
            "=== END REASONING GUIDANCE ==="
        )
        blocks = build_prompt_blocks(
            execution_route="NONE",
            explicit_remember_active=False,
            skill_guidance=guidance,
        )
        sys_p = compose_system_prompt(blocks)
        self.assertIn("REASONING GUIDANCE", sys_p)
        self.assertIn("You are Qube", sys_p)

    def test_skill_guidance_skipped_on_explicit_remember(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="NONE",
            explicit_remember_active=True,
            explicit_remember_body="My cat is Rex",
            skill_guidance="=== REASONING GUIDANCE ===",
        )
        sys_p = compose_system_prompt(blocks)
        self.assertNotIn("REASONING GUIDANCE", sys_p)

    def test_legal_sources_empty_blocks_parametric_case_law(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="WEB",
            explicit_remember_active=False,
            has_retrieval_sources=False,
            legal_sources_empty=True,
            legal_disclaimer=True,
        )
        sys_p = compose_system_prompt(blocks)
        self.assertIn(LEGAL_SOURCES_EMPTY_SUFFIX, sys_p)
        self.assertTrue(blocks.no_sources_mode)

    def test_finance_sources_empty_blocks_parametric_filings(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="WEB",
            explicit_remember_active=False,
            has_retrieval_sources=False,
            finance_sources_empty=True,
            financial_disclaimer=True,
        )
        sys_p = compose_system_prompt(blocks)
        self.assertIn(FINANCE_SOURCES_EMPTY_SUFFIX, sys_p)
        self.assertTrue(blocks.no_sources_mode)

    def test_composer_web_empty_uses_composer_suffix(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="WEB",
            explicit_remember_active=False,
            has_retrieval_sources=False,
            explicit_web_empty_results=True,
            composer_web_empty=True,
        )
        sys_p = compose_system_prompt(blocks)
        self.assertIn(COMPOSER_WEB_EMPTY_SUFFIX.strip()[:40], sys_p)
        self.assertIn("pinned a web, fetch, recipe", sys_p)
        self.assertNotIn("explicitly asked for an online/web search", sys_p)

    def test_prior_web_empty_follow_up_suffix(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="NONE",
            explicit_remember_active=False,
            has_retrieval_sources=False,
            follow_up_active=True,
            prior_web_empty_follow_up=True,
        )
        sys_p = compose_system_prompt(blocks)
        self.assertIn(CHAT_FOLLOW_UP_WEB_EMPTY_SUFFIX.strip()[:40], sys_p)
        self.assertTrue(blocks.no_sources_mode)


if __name__ == "__main__":
    unittest.main()
