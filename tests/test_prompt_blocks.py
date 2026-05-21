"""PR2: PromptBlocks builder + system_ok renderer parity with legacy LLMWorker assembly."""
from __future__ import annotations

import unittest

from core.memory_filters import (
    CITATION_DISCIPLINE_SUFFIX,
    FILE_SEARCH_SYSTEM_SUFFIX,
    GROUNDED_ANSWER_SYSTEM_SUFFIX,
    NARRATIVE_RECALL_SYSTEM_SUFFIX,
    NO_SOURCES_SYSTEM_SUFFIX,
    RECALL_FUSION_SYSTEM_SUFFIX,
)
from core.prompt_blocks import (
    build_prompt_blocks,
    compose_system_prompt,
    render_system_ok_messages,
)


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
            "You are Qube. The user has just asked you to remember a fact for future reference. "
            "Acknowledge briefly — one short sentence — that you've made a note of it, "
            "and optionally paraphrase the fact naturally. "
            "Do NOT use bracket tokens like [1], [2], or [W]. "
            "Do NOT cite sources. "
            "Do NOT say you cannot remember things; Qube persists long-term memories automatically."
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
                "Write citations as plain bracket tokens only—do not wrap them in Markdown links, "
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
            "You are Qube. You have just been provided with real-time, live web search results. "
            "You MUST use the TOOLS context provided below to answer the user's query. "
            "Do not state that you are offline or cannot browse the internet. "
            "CRITICAL: Respond directly to the user in a natural, conversational tone. "
            "Do NOT output your internal reasoning, 'Step 1' thoughts, or search metadata. "
            "Write only the user-facing response. "
            "Cite web sources using only the plain id from the SOURCE blocks (e.g. [W] or [1])—"
            "never labels like [W: Live Web Search], no Markdown hyperlink syntax, "
            "no URL in parentheses after the citation token, and no backticks around citations. "
            "Use [W] at most once at the end of each sentence that relies on the web results, "
            "and never output [W] two or more times in a row."
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
        )
        self.assertTrue(blocks.no_sources_mode)
        self.assertIn(NO_SOURCES_SYSTEM_SUFFIX, compose_system_prompt(blocks))
        self.assertEqual(
            compose_system_prompt(blocks),
            _legacy_compose(
                execution_route="MEMORY",
                explicit_remember_active=False,
                has_retrieval_sources=False,
            ),
        )

    def test_parity_rag_with_sources_and_narrative(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="RAG",
            explicit_remember_active=False,
            has_retrieval_sources=True,
            narrative_active=True,
            file_search_active=True,
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
            ),
        )

    def test_parity_web_route(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="WEB",
            explicit_remember_active=False,
        )
        self.assertIn(CITATION_DISCIPLINE_SUFFIX, compose_system_prompt(blocks))
        self.assertEqual(
            compose_system_prompt(blocks),
            _legacy_compose(execution_route="WEB", explicit_remember_active=False),
        )

    def test_parity_explicit_remember(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="MEMORY",
            explicit_remember_active=True,
            explicit_remember_body="My dog is named Rex",
        )
        self.assertIn("Rex", compose_system_prompt(blocks))
        self.assertEqual(
            compose_system_prompt(blocks),
            _legacy_compose(
                execution_route="MEMORY",
                explicit_remember_active=True,
                explicit_remember_body="My dog is named Rex",
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

    def test_render_skips_wrapper_without_context(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="NONE",
            explicit_remember_active=False,
            conversation_history=[{"role": "user", "content": "Hi"}],
        )
        messages = render_system_ok_messages(blocks)
        self.assertEqual(messages[-1]["content"], "Hi")


if __name__ == "__main__":
    unittest.main()
