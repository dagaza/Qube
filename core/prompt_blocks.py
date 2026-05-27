"""
Canonical per-turn prompt blocks for chat (PR2).

``build_prompt_blocks`` centralizes route/persona/suffix assembly.
Rendering lives in ``core/prompt_renderers.py`` (PR3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

RetrievalWrapperMode = Literal["grounded", "background", "none"]

from core.memory_filters import (
    CITATION_DISCIPLINE_SUFFIX,
    CONVERSATION_REF_SYSTEM_SUFFIX,
    FILE_SEARCH_SYSTEM_SUFFIX,
    GROUNDED_ANSWER_SYSTEM_SUFFIX,
    NARRATIVE_RECALL_SYSTEM_SUFFIX,
    NO_SOURCES_SYSTEM_SUFFIX,
    PREFERENCE_APPLICATION_SUFFIX,
    RECALL_FUSION_SYSTEM_SUFFIX,
    WEB_CAPABILITY_DISABLED_SUFFIX,
)

_BASE_PERSONA = (
    "You are Qube, a highly capable offline AI assistant. "
    "Answer naturally and accurately."
)

_RETRIEVAL_CITE_MUST = (
    " You MUST cite your sources inline using brackets and the ID, like [1] or [2]. "
    "Write citations as plain bracket tokens only—one id per bracket (e.g. [1] and [2], "
    "never [1, 2, 3] in a single bracket)—do not wrap them in Markdown links, "
    "do not add URLs in parentheses after the token, and do not put them inside code fences or backticks."
)

_EXPLICIT_REMEMBER_PERSONA_HEAD = (
    "The user has just asked you to remember a fact for future reference."
)

_WEB_PERSONA = (
    "Real-time live web search results have been provided for this turn. "
    "You MUST use the TOOLS context provided below to answer the user's query. "
    "Do not state that you are offline or cannot browse the internet. "
    "CRITICAL: Respond directly to the user in a natural, conversational tone. "
    "Do NOT output your internal reasoning, 'Step 1' thoughts, or search metadata. "
    "Write only the user-facing response. "
    "Cite web sources using only the bracket ids shown on each context block (e.g. [W] "
    "when there is a single web hit, or [1] and [2] when multiple web hits are listed)—"
    "never echo header words like SOURCE or [SOURCE 1], never labels like "
    "[W: Live Web Search], no Markdown hyperlink syntax, "
    "no URL in parentheses after the citation token, and no backticks around citations. "
    "Use separate brackets per source ([1], [2], or [W])—never combine ids like [1, 2, 3]. "
    "Use [W] at most once at the end of each sentence that relies on the web results, "
    "and never output [W] two or more times in a row."
)

_INTERNAL_ALIGN_NVIDIA = (
    " Start directly with the answer content in natural language. "
    "Do not narrate instructions, planning notes, request analysis, or hidden reasoning. "
    "Write only what the user should see. "
    "Prioritize clarity and completeness. "
    "Use short answers for simple questions, but give fuller explanations when the user asks to explain, compare, or summarize."
)

_INTERNAL_ALIGN_DEFAULT = (
    " Start directly with the answer content in natural language. "
    "Do not include preamble, planning, or meta commentary. "
    "Do not restate or analyze the user's request. "
    "Write only what the user should see. "
    "Keep the response natural and focused."
)

@dataclass
class PromptBlocks:
    """Structured prompt pieces before layout-specific rendering."""

    persona: str
    system_suffixes: list[str] = field(default_factory=list)
    retrieval_context: str = ""
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    no_sources_mode: bool = False
    execution_route: str = ""
    composer_conversation_ref: bool = False
    retrieval_wrapper_mode: RetrievalWrapperMode = "none"
    topic_salience_hint: str = ""
    follow_up_active: bool = False


def resolve_retrieval_wrapper_mode(
    execution_route: str,
    has_retrieval_sources: bool,
    *,
    memory_only_sources: bool = False,
) -> RetrievalWrapperMode:
    """Choose retrieval framing: grounded (RAG/recall), background (CHAT core memory), none."""
    if not has_retrieval_sources:
        return "none"
    route = str(execution_route or "").upper()
    if route == "NONE" and memory_only_sources:
        return "background"
    if route in ("RAG", "HYBRID", "MEMORY", "WEB", "INTERNET"):
        return "grounded"
    if route == "NONE":
        return "background"
    return "grounded"


def is_explicit_remember_persona(persona: str) -> bool:
    """True when ``persona`` is the explicit-remember acknowledgement block."""
    return (persona or "").startswith(_EXPLICIT_REMEMBER_PERSONA_HEAD)


def build_prompt_blocks(
    *,
    execution_route: str,
    explicit_remember_active: bool,
    explicit_remember_body: str = "",
    file_search_active: bool = False,
    narrative_active: bool = False,
    has_retrieval_sources: bool = False,
    engine_mode: str = "internal",
    internal_nvidia_family: bool = False,
    retrieval_context: str = "",
    conversation_history: list[dict[str, Any]] | None = None,
    composer_conversation_ref: bool = False,
    web_capability_blocked: bool = False,
    preference_context: str = "",
    apply_preference_suffix: bool = False,
    retrieval_wrapper_mode: RetrievalWrapperMode | None = None,
    topic_salience_hint: str = "",
    follow_up_active: bool = False,
) -> PromptBlocks:
    """
    Assemble persona + suffix lists for the current turn.

    Mirrors the former inline logic in ``LLMWorker`` (PR2 normalization only).
    """
    route = str(execution_route or "").upper()
    suffixes: list[str] = []
    persona = _BASE_PERSONA
    no_sources = False

    if explicit_remember_active:
        quoted = (explicit_remember_body or "").strip()
        persona = (
            "The user has just asked you to remember a fact for future reference. "
            "Acknowledge briefly — one short sentence — that you've made a note of it, "
            "and optionally paraphrase the fact naturally. "
            "Do NOT use bracket tokens like [1], [2], or [W]. "
            "Do NOT cite sources. "
            "Do NOT say you cannot remember things; durable facts are persisted "
            "automatically for future turns."
        )
        if quoted:
            persona += f' The fact to acknowledge is: "{quoted}".'
    elif web_capability_blocked:
        persona = _BASE_PERSONA
        suffixes.append(WEB_CAPABILITY_DISABLED_SUFFIX)
    elif route in ("RAG", "HYBRID", "MEMORY"):
        if not has_retrieval_sources:
            no_sources = True
            persona = _BASE_PERSONA
            suffixes.append(NO_SOURCES_SYSTEM_SUFFIX)
        else:
            suffixes.append(_RETRIEVAL_CITE_MUST)
            suffixes.append(RECALL_FUSION_SYSTEM_SUFFIX)
            suffixes.append(CITATION_DISCIPLINE_SUFFIX)
            suffixes.append(GROUNDED_ANSWER_SYSTEM_SUFFIX)
            if file_search_active:
                suffixes.append(FILE_SEARCH_SYSTEM_SUFFIX)
            if narrative_active:
                suffixes.append(NARRATIVE_RECALL_SYSTEM_SUFFIX)
    elif route in ("WEB", "INTERNET"):
        persona = _WEB_PERSONA
        suffixes.append(CITATION_DISCIPLINE_SUFFIX)
    elif composer_conversation_ref and (retrieval_context or "").strip():
        suffixes.append(CONVERSATION_REF_SYSTEM_SUFFIX)

    if apply_preference_suffix and not explicit_remember_active:
        suffixes.append(PREFERENCE_APPLICATION_SUFFIX)
    pref_ctx = (preference_context or "").strip()
    if pref_ctx and not explicit_remember_active:
        suffixes.append(f" {pref_ctx}")

    salience = (topic_salience_hint or "").strip()
    if salience and not explicit_remember_active:
        suffixes.append(salience)

    if str(engine_mode or "").lower() == "internal":
        suffixes.append(
            _INTERNAL_ALIGN_NVIDIA if internal_nvidia_family else _INTERNAL_ALIGN_DEFAULT
        )

    wrapper_mode = retrieval_wrapper_mode
    if wrapper_mode is None:
        wrapper_mode = resolve_retrieval_wrapper_mode(
            route,
            has_retrieval_sources,
            memory_only_sources=False,
        )

    return PromptBlocks(
        persona=persona,
        system_suffixes=suffixes,
        retrieval_context=str(retrieval_context or ""),
        conversation_history=list(conversation_history or []),
        no_sources_mode=no_sources,
        execution_route=route,
        composer_conversation_ref=bool(composer_conversation_ref),
        retrieval_wrapper_mode=wrapper_mode,
        topic_salience_hint=salience,
        follow_up_active=bool(follow_up_active),
    )


def compose_system_prompt(blocks: PromptBlocks) -> str:
    """Concatenate persona + suffixes (legacy system_ok shape)."""
    out = blocks.persona or ""
    for suf in blocks.system_suffixes:
        out += suf
    return out


from core.prompt_renderers import render_system_ok_messages  # noqa: E402,F401
