"""
Layout-specific renderers for ``PromptBlocks`` (PR3).

``system_ok`` preserves PR2 parity; ``short_system`` and ``flatten_user`` are
compatibility layouts for weak system-role adherence (e.g. Mistral Instruct v0.x).
"""
from __future__ import annotations

from typing import Any

from core.memory_filters import RECALL_FUSION_SYSTEM_SUFFIX
from core.prompt_blocks import (
    PromptBlocks,
    RetrievalWrapperMode,
    compose_system_prompt,
    is_explicit_remember_persona,
)
from core.prompt_layout import PromptLayout, normalize_prompt_layout

_RETRIEVAL_WRAPPER_HEAD = (
    "=== SYSTEM RETRIEVED CONTEXT ===\n"
    "The following numbered sources are available for this answer.\n\n"
)

_RETRIEVAL_WRAPPER_HEAD_MULTI = (
    "=== SYSTEM RETRIEVED CONTEXT ===\n"
    "The following numbered sources are available for this answer. "
    "Do NOT use [W] when multiple numbered sources are listed.\n\n"
)

_WEB_CITATION_EXEMPLAR = (
    "=== CITATION FORMAT (follow exactly) ===\n"
    "Every factual sentence using the sources above must end with a citation token.\n\n"
    "Example style (pattern only):\n"
    "The first event was reported on Tuesday [1].\n"
    "A second development followed later the same day [2].\n\n"
    "Do not explain citations. Do not describe rules. Apply the pattern in your answer.\n"
)

_RETRIEVAL_WRAPPER_TAIL = "================================\n\nUSER QUERY:\n"

_BACKGROUND_WRAPPER_HEAD = (
    "=== POTENTIALLY RELEVANT USER CONTEXT (optional) ===\n"
    "Background preferences or context below may apply. Prefer the conversation "
    "history above unless directly relevant. Do NOT cite with [1] or [W] tokens.\n\n"
)

_BACKGROUND_WRAPPER_TAIL = "================================\n\nUSER QUERY:\n"

_CONVERSATION_REF_HEAD = (
    "=== REFERENCED CONVERSATION (this is a separate prior chat — "
    "answer ONLY from the transcript below) ===\n"
)
_CONVERSATION_REF_TAIL = "=== END REFERENCED CONVERSATION ===\n\nUSER QUESTION:\n"


def _wrap_conversation_ref(user_content: str, transcript_body: str) -> str:
    body = (transcript_body or "").strip()
    if not body:
        return user_content
    return f"{_CONVERSATION_REF_HEAD}{body}\n{_CONVERSATION_REF_TAIL}{user_content}"


def _normalize_history(blocks: PromptBlocks) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in blocks.conversation_history:
        if not isinstance(m, dict):
            continue
        out.append(
            {
                "role": str(m.get("role", "user")).strip().lower() or "user",
                "content": str(m.get("content") or ""),
            }
        )
    return out


def _strip_wrapped_query(content: str) -> str:
    c = str(content or "")
    for tail in (_RETRIEVAL_WRAPPER_TAIL, _BACKGROUND_WRAPPER_TAIL):
        if tail in c:
            return c.split(tail, 1)[-1].strip()
    if "[USER QUESTION]\n" in c:
        return c.split("[USER QUESTION]\n", 1)[-1].strip()
    return c


def _last_user_raw_content(blocks: PromptBlocks) -> str:
    for m in reversed(_normalize_history(blocks)):
        if m.get("role") == "user":
            return _strip_wrapped_query(str(m.get("content") or ""))
    return ""


def _retrieval_wrapper_head(blocks: PromptBlocks) -> str:
    count = int(getattr(blocks, "retrieval_source_count", 0) or 0)
    if count > 1:
        return _RETRIEVAL_WRAPPER_HEAD_MULTI
    return _RETRIEVAL_WRAPPER_HEAD


def _web_citation_exemplar(blocks: PromptBlocks | None) -> str:
    if blocks is None:
        return ""
    route = str(blocks.execution_route or "").upper()
    if route not in ("WEB", "INTERNET"):
        return ""
    if int(getattr(blocks, "retrieval_source_count", 0) or 0) < 1:
        return ""
    return _WEB_CITATION_EXEMPLAR


def _wrap_retrieval(
    user_content: str,
    retrieval_body: str,
    mode: RetrievalWrapperMode,
    *,
    blocks: PromptBlocks | None = None,
) -> str:
    body = (retrieval_body or "").strip()
    if not body or mode == "none":
        return user_content
    if mode == "background":
        return f"{_BACKGROUND_WRAPPER_HEAD}{body}\n{_BACKGROUND_WRAPPER_TAIL}{user_content}"
    head = _retrieval_wrapper_head(blocks) if blocks is not None else _RETRIEVAL_WRAPPER_HEAD
    exemplar = _web_citation_exemplar(blocks)
    if exemplar:
        return f"{head}{body}\n{exemplar}{_RETRIEVAL_WRAPPER_TAIL}{user_content}"
    return f"{head}{body}\n{_RETRIEVAL_WRAPPER_TAIL}{user_content}"


def _short_web_persona(blocks: PromptBlocks | None = None) -> str:
    multi = blocks is not None and int(getattr(blocks, "retrieval_source_count", 0) or 0) > 1
    base = (
        "Use the live web results in context to answer the user's query. "
        "Cite web hits with [1], [2], etc. matching the bracket ids in context. "
        "Never write [SOURCE N]. Never reply with only a citation token."
    )
    if multi:
        return base + " Do NOT use [W] when multiple sources are listed."
    return base


def _short_persona(blocks: PromptBlocks) -> str:
    if is_explicit_remember_persona(blocks.persona):
        return blocks.persona
    if blocks.execution_route in ("WEB", "INTERNET"):
        return _short_web_persona(blocks)
    return (blocks.persona or "").strip() or "Answer naturally and accurately."


def _suffixes_for_compact_layout(blocks: PromptBlocks) -> list[str]:
    """Drop verbose soft steering (PR4 will formalize hard vs soft tiers)."""
    out: list[str] = []
    for suf in blocks.system_suffixes:
        if suf == RECALL_FUSION_SYSTEM_SUFFIX:
            continue
        out.append(suf)
    return out


def _compact_system_prompt(blocks: PromptBlocks) -> str:
    text = _short_persona(blocks)
    for suf in _suffixes_for_compact_layout(blocks):
        text += suf
    return text


def _flatten_instruction_bullets(blocks: PromptBlocks) -> str:
    lines: list[str] = []
    if is_explicit_remember_persona(blocks.persona):
        lines.append(blocks.persona.strip())
    elif blocks.execution_route in ("WEB", "INTERNET"):
        lines.append(_short_web_persona(blocks))
    else:
        lines.append((blocks.persona or "").strip() or "Answer for the user in natural language.")
    for suf in _suffixes_for_compact_layout(blocks):
        chunk = " ".join(str(suf).split())
        if chunk:
            lines.append(chunk)
    return "\n".join(f"- {ln}" for ln in lines if ln.strip())


def _build_flatten_last_user(blocks: PromptBlocks) -> str:
    query = _last_user_raw_content(blocks)
    body = (blocks.retrieval_context or "").strip()
    mode = blocks.retrieval_wrapper_mode or "none"
    salience = (blocks.topic_salience_hint or "").strip()
    follow_up = bool(blocks.follow_up_active)

    parts = [f"[ASSISTANT INSTRUCTIONS]\n{_flatten_instruction_bullets(blocks)}"]

    if salience:
        parts.append(f"[ACTIVE TOPIC]\n{salience.strip()}")

    if body and mode == "background" and follow_up:
        parts.append(f"[USER QUESTION]\n{query}")
        parts.append(f"[BACKGROUND CONTEXT]\n{body}")
    else:
        if body:
            if blocks.composer_conversation_ref:
                label = "[REFERENCED CONVERSATION]"
            elif mode == "background":
                label = "[BACKGROUND CONTEXT]"
            else:
                label = "[RETRIEVED CONTEXT]"
            chunk = body
            exemplar = _web_citation_exemplar(blocks)
            if exemplar:
                chunk = f"{body}\n{exemplar.rstrip()}"
            parts.append(f"{label}\n{chunk}")
        parts.append(f"[USER QUESTION]\n{query}")

    return "\n\n".join(parts)


def render_system_ok_messages(blocks: PromptBlocks) -> list[dict[str, Any]]:
    """One system message + history; retrieval wrapper on last user turn."""
    system_prompt = compose_system_prompt(blocks)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        *_normalize_history(blocks),
    ]
    body = (blocks.retrieval_context or "").strip()
    mode = blocks.retrieval_wrapper_mode or "none"
    if body and messages and messages[-1].get("role") == "user":
        original = str(messages[-1].get("content") or "")
        if blocks.composer_conversation_ref:
            messages[-1]["content"] = _wrap_conversation_ref(original, body)
        else:
            messages[-1]["content"] = _wrap_retrieval(original, body, mode, blocks=blocks)
    return messages


def render_short_system_messages(blocks: PromptBlocks) -> list[dict[str, Any]]:
    """Compact system block; retrieval stays on the last user message."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _compact_system_prompt(blocks)},
        *_normalize_history(blocks),
    ]
    body = (blocks.retrieval_context or "").strip()
    mode = blocks.retrieval_wrapper_mode or "none"
    if body and messages and messages[-1].get("role") == "user":
        original = str(messages[-1].get("content") or "")
        if blocks.composer_conversation_ref:
            messages[-1]["content"] = _wrap_conversation_ref(original, body)
        else:
            messages[-1]["content"] = _wrap_retrieval(original, body, mode, blocks=blocks)
    return messages


def render_flattened_instruct_messages(blocks: PromptBlocks) -> list[dict[str, Any]]:
    """No system role — instructions + context + query on the last user turn."""
    hist = _normalize_history(blocks)
    if hist and hist[-1].get("role") == "user":
        prefix = hist[:-1]
    else:
        prefix = hist
    return [
        *prefix,
        {"role": "user", "content": _build_flatten_last_user(blocks)},
    ]


def render_messages(
    blocks: PromptBlocks,
    layout: PromptLayout | str,
) -> list[dict[str, Any]]:
    """Dispatch to the layout renderer (unknown values fall back to system_ok)."""
    lay = normalize_prompt_layout(layout) or "system_ok"
    if lay == "flatten_user":
        return render_flattened_instruct_messages(blocks)
    if lay == "short_system":
        return render_short_system_messages(blocks)
    return render_system_ok_messages(blocks)


def openai_messages_to_alpaca_prompt(messages: list[dict]) -> str:
    """
    Alpaca-style completion prompt for ``adaptive_retry`` rendered fallback.

    Merges any system messages into the instruction block, then uses the last user turn.
    """
    sys_parts: list[str] = []
    user_parts: list[str] = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "user")).strip().lower()
        content = str(m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            sys_parts.append(content)
        elif role == "user":
            user_parts.append(content)
    last_user = user_parts[-1] if user_parts else ""
    instruction = "\n\n".join(sys_parts).strip()
    if instruction:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{last_user}\n\n### Response:\n"
    return f"### Instruction:\n{last_user}\n\n### Response:\n"
