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
    compose_system_prompt,
    is_explicit_remember_persona,
)
from core.prompt_layout import PromptLayout, normalize_prompt_layout

_RETRIEVAL_WRAPPER_HEAD = (
    "=== SYSTEM RETRIEVED CONTEXT ===\n"
    "Use the following numbered sources to answer the query. "
    "In the prose of your reply, cite with plain tokens [1], [2], or [W] only—one id per "
    "bracket (never [1, 2, 3] combined), never [SOURCE 1] header echoes (no markdown links).\n\n"
)

_RETRIEVAL_WRAPPER_TAIL = "================================\n\nUSER QUERY:\n"

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


def _last_user_raw_content(blocks: PromptBlocks) -> str:
    for m in reversed(_normalize_history(blocks)):
        if m.get("role") == "user":
            c = str(m.get("content") or "")
            if _RETRIEVAL_WRAPPER_TAIL in c:
                return c.split(_RETRIEVAL_WRAPPER_TAIL, 1)[-1].strip()
            if "[USER QUESTION]\n" in c:
                return c.split("[USER QUESTION]\n", 1)[-1].strip()
            return c
    return ""


def _wrap_retrieval_legacy(user_content: str, retrieval_body: str) -> str:
    body = (retrieval_body or "").strip()
    if not body:
        return user_content
    return f"{_RETRIEVAL_WRAPPER_HEAD}{body}\n{_RETRIEVAL_WRAPPER_TAIL}{user_content}"


def _short_web_persona() -> str:
    return (
        "Use the live web results in context to answer the user's query. "
        "Cite web hits with [W] when only one block is tagged [W]; otherwise use "
        "[1], [2], etc. matching the bracket ids in context. Never write [SOURCE N]. "
        "Never reply with only a citation token."
    )


def _short_persona(blocks: PromptBlocks) -> str:
    if is_explicit_remember_persona(blocks.persona):
        return blocks.persona
    if blocks.execution_route in ("WEB", "INTERNET"):
        return _short_web_persona()
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
        lines.append(_short_web_persona())
    else:
        lines.append((blocks.persona or "").strip() or "Answer for the user in natural language.")
    for suf in _suffixes_for_compact_layout(blocks):
        chunk = " ".join(str(suf).split())
        if chunk:
            lines.append(chunk)
    return "\n".join(f"- {ln}" for ln in lines if ln.strip())


def _build_flatten_last_user(blocks: PromptBlocks) -> str:
    query = _last_user_raw_content(blocks)
    parts = [f"[ASSISTANT INSTRUCTIONS]\n{_flatten_instruction_bullets(blocks)}"]
    body = (blocks.retrieval_context or "").strip()
    if body:
        label = (
            "[REFERENCED CONVERSATION]"
            if blocks.composer_conversation_ref
            else "[RETRIEVED CONTEXT]"
        )
        parts.append(f"{label}\n{body}")
    parts.append(f"[USER QUESTION]\n{query}")
    return "\n\n".join(parts)


def render_system_ok_messages(blocks: PromptBlocks) -> list[dict[str, Any]]:
    """One system message + history; legacy retrieval wrapper on last user turn."""
    system_prompt = compose_system_prompt(blocks)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        *_normalize_history(blocks),
    ]
    body = (blocks.retrieval_context or "").strip()
    if body and messages and messages[-1].get("role") == "user":
        original = str(messages[-1].get("content") or "")
        if blocks.composer_conversation_ref:
            messages[-1]["content"] = _wrap_conversation_ref(original, body)
        else:
            messages[-1]["content"] = _wrap_retrieval_legacy(original, body)
    return messages


def render_short_system_messages(blocks: PromptBlocks) -> list[dict[str, Any]]:
    """Compact system block; retrieval stays on the last user message."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _compact_system_prompt(blocks)},
        *_normalize_history(blocks),
    ]
    body = (blocks.retrieval_context or "").strip()
    if body and messages and messages[-1].get("role") == "user":
        original = str(messages[-1].get("content") or "")
        if blocks.composer_conversation_ref:
            messages[-1]["content"] = _wrap_conversation_ref(original, body)
        else:
            messages[-1]["content"] = _wrap_retrieval_legacy(original, body)
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
