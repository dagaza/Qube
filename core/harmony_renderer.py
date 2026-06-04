"""
Canonical Harmony prompt renderer for gpt-oss / OpenAI Harmony GGUF models.

Produces a single assistant final-channel generation anchor per turn.
"""
from __future__ import annotations

from typing import Any

from core.harmony_protocol import HARMONY_FINAL_ANCHOR
from core.harmony_reply_guidance import merge_harmony_system_content


def _messages_payload(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        out.append({"role": m.get("role", "user"), "content": m.get("content") or ""})
    return out


def _harmony_has_prior_assistant(messages: list[dict[str, Any]]) -> bool:
    for m in _messages_payload(messages):
        role = str(m.get("role") or "user").strip().lower()
        content = str(m.get("content") or "").strip()
        if role == "assistant" and content:
            return True
    return False


def _collapse_duplicate_user_tail(
    dialogue: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    if len(dialogue) < 2:
        return dialogue
    if dialogue[-1][0] != "User" or dialogue[-2][0] != "User":
        return dialogue
    if dialogue[-1][1].strip() == dialogue[-2][1].strip():
        return dialogue[:-2] + [dialogue[-1]]
    return dialogue


def render_harmony_single_turn(payload: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    system_chunks: list[str] = []
    for m in payload:
        role = str(m.get("role") or "user").strip().lower()
        content = str(m.get("content") or "").strip()
        if not content or role == "assistant":
            continue
        if role == "system":
            system_chunks.append(content)
        else:
            parts.append(f"<|start|>user<|message|>{content}<|end|>")
    if system_chunks:
        parts.insert(
            0,
            f"<|start|>system<|message|>{merge_harmony_system_content(system_chunks)}<|end|>",
        )
    else:
        parts.insert(
            0,
            f"<|start|>system<|message|>{merge_harmony_system_content([])}<|end|>",
        )
    parts.append(HARMONY_FINAL_ANCHOR)
    return "\n".join(parts)


def render_harmony_compact_multiturn(payload: list[dict[str, Any]]) -> str:
    """
    One open final channel; prior turns live in a labeled user block.

    Multiple closed assistant final segments push gpt-oss into planning monologues.
    """
    parts: list[str] = []
    system_chunks: list[str] = []
    dialogue: list[tuple[str, str]] = []

    for m in payload:
        role = str(m.get("role") or "user").strip().lower()
        content = str(m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            system_chunks.append(content)
        elif role == "assistant":
            dialogue.append(("Assistant", content))
        else:
            dialogue.append(("User", content))

    if system_chunks:
        parts.append(
            f"<|start|>system<|message|>{merge_harmony_system_content(system_chunks)}<|end|>"
        )
    else:
        parts.insert(
            0,
            f"<|start|>system<|message|>{merge_harmony_system_content([])}<|end|>",
        )

    dialogue = _collapse_duplicate_user_tail(dialogue)
    if not dialogue:
        parts.append(HARMONY_FINAL_ANCHOR)
        return "\n".join(parts)

    if dialogue[-1][0] == "User":
        prior = dialogue[:-1]
        current_user = dialogue[-1][1]
    else:
        prior = dialogue
        current_user = ""

    body_lines: list[str] = []
    if prior:
        body_lines.append("[Conversation so far]")
        for label, text in prior:
            body_lines.append(f"{label}: {text}")
        body_lines.append("")
        body_lines.append("[Current message]")
    if current_user:
        body_lines.append(f"User: {current_user}")
    elif prior:
        body_lines.append("User: (continue from the conversation above)")

    parts.append(
        f"<|start|>user<|message|>{chr(10).join(body_lines)}<|end|>"
    )
    parts.append(HARMONY_FINAL_ANCHOR)
    return "\n".join(parts)


def render_harmony_final_prompt(messages: list[dict[str, Any]]) -> str:
    """
    Render Harmony prompt with assistant pre-filled in the final channel.

    Single-turn uses native Harmony roles. Multi-turn compacts history into one user
    block so only a single final-channel generation anchor remains.
    """
    payload = _messages_payload(messages)
    if _harmony_has_prior_assistant(messages):
        return render_harmony_compact_multiturn(payload)
    return render_harmony_single_turn(payload)
