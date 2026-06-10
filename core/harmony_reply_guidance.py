"""
Extra system guidance for gpt-oss / Harmony final-channel completions.

Kept separate from the global persona so only Harmony rendered prompts receive it.
Wording avoids phrases that gpt-oss echoes as visible planning (e.g. ``final answer``).
"""
from __future__ import annotations

from core.chat_format_mode import ChatFormatMode
from core.reply_shape_policy import ReplyShapePolicy

HARMONY_FINAL_REPLY_GUIDANCE = (
    "Structure the user-visible reply as a brief opening line, then 2–4 short sections. "
    "Put each section label on its own line (examples: Cleaning, Temperature, Social). "
    "Write complete sentences in every section and end the reply after the last section. "
    "Do not restate the question, add a sources list, or narrate your reasoning process."
)

HARMONY_BRIEF_REPLY_GUIDANCE = (
    "Do not use section formatting. Respond in a single concise answer."
)

HARMONY_MIXED_REPLY_GUIDANCE = (
    "Use section formatting only if it improves clarity."
)

HARMONY_ENUMERATION_REPLY_GUIDANCE = (
    "Provide a complete enumerated answer: a brief opening line, then numbered or "
    "bulleted items. Stop after the list is complete; do not repeat items or restart "
    "numbering."
)


def harmony_reply_guidance_for_mode(mode: ChatFormatMode) -> str:
    if mode == "brief":
        return HARMONY_BRIEF_REPLY_GUIDANCE
    if mode == "mixed":
        return HARMONY_MIXED_REPLY_GUIDANCE
    return HARMONY_FINAL_REPLY_GUIDANCE


def harmony_reply_guidance_for_policy(policy: ReplyShapePolicy) -> str:
    """Harmony reply guidance from unified reply-shape policy."""
    if policy.format_intent == "enumeration":
        return HARMONY_ENUMERATION_REPLY_GUIDANCE
    return harmony_reply_guidance_for_mode(policy.chat_format_mode)


def merge_harmony_system_content(
    system_chunks: list[str],
    *,
    include_reply_guidance: bool = True,
    chat_format_mode: ChatFormatMode = "structured",
    reply_shape_policy: ReplyShapePolicy | None = None,
) -> str:
    """Append Harmony reply shape hints to the system message when allowed (idempotent)."""
    parts = [str(s).strip() for s in system_chunks if str(s).strip()]
    merged = "\n\n".join(parts).strip()
    if not include_reply_guidance:
        return merged
    if reply_shape_policy is not None:
        guidance = harmony_reply_guidance_for_policy(reply_shape_policy)
    else:
        guidance = harmony_reply_guidance_for_mode(chat_format_mode)
    if guidance in merged:
        return merged
    if merged:
        return f"{merged}\n\n{guidance}"
    return guidance
