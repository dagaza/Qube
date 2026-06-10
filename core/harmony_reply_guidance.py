"""
Extra system guidance for gpt-oss / Harmony final-channel completions.

Kept separate from the global persona so only Harmony rendered prompts receive it.
Wording avoids phrases that gpt-oss echoes as visible planning (e.g. ``final answer``).
"""
from __future__ import annotations

from core.chat_format_mode import ChatFormatMode

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


def harmony_reply_guidance_for_mode(mode: ChatFormatMode) -> str:
    if mode == "brief":
        return HARMONY_BRIEF_REPLY_GUIDANCE
    if mode == "mixed":
        return HARMONY_MIXED_REPLY_GUIDANCE
    return HARMONY_FINAL_REPLY_GUIDANCE


def merge_harmony_system_content(
    system_chunks: list[str],
    *,
    include_reply_guidance: bool = True,
    chat_format_mode: ChatFormatMode = "structured",
) -> str:
    """Append Harmony reply shape hints to the system message when allowed (idempotent)."""
    parts = [str(s).strip() for s in system_chunks if str(s).strip()]
    merged = "\n\n".join(parts).strip()
    if not include_reply_guidance:
        return merged
    guidance = harmony_reply_guidance_for_mode(chat_format_mode)
    if guidance in merged:
        return merged
    if merged:
        return f"{merged}\n\n{guidance}"
    return guidance
