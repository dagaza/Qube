"""
Extra system guidance for gpt-oss / Harmony final-channel completions.

Kept separate from the global persona so only Harmony rendered prompts receive it.
Wording avoids phrases that gpt-oss echoes as visible planning (e.g. ``final answer``).
"""

HARMONY_FINAL_REPLY_GUIDANCE = (
    "Structure the user-visible reply as a brief opening line, then 2–4 short sections. "
    "Put each section label on its own line (examples: Cleaning, Temperature, Social). "
    "Write complete sentences in every section and end the reply after the last section. "
    "Do not restate the question, add a sources list, or narrate your reasoning process."
)


def merge_harmony_system_content(system_chunks: list[str]) -> str:
    """Append Harmony reply shape hints to the system message (idempotent)."""
    parts = [str(s).strip() for s in system_chunks if str(s).strip()]
    merged = "\n\n".join(parts).strip()
    if HARMONY_FINAL_REPLY_GUIDANCE in merged:
        return merged
    if merged:
        return f"{merged}\n\n{HARMONY_FINAL_REPLY_GUIDANCE}"
    return HARMONY_FINAL_REPLY_GUIDANCE
