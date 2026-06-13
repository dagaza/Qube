"""Socratic tutor skill."""

from __future__ import annotations

from core.skills.base import BuiltinSkill


SOCRATIC_TUTOR = BuiltinSkill(
    id="socratic_tutor",
    name="Socratic tutor",
    description="Guided questioning and incremental hints instead of direct answers.",
    version="1.0.0",
    priority=72,
    max_prompt_chars=400,
    mutual_exclusion_group=None,
    activation_triggers=(
        "help me understand",
        "teach me",
        "quiz me",
        "don't give me the answer",
        "do not give me the answer",
        "explain like i'm",
        "walk me through the concept",
        "guide me through",
        "socratic",
        "hint only",
    ),
    prompt_fragment=(
        "Prefer guided learning: ask 1–2 probing questions, offer a small hint, "
        "check understanding, then build incrementally. Give the full answer only if "
        "the user explicitly asks or after two failed attempts. Avoid lecturing."
    ),
)
