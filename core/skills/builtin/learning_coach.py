"""Learning coach skill."""

from __future__ import annotations

from core.skills.base import BuiltinSkill


LEARNING_COACH = BuiltinSkill(
    id="learning_coach",
    name="Learning coach",
    description="Curriculum design, practice, and spaced-repetition study plans.",
    version="1.0.0",
    priority=71,
    max_prompt_chars=400,
    activation_triggers=(
        "study plan",
        "learning plan",
        "curriculum",
        "spaced repetition",
        "practice problems",
        "knowledge gap",
        "want to learn",
        "learning path",
        "self-study",
        "how do i learn",
        "study schedule",
    ),
    prompt_fragment=(
        "Structure learning: current level → learning goals → ordered modules → "
        "practice exercises → checkpoints / self-quiz prompts → suggested review cadence. "
        "Keep modules small and actionable for self-study."
    ),
)
