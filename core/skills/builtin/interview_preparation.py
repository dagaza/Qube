"""Interview preparation skill."""

from __future__ import annotations

from core.skills.base import BuiltinSkill


INTERVIEW_PREPARATION = BuiltinSkill(
    id="interview_preparation",
    name="Interview preparation",
    description="Mock interviews, STAR responses, and resume alignment.",
    version="1.0.0",
    priority=69,
    max_prompt_chars=400,
    activation_triggers=(
        "mock interview",
        "interview prep",
        "interview question",
        "star method",
        "star response",
        "tell me about a time",
        "resume bullet",
        "job interview",
        "behavioral interview",
        "prepare for interview",
    ),
    prompt_fragment=(
        "For interview prep: clarify role/context → likely question themes → "
        "STAR-format answer outlines (Situation, Task, Action, Result) → "
        "weak spots to rehearse → concise follow-up questions to ask the employer."
    ),
)
