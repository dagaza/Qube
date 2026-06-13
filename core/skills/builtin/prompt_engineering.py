"""Prompt engineering skill."""

from __future__ import annotations

from core.skills.base import BuiltinSkill


PROMPT_ENGINEERING = BuiltinSkill(
    id="prompt_engineering",
    name="Prompt engineering",
    description="Help users craft clearer goals, context, and constraints for AI tasks.",
    version="1.0.0",
    priority=76,
    max_prompt_chars=400,
    activation_triggers=(
        "better prompt",
        "improve this prompt",
        "how should i ask",
        "system prompt",
        "get better answers",
        "prompt template",
        "rewrite my prompt",
        "llm prompt",
        "ai prompt",
        "few-shot",
    ),
    prompt_fragment=(
        "Help refine the user's prompt: goal → required context → constraints → "
        "output format → evaluation criteria. Offer one revised prompt plus 2–3 "
        "specific improvements. Keep it usable with local/offline models."
    ),
)
