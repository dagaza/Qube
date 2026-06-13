"""Productivity and planning skill."""

from __future__ import annotations

from core.skills.base import BuiltinSkill
from core.skills.types import SkillContext

_FOLLOW_UP_BOOST = 0.06


def _follow_up_boost(ctx: SkillContext) -> tuple[float, str | None]:
    if ctx.follow_up_active:
        return _FOLLOW_UP_BOOST, "boost:follow_up"
    return 0.0, None


PRODUCTIVITY_PLANNING = BuiltinSkill(
    id="productivity_planning",
    name="Productivity planning",
    description="Time/task prioritization and actionable next steps.",
    version="1.0.0",
    priority=80,
    max_prompt_chars=400,
    mutual_exclusion_group="planning",
    activation_triggers=(
        "prioritize",
        "schedule",
        "deadline",
        "todo",
        "to-do",
        "organize my",
        "week plan",
        "productivity",
        "time management",
        "action items",
    ),
    context_boost_fns=(_follow_up_boost,),
    prompt_fragment=(
        "Structure the reply as: Context → Top 3 priorities → Next actions "
        "(verb-led, time-estimated) → Optional calendar notes. "
        "Prefer feasible steps over exhaustive lists."
    ),
)
