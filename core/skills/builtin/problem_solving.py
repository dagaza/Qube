"""Problem-solving framework skill."""

from __future__ import annotations

from core.skills.base import BuiltinSkill
from core.skills.types import SkillContext

_FOLLOW_UP_BOOST = 0.05


def _follow_up_boost(ctx: SkillContext) -> tuple[float, str | None]:
    if ctx.follow_up_active:
        return _FOLLOW_UP_BOOST, "boost:follow_up"
    return 0.0, None


PROBLEM_SOLVING = BuiltinSkill(
    id="problem_solving",
    name="Problem solving",
    description="Root-cause analysis, assumption checks, and tradeoff evaluation.",
    version="1.0.0",
    priority=88,
    max_prompt_chars=420,
    activation_triggers=(
        "root cause",
        "first principles",
        "why does this keep",
        "what's really going on",
        "underlying issue",
        "assumption",
        "trade-off",
        "tradeoff",
        "five whys",
    ),
    context_boost_fns=(_follow_up_boost,),
    prompt_fragment=(
        "Structure the reply: (1) restate the problem precisely, "
        "(2) list key assumptions and constraints, (3) identify likely root causes, "
        "(4) propose 2–3 solutions with tradeoffs, (5) recommend one path with caveats. "
        "Avoid jumping to a single answer before analysis."
    ),
)
