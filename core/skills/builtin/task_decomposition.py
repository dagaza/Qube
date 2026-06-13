"""Task decomposition skill."""

from __future__ import annotations

from core.skills.base import BuiltinSkill
from core.skills.types import SkillContext

_FOLLOW_UP_BOOST = 0.06


def _follow_up_boost(ctx: SkillContext) -> tuple[float, str | None]:
    if ctx.follow_up_active:
        return _FOLLOW_UP_BOOST, "boost:follow_up"
    return 0.0, None


TASK_DECOMPOSITION = BuiltinSkill(
    id="task_decomposition",
    name="Task decomposition",
    description="Break complex asks into ordered steps before answering.",
    version="1.0.0",
    priority=90,
    max_prompt_chars=400,
    mutual_exclusion_group=None,
    activation_triggers=(
        "step by step",
        "break down",
        "how do i approach",
        "how should i approach",
        "roadmap",
        "walk me through",
        "multi-step",
        "step-by-step",
    ),
    context_boost_fns=(_follow_up_boost,),
    prompt_fragment=(
        "Before answering: (1) restate the goal in one sentence, "
        "(2) list 3–5 ordered steps, (3) execute each step in the final reply. "
        "Keep steps concrete; skip this structure if the question is a single fact."
    ),
)
