"""Memory reflection skill."""

from __future__ import annotations

from core.skills.base import BuiltinSkill
from core.skills.types import SkillContext

_NARRATIVE_BOOST = 0.05


def _narrative_boost(ctx: SkillContext) -> tuple[float, str | None]:
    if ctx.narrative_active:
        return _NARRATIVE_BOOST, "boost:narrative"
    return 0.0, None


MEMORY_REFLECTION = BuiltinSkill(
    id="memory_reflection",
    name="Memory reflection",
    description="Reflect on past context and preferences with grounded recall.",
    version="1.0.0",
    priority=65,
    max_prompt_chars=380,
    activation_triggers=(
        "what did we discuss",
        "my preference",
        "journal",
        "reflect on",
        "looking back",
        "earlier you said",
        "you mentioned",
    ),
    context_boost_fns=(_narrative_boost,),
    prompt_fragment=(
        "Ground reflections in conversation history and retrieved memory. "
        "Distinguish confirmed facts from inference; ask one clarifying question "
        "if memory is thin."
    ),
)
