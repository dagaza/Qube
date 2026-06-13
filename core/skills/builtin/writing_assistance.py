"""Writing assistance skill."""

from __future__ import annotations

from core.skills.base import BuiltinSkill
from core.skills.types import SkillContext

_NONE_ROUTE_BOOST = 0.05


def _none_route_boost(ctx: SkillContext) -> tuple[float, str | None]:
    if ctx.execution_route == "NONE":
        return _NONE_ROUTE_BOOST, "boost:none_route"
    return 0.0, None


WRITING_ASSISTANCE = BuiltinSkill(
    id="writing_assistance",
    name="Writing assistance",
    description="Drafting and editing structure without replacing user voice.",
    version="1.0.0",
    priority=70,
    max_prompt_chars=380,
    activation_triggers=(
        "rewrite",
        "draft",
        "email",
        "tone",
        "proofread",
        "make this clearer",
        "paragraph",
        "wording",
        "edit this",
        "polish",
    ),
    context_boost_fns=(_none_route_boost,),
    prompt_fragment=(
        "Preserve the user's intent and facts. Offer one revised version plus "
        "2–3 bullet edits (clarity, tone, brevity). Do not invent facts."
    ),
)
