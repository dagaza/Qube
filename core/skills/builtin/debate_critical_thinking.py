"""Debate and critical thinking skill."""

from __future__ import annotations

from core.skills.base import BuiltinSkill
from core.skills.types import SkillContext

_SOURCES_BOOST = 0.06


def _sources_boost(ctx: SkillContext) -> tuple[float, str | None]:
    if ctx.has_retrieval_sources:
        return _SOURCES_BOOST, "boost:has_sources"
    return 0.0, None


DEBATE_CRITICAL_THINKING = BuiltinSkill(
    id="debate_critical_thinking",
    name="Debate & critical thinking",
    description="Steelmanning, counterarguments, and evidence-weighted reasoning.",
    version="1.0.0",
    priority=74,
    max_prompt_chars=420,
    activation_triggers=(
        "counterargument",
        "steelman",
        "devil's advocate",
        "devils advocate",
        "both sides",
        "argue against",
        "logical fallacy",
        "bias",
        "critically evaluate",
        "challenge this claim",
    ),
    context_boost_fns=(_sources_boost,),
    prompt_fragment=(
        "Present: strongest version of each side (steelman) → key evidence for/against → "
        "likely biases or fallacies → where uncertainty remains → calibrated conclusion. "
        "Separate facts from interpretation."
    ),
)
