"""Research synthesis skill (non-web-specific reasoning)."""

from __future__ import annotations

from core.skills.base import BuiltinSkill
from core.skills.types import SkillContext

_SOURCES_BOOST = 0.08


def _sources_boost(ctx: SkillContext) -> tuple[float, str | None]:
    if ctx.has_retrieval_sources:
        return _SOURCES_BOOST, "boost:has_sources"
    return 0.0, None


RESEARCH_SYNTHESIS = BuiltinSkill(
    id="research_synthesis",
    name="Research synthesis",
    description="Synthesize provided sources with epistemic humility.",
    version="1.0.0",
    priority=75,
    max_prompt_chars=400,
    activation_triggers=(
        "summarize",
        "compare",
        "synthesize",
        "according to",
        "key findings",
        "main points",
        "contrast",
        "agree",
        "disagree",
    ),
    context_boost_fns=(_sources_boost,),
    retrieval_hint=(
        "Treat numbered sources as evidence; attribute claims; note disagreements."
    ),
    prompt_fragment=(
        "Use only provided context. Structure: Key findings → Agreements/conflicts → "
        "Open gaps. Cite with existing bracket tokens."
    ),
)
