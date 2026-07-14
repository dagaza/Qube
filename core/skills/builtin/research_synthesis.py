"""Research synthesis skill (non-web-specific reasoning)."""

from __future__ import annotations

from dataclasses import dataclass

from core.skills.base import BuiltinSkill
from core.skills.types import SkillContext

_SOURCES_BOOST = 0.08
_TRUSTED_BOOST = 0.1


def _sources_boost(ctx: SkillContext) -> tuple[float, str | None]:
    if ctx.has_retrieval_sources:
        return _SOURCES_BOOST, "boost:has_sources"
    return 0.0, None


def _trusted_boost(ctx: SkillContext) -> tuple[float, str | None]:
    if ctx.knowledge_service == "trusted_knowledge" and ctx.has_retrieval_sources:
        return _TRUSTED_BOOST, "boost:trusted_knowledge"
    return 0.0, None


@dataclass
class _ResearchSynthesisSkill(BuiltinSkill):
    def retrieval_framing_hint(self, ctx: SkillContext, score: float) -> str | None:
        if not ctx.has_retrieval_sources or not self.retrieval_hint:
            return None
        hint = self.retrieval_hint
        if ctx.knowledge_service == "trusted_knowledge":
            hint = (
                f"{hint} Sources are Wikipedia or allowlisted domains; "
                "attribute claims and note thin corroboration."
            )
        return hint


RESEARCH_SYNTHESIS = _ResearchSynthesisSkill(
    id="research_synthesis",
    name="Research synthesis",
    description="Synthesize provided sources with epistemic humility.",
    version="1.1.0",
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
    context_boost_fns=(_sources_boost, _trusted_boost),
    retrieval_hint=(
        "Treat numbered sources as evidence; attribute claims; note disagreements."
    ),
    prompt_fragment=(
        "Use only provided context. Structure: Key findings → Agreements/conflicts → "
        "Open gaps. Cite with existing bracket tokens."
    ),
)
