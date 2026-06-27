"""Scientific research skill (evidence-bundle aware)."""

from __future__ import annotations

from dataclasses import dataclass

from core.skills.base import BuiltinSkill
from core.skills.types import SkillContext

_EVIDENCE_BOOST = 0.12
_CONFLICT_BOOST = 0.06


def _scientific_boost(ctx: SkillContext) -> tuple[float, str | None]:
    if ctx.knowledge_service == "scientific_evidence" and ctx.has_retrieval_sources:
        return _EVIDENCE_BOOST, "boost:scientific_evidence"
    summary = ctx.evidence_summary
    if summary and summary.present and summary.knowledge_service == "scientific_evidence":
        return _EVIDENCE_BOOST, "boost:scientific_bundle"
    return 0.0, None


def _conflict_boost(ctx: SkillContext) -> tuple[float, str | None]:
    summary = ctx.evidence_summary
    if summary and summary.has_conflicts:
        return _CONFLICT_BOOST, "boost:material_conflict"
    return 0.0, None


@dataclass
class _ScientificResearchSkill(BuiltinSkill):
    def retrieval_framing_hint(self, ctx: SkillContext, score: float) -> str | None:
        if not ctx.has_retrieval_sources or not self.retrieval_hint:
            return None
        hint = self.retrieval_hint
        summary = ctx.evidence_summary
        if summary and summary.has_conflicts:
            hint = (
                f"{hint} Material disagreement detected — present both sides "
                "with caveats."
            )
        elif summary and summary.coverage in {"poor", "none"}:
            hint = (
                f"{hint} Coverage is limited — state uncertainty and avoid "
                "overconfident claims."
            )
        return hint


SCIENTIFIC_RESEARCH = _ScientificResearchSkill(
    id="scientific_research",
    name="Scientific research",
    description="Summarize abstracts with epistemic humility and conflict awareness.",
    version="1.0.0",
    priority=80,
    max_prompt_chars=450,
    activation_triggers=(
        "study",
        "studies",
        "research",
        "clinical trial",
        "meta-analysis",
        "systematic review",
        "peer-reviewed",
        "pubmed",
        "evidence",
        "efficacy",
        "side effect",
    ),
    context_boost_fns=(_scientific_boost, _conflict_boost),
    retrieval_hint=(
        "Treat abstracts as primary evidence; note preprints, single-study limits, "
        "and missing full text."
    ),
    prompt_fragment=(
        "Structure: Findings → Strength/limitations → Open questions. "
        "Place bracket citations at the end of each claim, not at the start. "
        "Distinguish correlation from causation."
    ),
)
