"""LLM synthesis for deep-research merged bundles (Phase 4 slice 3)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from core.citation_renumber import renumber_citations_by_appearance
from core.knowledge.deep_research import build_bibliography_report
from core.knowledge.types import EvidenceBundle
from core.knowledge.ui_adapter import bundle_to_ui_sources
from core.skills.builtin.scientific_research import SCIENTIFIC_RESEARCH

DEEP_RESEARCH_CONTEXT_CHAR_BUDGET = 12000
DEEP_RESEARCH_SYNTHESIS_MAX_TOKENS = 1400


@dataclass(frozen=True)
class DeepResearchSynthesisResult:
    findings_markdown: str
    ui_sources: list[dict]
    synthesized: bool


def build_numbered_retrieval_context(
    ui_sources: list[dict],
    *,
    char_budget: int = DEEP_RESEARCH_CONTEXT_CHAR_BUDGET,
) -> str:
    """Format UI sources as numbered blocks for bracket citations."""
    blocks: list[str] = []
    for src in ui_sources:
        if not isinstance(src, dict):
            continue
        sid = src.get("id")
        title = str(src.get("filename") or src.get("title") or "Source").strip()
        header = f"--- [{sid}]: {title} ---"
        body_parts: list[str] = []
        for key in ("venue", "publication_date", "authors"):
            val = src.get(key)
            if val:
                if isinstance(val, list):
                    body_parts.append(", ".join(str(v) for v in val[:3]))
                else:
                    body_parts.append(str(val))
        content = str(src.get("content") or "").strip()
        if content:
            body_parts.append(content)
        block = header
        if body_parts:
            block = f"{block}\n" + "\n".join(body_parts)
        blocks.append(block)

    body = "\n\n".join(blocks)
    if char_budget > 0 and len(body) > char_budget:
        body = body[:char_budget].rsplit("\n\n", 1)[0]
    return body


def _strip_redundant_findings_heading(text: str) -> str:
    """Remove a leading Findings heading when the report wrapper adds ## Findings."""
    import re

    cleaned = (text or "").strip()
    cleaned = re.sub(r"^#+\s*Findings\s*:?\s*\n+", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def build_deep_research_system_prompt(*, coverage: str, has_conflicts: bool) -> str:
    skill = (SCIENTIFIC_RESEARCH.prompt_fragment or "").strip()
    hint = SCIENTIFIC_RESEARCH.retrieval_hint or ""
    if has_conflicts:
        hint = (
            f"{hint} Material disagreement detected — present both sides with caveats."
        )
    elif coverage in {"poor", "none"}:
        hint = (
            f"{hint} Coverage is limited — state uncertainty and avoid overconfident claims."
        )
    structure = skill if skill.lower().startswith("structure:") else f"Structure: {skill}"
    return (
        "You are Qube's deep-research synthesizer. Write a concise, evidence-grounded "
        "answer using ONLY the numbered source blocks provided.\n\n"
        f"Guidance: {hint}\n"
        f"{structure}\n\n"
        "Citation rules:\n"
        "- Place bracket citations at the END of each claim, e.g. One sentence. [1]\n"
        "- Match citation numbers to --- [N]: headers in the context.\n"
        "- Do not invent sources or cite ids not present in the context.\n"
        "- Do not repeat the bibliography; findings body only.\n"
        "- Do not include a top-level 'Findings' heading; optional ### subsections only."
    )


def build_deep_research_user_prompt(*, query: str, retrieval_context: str, bundle: EvidenceBundle) -> str:
    meta = (
        f"Coverage: {bundle.coverage} | Confidence: {bundle.confidence:.2f}\n"
        f"{bundle.coverage_rationale}"
    )
    if bundle.warnings:
        meta += f"\nWarnings: {', '.join(bundle.warnings)}"
    return (
        f"Research question:\n{query.strip()}\n\n"
        f"Evidence metadata:\n{meta}\n\n"
        "=== NUMBERED SOURCES ===\n"
        f"{retrieval_context.strip()}\n"
        "=== END SOURCES ===\n\n"
        "Write a cited findings body answering the research question. "
        "Do not include a 'Findings' title or heading."
    )


def synthesize_deep_research_findings(
    query: str,
    bundle: EvidenceBundle | None,
    *,
    generate_fn: Callable[..., str] | None = None,
) -> DeepResearchSynthesisResult:
    """Return cited findings markdown and cited-only UI sources."""
    if bundle is None or not bundle.sources:
        return DeepResearchSynthesisResult("", [], synthesized=False)

    ui_sources = bundle_to_ui_sources(bundle)
    for i, src in enumerate(ui_sources, start=1):
        src["id"] = i

    retrieval_context = build_numbered_retrieval_context(ui_sources)
    if not retrieval_context.strip() or generate_fn is None:
        return DeepResearchSynthesisResult("", ui_sources, synthesized=False)

    raw = generate_fn(
        system=build_deep_research_system_prompt(
            coverage=bundle.coverage,
            has_conflicts=bool(bundle.conflicts),
        ),
        user=build_deep_research_user_prompt(
            query=query,
            retrieval_context=retrieval_context,
            bundle=bundle,
        ),
    )
    findings = str(raw or "").strip()
    if not findings:
        return DeepResearchSynthesisResult("", ui_sources, synthesized=False)

    findings = _strip_redundant_findings_heading(findings)
    findings, cited_sources = renumber_citations_by_appearance(findings, ui_sources)
    return DeepResearchSynthesisResult(findings, cited_sources, synthesized=True)


def compose_deep_research_report(
    *,
    query: str,
    bundle: EvidenceBundle | None,
    sub_queries: tuple[str, ...],
    synthesis: DeepResearchSynthesisResult,
) -> str:
    """Stitch synthesis + bibliography into the user-facing report."""
    lines = [
        "# Deep Research Report",
        "",
        f"**Query:** {query.strip()}",
        "",
    ]
    if sub_queries and len(sub_queries) > 1:
        lines.extend(["## Sub-queries", ""])
        for idx, sq in enumerate(sub_queries, start=1):
            lines.append(f"{idx}. {sq}")
        lines.append("")

    if synthesis.findings_markdown.strip():
        lines.extend(["## Findings", "", synthesis.findings_markdown.strip(), ""])
    elif bundle is None or not bundle.sources:
        lines.extend(
            [
                "## Findings",
                "",
                "_No evidence sources were retained after retrieval and ranking._",
                "",
            ]
        )

    bib = build_bibliography_report(
        query=query,
        bundle=bundle,
        sub_queries=sub_queries,
        include_summary=False,
    )
    if bundle is not None and bundle.sources:
        lines.extend(
            [
                "## Evidence summary",
                "",
                bundle.coverage_rationale,
                "",
                f"Coverage: **{bundle.coverage}** · Confidence: **{bundle.confidence:.2f}**",
                "",
            ]
        )
    if "## Bibliography" in bib:
        tail = bib.split("## Bibliography", 1)[1].lstrip()
        lines.extend(["## Bibliography", "", tail.strip(), ""])
    else:
        lines.append(bib.strip())
    return "\n".join(lines).strip() + "\n"
