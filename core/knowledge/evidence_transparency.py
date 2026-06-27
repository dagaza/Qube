"""User-facing transparency summaries for evidence bundles."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from core.knowledge.types import EvidenceBundle, EvidenceConflict


def _conflict_lines(conflicts: Sequence[EvidenceConflict]) -> list[str]:
    lines: list[str] = []
    for conflict in conflicts or ():
        lines.append(f"- {conflict.topic} ({conflict.severity})")
    return lines


def build_evidence_transparency(
    bundle: EvidenceBundle | None,
    *,
    diagnostics: Mapping[str, Any] | None = None,
    sub_queries: Sequence[str] = (),
) -> dict[str, Any]:
    """Structured + prose summary explaining why sources were retained."""
    if bundle is None:
        return {}

    diag = dict(diagnostics or {})
    adapters = sorted({s.adapter for s in bundle.sources})
    abstract_count = sum(1 for s in bundle.sources if s.fetch_status == "abstract")

    why_lines = [
        f"Query: {bundle.query_resolved}",
        f"Coverage: {bundle.coverage} — {bundle.coverage_rationale}",
        f"Confidence: {bundle.confidence:.2f}",
        f"Retrieval: {bundle.retrieval_strategy} via {', '.join(adapters) or 'none'}",
        f"Sources kept: {len(bundle.sources)} ({abstract_count} with abstracts)",
    ]
    if sub_queries and len(sub_queries) > 1:
        why_lines.append("Sub-queries:")
        why_lines.extend(f"  {idx}. {sq}" for idx, sq in enumerate(sub_queries, start=1))

    pre = diag.get("merged_sources_pre_filter")
    post = diag.get("merged_sources_post_filter")
    dropped = diag.get("merged_relevance_dropped")
    if pre is not None and post is not None:
        why_lines.append(
            f"Merge filter: {pre} merged → {post} retained"
            + (f" ({dropped} dropped)" if dropped else "")
        )
    anchor_dropped = diag.get("merged_anchor_dropped")
    if anchor_dropped:
        anchors = diag.get("merged_anchor_tokens") or []
        anchor_text = ", ".join(str(a) for a in anchors[:6])
        why_lines.append(
            f"Anchor filter removed {anchor_dropped} off-topic hit(s)"
            + (f" (required: {anchor_text})" if anchor_text else "")
        )

    if bundle.warnings:
        why_lines.append(f"Warnings: {', '.join(bundle.warnings)}")
    conflict_lines = _conflict_lines(bundle.conflicts)
    if conflict_lines:
        why_lines.append("Conflicts noted:")
        why_lines.extend(f"  {line}" for line in conflict_lines)

    return {
        "query": bundle.query_resolved,
        "knowledge_service": bundle.knowledge_service,
        "retrieval_strategy": bundle.retrieval_strategy,
        "coverage": bundle.coverage,
        "coverage_rationale": bundle.coverage_rationale,
        "confidence": round(bundle.confidence, 4),
        "adapter_calls": list(bundle.adapter_calls),
        "source_count": len(bundle.sources),
        "abstract_count": abstract_count,
        "warnings": list(bundle.warnings),
        "conflicts": [
            {"topic": c.topic, "severity": c.severity} for c in bundle.conflicts
        ],
        "sub_queries": list(sub_queries),
        "filter_diag": {
            k: diag[k]
            for k in (
                "merged_relevance_dropped",
                "merged_sources_pre_filter",
                "merged_sources_post_filter",
                "merged_anchor_dropped",
                "merged_anchor_tokens",
                "merged_semantic_gate",
            )
            if k in diag
        },
        "why_summary": "\n".join(why_lines),
    }
