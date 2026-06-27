"""Map EvidenceBundle objects to legacy UI source dicts and prompt text."""

from __future__ import annotations

from core.knowledge.types import EvidenceBundle, EvidenceObject


def evidence_to_ui_source(obj: EvidenceObject, *, ui_id: int) -> dict:
    """Produce a legacy ``all_ui_sources`` row (additive evidence fields)."""
    row: dict = {
        "id": ui_id,
        "filename": obj.title,
        "content": obj.excerpt,
        "type": "web",
        "evidence_id": obj.id,
        "source_adapter": obj.adapter,
        "document_type": obj.document_type,
        "fetch_status": obj.fetch_status,
        "relevance_score": round(float(obj.relevance_score), 4),
        "authority_score": round(float(obj.authority_score), 4),
    }
    if obj.url:
        row["url"] = obj.url
    if obj.doi:
        row["doi"] = obj.doi
    if obj.venue:
        row["venue"] = obj.venue
    if obj.authors:
        row["authors"] = list(obj.authors)
    if obj.publication_date:
        row["publication_date"] = obj.publication_date
    if obj.preprint:
        row["preprint"] = True
    if obj.peer_reviewed is not None:
        row["peer_reviewed"] = bool(obj.peer_reviewed)
    return row


def bundle_to_ui_sources(bundle: EvidenceBundle) -> list[dict]:
    return [
        evidence_to_ui_source(obj, ui_id=i)
        for i, obj in enumerate(bundle.sources, start=1)
    ]


def bundle_to_prompt_context(
    bundle: EvidenceBundle,
    *,
    char_budget: int,
    header: str = "WEB SEARCH RESULTS",
) -> str:
    """Format bundle excerpts for LLM tool context (legacy header preserved)."""
    if not bundle.sources:
        return ""

    meta_lines = [
        f"Service: {bundle.knowledge_service}",
        f"Coverage: {bundle.coverage} | Confidence: {bundle.confidence:.2f}",
    ]
    if bundle.warnings:
        meta_lines.append(f"Warnings: {', '.join(bundle.warnings)}")

    parts: list[str] = ["\n".join(meta_lines), ""]
    for obj in bundle.sources:
        block = obj.title
        meta_bits: list[str] = []
        if obj.venue:
            meta_bits.append(str(obj.venue))
        if obj.publication_date:
            meta_bits.append(str(obj.publication_date))
        if obj.authors:
            meta_bits.append(", ".join(obj.authors[:3]))
        if meta_bits:
            block = f"{block}\n({' · '.join(meta_bits)})"
        if obj.excerpt:
            block = f"{block}\n{obj.excerpt}".strip()
        if block:
            parts.append(block)

    body = "\n\n".join(p for p in parts if p)
    if char_budget > 0:
        body = body[:char_budget]
    return f"{header}:\n{body}" if body else ""
