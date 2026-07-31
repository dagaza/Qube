"""Map EvidenceBundle objects to legacy UI source dicts and prompt text."""

from __future__ import annotations

import re

from core.integrations.capabilities.urn import CapabilityURN
from core.knowledge.types import EvidenceBundle, EvidenceObject


def _capability_urn_from_source_row(src: dict) -> str:
    """Resolve a cap: URN string from UI source row metadata (P8)."""
    cap = str(src.get("source_capability") or "").strip()
    if cap:
        return cap
    adapter = str(src.get("source_adapter") or "").strip()
    if adapter.startswith("cap:"):
        return adapter
    return ""


def source_type_label_for_row(src: dict) -> str:
    """Type line for CitationSourcesDialog — capability label when provenance exists."""
    cap = _capability_urn_from_source_row(src)
    if cap:
        urn = CapabilityURN.try_parse(cap)
        if urn is not None:
            return urn.display_label
        return cap
    adapter = str(src.get("source_adapter") or "").strip()
    if adapter:
        return adapter.replace("_", " ").title()
    st = str(src.get("type") or "").strip().lower()
    if st == "web":
        return "Web"
    if st == "memory":
        return "Memory"
    if st == "rag":
        return "Document"
    return st.title() if st else "Source"


def source_provenance_metadata_parts(src: dict) -> list[str]:
    """Extra metadata fragments when a row carries cap: provenance."""
    cap = _capability_urn_from_source_row(src)
    if not cap:
        return []
    parts: list[str] = []
    method = str(src.get("retrieval_method") or "").strip()
    if method:
        parts.append(method.replace("_", " "))
    body = cap[4:] if cap.startswith("cap:") else cap
    if body:
        parts.append(body)
    return parts


def evidence_to_ui_source(obj: EvidenceObject, *, ui_id: int) -> dict:
    """Produce a legacy ``all_ui_sources`` row (additive evidence fields)."""
    source_type = "library" if obj.document_type == "library_chunk" else "web"
    row: dict = {
        "id": ui_id,
        "filename": obj.title,
        "content": obj.excerpt,
        "type": source_type,
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
    if obj.entity_ids:
        row["entity_ids"] = list(obj.entity_ids)
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
    # Capability provenance (cap: URN) for INSPECT / Sources, when the hit came
    # from a capability provider (e.g. MCP). Preserved end-to-end (P8, KI1).
    capability = (obj.raw_metadata or {}).get("capability")
    if capability:
        row["source_capability"] = str(capability)
        row["retrieval_method"] = obj.retrieval_method
    return row


def bundle_to_ui_sources(bundle: EvidenceBundle) -> list[dict]:
    return [
        evidence_to_ui_source(obj, ui_id=i)
        for i, obj in enumerate(bundle.sources, start=1)
    ]


def append_turn_evidence_bundle_sources(
    all_ui_sources: list[dict],
    bundle: EvidenceBundle | None,
) -> None:
    """Append UI rows from a turn :class:`EvidenceBundle` (P8 / KI1 main path).

    Citation ``id`` values are provisional; :meth:`LLMWorker._apply_sequential_source_ids`
    renumbers the merged list so memory, RAG, and web/capability rows stay unique.
    """
    if bundle is None or not bundle.sources:
        return
    all_ui_sources.extend(bundle_to_ui_sources(bundle))


def _truncate_at_boundary(text: str, char_budget: int) -> str:
    """Trim text to char_budget without mid-word chops when possible."""
    text = (text or "").strip()
    if char_budget <= 0:
        return ""
    if len(text) <= char_budget:
        return text

    truncated = text[:char_budget]
    for separator in ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "):
        head = truncated.rsplit(separator, 1)[0].strip()
        if head and len(head) <= char_budget:
            return head

    return re.sub(r"\w+$", "", truncated).rstrip() or truncated.rstrip()


def _fit_source_blocks_to_budget(blocks: list[str], char_budget: int) -> str:
    if char_budget <= 0:
        return "\n\n".join(blocks)

    kept: list[str] = []
    for block in blocks:
        candidate = "\n\n".join(kept + [block]) if kept else block
        if len(candidate) <= char_budget:
            kept.append(block)
            continue

        remaining = char_budget - (len("\n\n".join(kept)) + (2 if kept else 0))
        if remaining > 40:
            trimmed = _truncate_at_boundary(block, remaining)
            if trimmed:
                kept.append(trimmed)
        break

    return "\n\n".join(kept)


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

    meta_block = "\n".join(meta_lines)
    source_blocks: list[str] = []
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
            source_blocks.append(block)

    header_prefix = f"{header}:\n"
    if char_budget > 0:
        body_budget = max(0, char_budget - len(header_prefix))
        meta_allowance = min(len(meta_block) + 2, max(0, body_budget // 4))
        content_budget = max(0, body_budget - meta_allowance)
        body = _fit_source_blocks_to_budget(source_blocks, content_budget)
        parts = [meta_block, body] if body else [meta_block]
        body = "\n\n".join(part for part in parts if part)
        if len(body) > body_budget:
            body = _truncate_at_boundary(body, body_budget)
    else:
        body = "\n\n".join([meta_block, *source_blocks])

    if not body:
        return ""
    result = f"{header_prefix}{body}"
    if char_budget > 0 and len(result) > char_budget:
        result = _truncate_at_boundary(result, char_budget)
    return result
