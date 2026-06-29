"""Canonical JSON encoding for EvidenceBundle (graph golden tests + snapshots)."""

from __future__ import annotations

from typing import Any

from core.knowledge.types import EvidenceBundle, EvidenceConflict, EvidenceObject


def _evidence_to_dict(obj: EvidenceObject) -> dict[str, Any]:
    return {
        "id": obj.id,
        "source_id": obj.source_id,
        "adapter": obj.adapter,
        "retrieval_method": obj.retrieval_method,
        "title": obj.title,
        "excerpt": obj.excerpt,
        "full_text": obj.full_text,
        "url": obj.url,
        "document_type": obj.document_type,
        "publication_date": obj.publication_date,
        "venue": obj.venue,
        "authors": list(obj.authors),
        "doi": obj.doi,
        "peer_reviewed": obj.peer_reviewed,
        "preprint": obj.preprint,
        "open_access": obj.open_access,
        "retracted": obj.retracted,
        "relevance_score": round(float(obj.relevance_score), 6),
        "authority_score": round(float(obj.authority_score), 6),
        "reliability_score": round(float(obj.reliability_score), 6),
        "freshness_score": (
            round(float(obj.freshness_score), 6)
            if obj.freshness_score is not None
            else None
        ),
        "retrieved_at": round(float(obj.retrieved_at), 6),
        "fetch_status": obj.fetch_status,
        "raw_metadata": dict(obj.raw_metadata or {}),
        "entity_ids": list(obj.entity_ids),
    }


def _conflict_to_dict(conflict: EvidenceConflict) -> dict[str, Any]:
    return {
        "topic": conflict.topic,
        "positions": [
            {"stance": stance, "label": label}
            for stance, label in conflict.positions
        ],
        "severity": conflict.severity,
    }


def bundle_to_dict(bundle: EvidenceBundle) -> dict[str, Any]:
    """Deterministic bundle JSON suitable for graph golden tests."""
    sources = sorted(bundle.sources, key=lambda s: s.id)
    conflicts = sorted(bundle.conflicts, key=lambda c: c.topic)
    return {
        "bundle_id": bundle.bundle_id,
        "query_raw": bundle.query_raw,
        "query_resolved": bundle.query_resolved,
        "knowledge_service": bundle.knowledge_service,
        "retrieval_strategy": bundle.retrieval_strategy,
        "profile_version": bundle.profile_version,
        "retrieved_at": round(float(bundle.retrieved_at), 6),
        "latency_ms": round(float(bundle.latency_ms), 3),
        "confidence": round(float(bundle.confidence), 6),
        "coverage": bundle.coverage,
        "coverage_rationale": bundle.coverage_rationale,
        "authority_summary": round(float(bundle.authority_summary), 6),
        "reliability_summary": round(float(bundle.reliability_summary), 6),
        "diversity_summary": round(float(bundle.diversity_summary), 6),
        "sources": [_evidence_to_dict(s) for s in sources],
        "rejected_count": int(bundle.rejected_count),
        "warnings": list(bundle.warnings),
        "conflicts": [_conflict_to_dict(c) for c in conflicts],
        "stop_reason": bundle.stop_reason,
        "adapter_calls": list(bundle.adapter_calls),
    }


def _evidence_from_dict(row: dict[str, Any]) -> EvidenceObject:
    return EvidenceObject(
        id=str(row["id"]),
        source_id=str(row["source_id"]),
        adapter=str(row["adapter"]),
        retrieval_method=str(row["retrieval_method"]),
        title=str(row.get("title") or ""),
        excerpt=str(row.get("excerpt") or ""),
        full_text=row.get("full_text"),
        url=row.get("url"),
        document_type=str(row.get("document_type") or ""),
        publication_date=row.get("publication_date"),
        venue=row.get("venue"),
        authors=tuple(str(a) for a in (row.get("authors") or [])),
        doi=row.get("doi"),
        peer_reviewed=row.get("peer_reviewed"),
        preprint=row.get("preprint"),
        open_access=row.get("open_access"),
        retracted=row.get("retracted"),
        relevance_score=float(row.get("relevance_score") or 0.0),
        authority_score=float(row.get("authority_score") or 0.0),
        reliability_score=float(row.get("reliability_score") or 0.0),
        freshness_score=row.get("freshness_score"),
        retrieved_at=float(row.get("retrieved_at") or 0.0),
        fetch_status=str(row.get("fetch_status") or "snippet_only"),
        raw_metadata=dict(row.get("raw_metadata") or {}),
        entity_ids=tuple(str(e) for e in (row.get("entity_ids") or [])),
    )


def _conflict_from_dict(row: dict[str, Any]) -> EvidenceConflict:
    positions: list[tuple[str, str]] = []
    for pos in row.get("positions") or []:
        if isinstance(pos, dict):
            positions.append((str(pos.get("stance") or ""), str(pos.get("label") or "")))
        elif isinstance(pos, (list, tuple)) and len(pos) >= 2:
            positions.append((str(pos[0]), str(pos[1])))
    return EvidenceConflict(
        topic=str(row.get("topic") or ""),
        positions=tuple(positions),
        severity=str(row.get("severity") or "minor"),
    )


def bundle_from_dict(data: dict[str, Any]) -> EvidenceBundle:
    sources = tuple(
        _evidence_from_dict(row)
        for row in sorted(data.get("sources") or [], key=lambda r: str(r.get("id") or ""))
    )
    conflicts = tuple(
        _conflict_from_dict(row)
        for row in sorted(data.get("conflicts") or [], key=lambda r: str(r.get("topic") or ""))
    )
    return EvidenceBundle(
        bundle_id=str(data["bundle_id"]),
        query_raw=str(data.get("query_raw") or ""),
        query_resolved=str(data.get("query_resolved") or ""),
        knowledge_service=str(data.get("knowledge_service") or ""),
        retrieval_strategy=str(data.get("retrieval_strategy") or ""),
        profile_version=str(data.get("profile_version") or ""),
        retrieved_at=float(data.get("retrieved_at") or 0.0),
        latency_ms=float(data.get("latency_ms") or 0.0),
        confidence=float(data.get("confidence") or 0.0),
        coverage=str(data.get("coverage") or "none"),
        coverage_rationale=str(data.get("coverage_rationale") or ""),
        authority_summary=float(data.get("authority_summary") or 0.0),
        reliability_summary=float(data.get("reliability_summary") or 0.0),
        diversity_summary=float(data.get("diversity_summary") or 0.0),
        sources=sources,
        rejected_count=int(data.get("rejected_count") or 0),
        warnings=tuple(str(w) for w in (data.get("warnings") or [])),
        conflicts=conflicts,
        stop_reason=str(data.get("stop_reason") or "no_evidence"),
        adapter_calls=tuple(str(a) for a in (data.get("adapter_calls") or [])),
    )
