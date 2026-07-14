"""Retrieval observability — JSONL traces (schema v2, observer-only)."""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from core.knowledge.types import EvidenceBundle, EvidenceObject
from core.web_search_audit import web_search_audit_log_enabled
from core.web_search_audit_sink import WEB_SEARCH_AUDIT_LOGGER_NAME

RETRIEVAL_TRACE_SCHEMA_VERSION = 3
RETRIEVAL_TRACE_EVENT = "retrieval_trace"


@dataclass(frozen=True)
class RetrievalTrace:
    schema_version: int
    ts: float
    request_id: str
    bundle_id: str
    query_raw: str
    query_resolved: str
    knowledge_service: str
    retrieval_strategy: str
    adapter_calls: tuple[str, ...]
    candidates_raw_count: int
    candidates_rejected_count: int
    evidence_ids_kept: tuple[str, ...]
    confidence: float
    coverage: str
    coverage_rationale: str
    stop_reason: str
    latency_ms: float
    warnings: tuple[str, ...]
    relevance_diag: Mapping[str, Any] | None = None
    session_id: str | None = None
    turn_id: int | None = None
    preset_id: str | None = None
    retrieval_profile: str | None = None
    context_fingerprint: Mapping[str, Any] | None = None
    pipeline_stages: tuple[Mapping[str, Any], ...] = ()


def _serialize_evidence(obj: EvidenceObject) -> dict[str, Any]:
    return {
        "id": obj.id,
        "adapter": obj.adapter,
        "title": obj.title[:120],
        "url": obj.url,
        "document_type": obj.document_type,
        "relevance_score": round(obj.relevance_score, 4),
        "authority_score": round(obj.authority_score, 4),
        "fetch_status": obj.fetch_status,
    }


def build_retrieval_trace(
    bundle: EvidenceBundle,
    *,
    relevance_diag: Mapping[str, Any] | None = None,
    session_id: str | None = None,
    turn_id: int | None = None,
    request_id: str | None = None,
    ts: float | None = None,
    preset_id: str | None = None,
    retrieval_profile: str | None = None,
    context_fingerprint: Mapping[str, Any] | None = None,
    pipeline_stages: Sequence[Mapping[str, Any]] | None = None,
) -> RetrievalTrace:
    raw_count = 0
    if relevance_diag is not None:
        try:
            raw_count = int(relevance_diag.get("web_results_raw_count") or 0)
        except (TypeError, ValueError):
            raw_count = 0
    if raw_count <= 0:
        raw_count = len(bundle.sources) + bundle.rejected_count

    return RetrievalTrace(
        schema_version=RETRIEVAL_TRACE_SCHEMA_VERSION,
        ts=float(ts if ts is not None else time.time()),
        request_id=str(request_id or uuid.uuid4()),
        bundle_id=bundle.bundle_id,
        query_raw=bundle.query_raw,
        query_resolved=bundle.query_resolved,
        knowledge_service=bundle.knowledge_service,
        retrieval_strategy=bundle.retrieval_strategy,
        adapter_calls=bundle.adapter_calls,
        candidates_raw_count=raw_count,
        candidates_rejected_count=bundle.rejected_count,
        evidence_ids_kept=tuple(s.id for s in bundle.sources),
        confidence=bundle.confidence,
        coverage=bundle.coverage,
        coverage_rationale=bundle.coverage_rationale,
        stop_reason=bundle.stop_reason,
        latency_ms=bundle.latency_ms,
        warnings=bundle.warnings,
        relevance_diag=relevance_diag,
        session_id=session_id,
        turn_id=turn_id,
        preset_id=preset_id,
        retrieval_profile=retrieval_profile,
        context_fingerprint=context_fingerprint,
        pipeline_stages=tuple(pipeline_stages or ()),
    )


def serialize_retrieval_trace(
    trace: RetrievalTrace,
    *,
    sources: Sequence[EvidenceObject] = (),
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": trace.schema_version,
        "event": RETRIEVAL_TRACE_EVENT,
        "ts": trace.ts,
        "request_id": trace.request_id,
        "bundle_id": trace.bundle_id,
        "session_id": trace.session_id,
        "turn_id": trace.turn_id,
        "query_raw": trace.query_raw,
        "query_resolved": trace.query_resolved,
        "knowledge_service": trace.knowledge_service,
        "retrieval_strategy": trace.retrieval_strategy,
        "adapter_calls": list(trace.adapter_calls),
        "candidates_raw_count": trace.candidates_raw_count,
        "candidates_rejected_count": trace.candidates_rejected_count,
        "evidence_ids_kept": list(trace.evidence_ids_kept),
        "confidence": round(trace.confidence, 4),
        "coverage": trace.coverage,
        "coverage_rationale": trace.coverage_rationale,
        "stop_reason": trace.stop_reason,
        "latency_ms": trace.latency_ms,
        "warnings": list(trace.warnings),
        "sources": [_serialize_evidence(s) for s in sources],
    }
    if trace.relevance_diag:
        payload["relevance_diag"] = dict(trace.relevance_diag)
    if trace.preset_id:
        payload["preset_id"] = trace.preset_id
    if trace.retrieval_profile:
        payload["retrieval_profile"] = trace.retrieval_profile
    if trace.context_fingerprint:
        payload["context_fingerprint"] = dict(trace.context_fingerprint)
    if trace.pipeline_stages:
        payload["pipeline_stages"] = [dict(s) for s in trace.pipeline_stages]
    return payload


def record_retrieval_trace(
    trace: RetrievalTrace,
    *,
    sources: Sequence[EvidenceObject] = (),
) -> None:
    """Append one JSONL line when web search audit logging is enabled."""
    if not web_search_audit_log_enabled():
        return
    try:
        payload = serialize_retrieval_trace(trace, sources=sources)
        logging.getLogger(WEB_SEARCH_AUDIT_LOGGER_NAME).info(
            json.dumps(payload, ensure_ascii=False, default=str)
        )
    except Exception:
        pass
