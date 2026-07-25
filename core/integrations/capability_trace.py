"""Capability retrieval trace helpers (Phase 4 / #62).

Records INSPECT capability steps for allowed, denied, and post-citation turns.
Provider-agnostic (P6); complements ``capability_inspect`` builders.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from core.integrations.capability_inspect import build_cited_step
from core.knowledge.bundle_builder import build_generic_bundle
from core.knowledge.observability import (
    RetrievalTrace,
    build_retrieval_trace,
    record_retrieval_trace,
)
from core.knowledge.retrieval_records import (
    RetrievalContextFingerprint,
    save_retrieval_record,
)
from core.knowledge.types import EvidenceBundle

__all__ = [
    "CapabilityTraceContext",
    "append_cited_step_to_trace",
    "build_capability_denial_bundle",
    "extract_citation_ids_from_text",
    "finalize_capability_cited_trace",
    "record_capability_retrieval_trace",
]

_CITATION_RE = re.compile(r"\[(\d+)\]")


def extract_citation_ids_from_text(text: str) -> list[str]:
    """Return unique bracket citation ids in first-appearance order."""
    seen: set[str] = set()
    out: list[str] = []
    for match in _CITATION_RE.finditer(text or ""):
        cid = match.group(1)
        if cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


def append_cited_step_to_trace(
    steps: Sequence[Mapping[str, Any]],
    *,
    cited_ids: Sequence[str],
    rows: Sequence[Mapping[str, Any]] | None = None,
    default_urn: str = "",
) -> list[dict[str, Any]]:
    """Append a cited step when citations exist and one is not already present."""
    if not cited_ids:
        return [dict(step) for step in steps]
    merged = [dict(step) for step in steps]
    if any(step.get("kind") == "cited" for step in merged):
        return merged
    caps = sorted(
        {
            str(row.get("_capability") or row.get("source_capability") or default_urn)
            for row in (rows or ())
            if row.get("_capability") or row.get("source_capability") or default_urn
        }
    )
    merged.append(
        build_cited_step(
            cited_ids=[str(item) for item in cited_ids if str(item).strip()],
            source_capabilities=caps or None,
            step=len(merged) + 1,
        )
    )
    return merged


def build_capability_denial_bundle(
    *,
    query_raw: str,
    query_resolved: str,
    latency_ms: float,
    preset_id: str | None = None,
    stop_reason: str = "capability_denied",
) -> EvidenceBundle:
    """Minimal bundle for denied/empty capability invokes (INSPECT trace only)."""
    return build_generic_bundle(
        query_raw=query_raw,
        query_resolved=query_resolved,
        kept_rows=[],
        rejected_count=0,
        latency_ms=latency_ms,
        knowledge_service="capability",
        retrieval_strategy="attachment_capability",
        stop_reason=stop_reason,
        preset_id=preset_id,
    )


@dataclass
class CapabilityTraceContext:
    """Mutable trace state for one CAPABILITY turn."""

    cap_steps: list[dict[str, Any]]
    query_raw: str
    query_resolved: str
    latency_ms: float
    preset_id: str
    cap_urn: str
    session_id: str | None
    turn_id: int | None
    kept_rows: list[dict[str, Any]]
    bundle: EvidenceBundle | None = None
    trace: RetrievalTrace | None = None
    fingerprint: RetrievalContextFingerprint | None = None


def record_capability_retrieval_trace(
    ctx: CapabilityTraceContext,
    *,
    db,
    retrieval_profile: str | None = None,
) -> None:
    """Persist retrieval trace + record for allowed, denied, or empty capability turns."""
    bundle = ctx.bundle
    if bundle is None:
        bundle = build_capability_denial_bundle(
            query_raw=ctx.query_raw,
            query_resolved=ctx.query_resolved,
            latency_ms=ctx.latency_ms,
            preset_id=ctx.preset_id or None,
            stop_reason="capability_denied"
            if not ctx.kept_rows
            else "capability_empty",
        )

    adapter_filter = tuple(
        sorted(
            {
                str(row.get("_adapter") or "")
                for row in ctx.kept_rows
                if row.get("_adapter")
            }
        )
    )
    fingerprint = ctx.fingerprint or RetrievalContextFingerprint(
        query_raw=ctx.query_raw,
        query_resolved=ctx.query_resolved,
        knowledge_service="capability",
        preset_id=ctx.preset_id or None,
        adapter_filter=adapter_filter,
        retrieval_profile=retrieval_profile or "",
        connector_config_hashes=(),
    )

    trace = build_retrieval_trace(
        bundle,
        session_id=ctx.session_id,
        turn_id=ctx.turn_id,
        retrieval_profile=retrieval_profile,
        context_fingerprint=fingerprint.to_dict(),
        capability_steps=ctx.cap_steps,
    )
    ctx.bundle = bundle
    ctx.fingerprint = fingerprint
    ctx.trace = trace

    record_retrieval_trace(trace, sources=bundle.sources)
    if db is not None:
        save_retrieval_record(
            db,
            request_id=trace.request_id,
            bundle=bundle,
            context_fingerprint=fingerprint,
            session_id=ctx.session_id,
            turn_id=ctx.turn_id,
        )


def finalize_capability_cited_trace(
    ctx: CapabilityTraceContext | None,
    *,
    final_text: str,
    all_ui_sources: Sequence[Mapping[str, Any]],
    db,
    retrieval_profile: str | None = None,
) -> list[dict[str, Any]] | None:
    """Append cited INSPECT step post-answer and re-record the trace."""
    if ctx is None or not ctx.cap_steps:
        return None
    cited_ids = extract_citation_ids_from_text(final_text)
    if not cited_ids:
        return ctx.cap_steps

    rows = list(ctx.kept_rows)
    for source in all_ui_sources or ():
        if not isinstance(source, Mapping):
            continue
        cap = source.get("source_capability")
        if cap:
            rows.append({"_capability": cap, "source_capability": cap})

    updated = append_cited_step_to_trace(
        ctx.cap_steps,
        cited_ids=cited_ids,
        rows=rows,
        default_urn=ctx.cap_urn,
    )
    if updated == ctx.cap_steps:
        return ctx.cap_steps

    ctx.cap_steps = updated
    if ctx.trace is not None and ctx.bundle is not None:
        from dataclasses import replace

        ctx.trace = replace(ctx.trace, capability_steps=tuple(updated))
        record_retrieval_trace(ctx.trace, sources=ctx.bundle.sources)
        if db is not None and ctx.fingerprint is not None:
            save_retrieval_record(
                db,
                request_id=ctx.trace.request_id,
                bundle=ctx.bundle,
                context_fingerprint=ctx.fingerprint,
                session_id=ctx.session_id,
                turn_id=ctx.turn_id,
            )
    return updated
