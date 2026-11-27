"""Internal corpus pipeline: LanceDB library → evidence bundle."""

from __future__ import annotations

import time
from typing import Any

from core.knowledge.adapters.lancedb_library import ADAPTER_ID, search_library_chunks
from core.knowledge.bundle_builder import build_empty_bundle, build_internal_corpus_bundle
from core.knowledge.types import EvidenceBundle, RetrievalContext, SERVICE_INTERNAL_CORPUS
from core.reindex_state import is_reindex_in_progress


class InternalCorpusEvidencePipeline:
    """Library hybrid search packaged as an external-knowledge evidence bundle."""

    def run(
        self, ctx: RetrievalContext
    ) -> tuple[EvidenceBundle, dict[str, Any] | None, list[dict[str, Any]]]:
        t0 = time.time()
        query = ctx.query
        semantic = ctx.semantic_query or query
        store = ctx.library_store

        if store is None:
            latency_ms = (time.time() - t0) * 1000
            return (
                build_empty_bundle(
                    query_raw=query,
                    query_resolved=semantic,
                    latency_ms=latency_ms,
                    stop_reason="no_library_store",
                    knowledge_service=SERVICE_INTERNAL_CORPUS,
                ),
                None,
                [],
            )

        if is_reindex_in_progress():
            latency_ms = (time.time() - t0) * 1000
            return (
                build_empty_bundle(
                    query_raw=query,
                    query_resolved=semantic,
                    latency_ms=latency_ms,
                    stop_reason="reindex_in_progress",
                    knowledge_service=SERVICE_INTERNAL_CORPUS,
                ),
                None,
                [],
            )

        query_vector = ctx.query_vector
        if query_vector is None and ctx.embed_fn:
            try:
                query_vector = ctx.embed_fn(semantic)
            except Exception:
                query_vector = None
        if query_vector is None:
            latency_ms = (time.time() - t0) * 1000
            return (
                build_empty_bundle(
                    query_raw=query,
                    query_resolved=semantic,
                    latency_ms=latency_ms,
                    stop_reason="no_query_vector",
                    knowledge_service=SERVICE_INTERNAL_CORPUS,
                ),
                None,
                [],
            )

        top_k = max(1, ctx.budget.max_results)
        gate_out: dict[str, Any] = {}
        kept, raw_audit = search_library_chunks(
            query,
            query_vector,
            store,
            top_k=top_k,
            source_filter=ctx.source_filter,
            source_prefix_filter=ctx.source_prefix_filter,
            gate_out=gate_out,
        )

        latency_ms = (time.time() - t0) * 1000
        rel_diag, sufficiency = _gate_diag(len(kept), gate_out)
        if not kept:
            insufficient = sufficiency == "insufficient"
            return (
                build_empty_bundle(
                    query_raw=query,
                    query_resolved=semantic,
                    latency_ms=latency_ms,
                    rejected_count=int((rel_diag.get("rerank") or {}).get("dropped") or 0),
                    stop_reason="relevance_filtered" if insufficient else "no_evidence",
                    knowledge_service=SERVICE_INTERNAL_CORPUS,
                    sufficiency=sufficiency,
                    sufficiency_rationale=_rationale(rel_diag) if insufficient else "",
                ),
                rel_diag,
                raw_audit,
            )

        bundle = build_internal_corpus_bundle(
            query_raw=query,
            query_resolved=semantic,
            kept_rows=kept,
            rejected_count=int((rel_diag.get("rerank") or {}).get("dropped") or 0),
            latency_ms=latency_ms,
            adapter_calls=(ADAPTER_ID,),
            sufficiency=sufficiency,
        )
        return bundle, rel_diag, raw_audit


def _gate_diag(kept_count: int, gate_out: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """``relevance_diag`` and the bundle state from the library gate (plan S35).

    The inspector reads ``rerank`` and ``rerank_dropped`` (D2) as it does for
    the web gate. Without a gate the state stays ``unscored``.
    """
    rel_diag: dict[str, Any] = {"library_results_kept_count": kept_count}
    gate = dict(gate_out.get("relevance_gate") or {})
    state = str(gate_out.get("sufficiency") or "unscored")
    if gate:
        dropped = gate.pop("rerank_dropped", None)
        rel_diag["rerank"] = gate
        rel_diag["sufficiency"] = state
        rel_diag["library_relevance_gate_mode"] = "rerank"
        if dropped:
            rel_diag["rerank_dropped"] = dropped
    # An empty pool ("none") keeps the pre-S35 empty bundle.
    if state not in ("sufficient", "partial", "insufficient"):
        state = "unscored"
    return rel_diag, state


def _rationale(rel_diag: dict[str, Any]) -> str:
    gate = rel_diag.get("rerank") or {}
    top = (gate.get("top_scores") or [None])[0]
    best = f"best {float(top):.2f}" if isinstance(top, (int, float)) else "no score"
    return (
        f"{gate.get('scored', 0)} passage(s) found in the library; none reached the relevance "
        f"threshold ({best} < {float(gate.get('threshold_keep', 0.0)):.2f})."
    )
