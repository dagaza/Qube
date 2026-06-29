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
        kept, raw_audit = search_library_chunks(
            query,
            query_vector,
            store,
            top_k=top_k,
            source_filter=ctx.source_filter,
        )

        latency_ms = (time.time() - t0) * 1000
        if not kept:
            return (
                build_empty_bundle(
                    query_raw=query,
                    query_resolved=semantic,
                    latency_ms=latency_ms,
                    knowledge_service=SERVICE_INTERNAL_CORPUS,
                ),
                {"library_results_kept_count": 0},
                raw_audit,
            )

        bundle = build_internal_corpus_bundle(
            query_raw=query,
            query_resolved=semantic,
            kept_rows=kept,
            rejected_count=0,
            latency_ms=latency_ms,
            adapter_calls=(ADAPTER_ID,),
        )
        rel_diag = {"library_results_kept_count": len(kept)}
        return bundle, rel_diag, raw_audit
