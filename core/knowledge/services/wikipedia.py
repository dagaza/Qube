"""Wikipedia-only knowledge service (advanced composer token)."""

from __future__ import annotations

import time

from core.knowledge.adapters.wikipedia_api import ADAPTER_ID as WIKI_ADAPTER
from core.knowledge.adapters.wikipedia_api import search_wikipedia
from core.knowledge.bundle_builder import build_empty_bundle, build_trusted_knowledge_bundle
from core.knowledge.types import RetrievalBudget, RetrievalContext, SERVICE_WIKIPEDIA

SERVICE_ID = SERVICE_WIKIPEDIA
SERVICE_VERSION = "0.1.0"


class WikipediaKnowledgeService:
    id = SERVICE_ID
    name = "Wikipedia"
    description = "English Wikipedia intro extracts only."
    version = SERVICE_VERSION

    def default_budget(self) -> RetrievalBudget:
        return RetrievalBudget(max_results=3, max_adapter_calls=1)

    def retrieve(self, ctx: RetrievalContext):
        t0 = time.time()
        query = ctx.query
        budget = ctx.budget or self.default_budget()
        rows = search_wikipedia(query, max_results=budget.max_results)
        latency_ms = (time.time() - t0) * 1000
        raw_audit = [dict(r) for r in rows]
        if not rows:
            return (
                build_empty_bundle(
                    query_raw=query,
                    query_resolved=query,
                    latency_ms=latency_ms,
                    knowledge_service=SERVICE_WIKIPEDIA,
                ),
                None,
                raw_audit,
            )
        bundle = build_trusted_knowledge_bundle(
            query_raw=query,
            query_resolved=query,
            kept_rows=[dict(r) for r in rows],
            rejected_count=0,
            latency_ms=latency_ms,
            adapter_calls=(WIKI_ADAPTER,),
            stop_reason="sufficient_evidence",
            knowledge_service=SERVICE_WIKIPEDIA,
        )
        return bundle, None, raw_audit
