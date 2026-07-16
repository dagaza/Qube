"""General web knowledge service (Phase 0)."""

from __future__ import annotations

from core.knowledge.pipeline import EvidencePipeline
from core.knowledge.types import RetrievalBudget, RetrievalContext

SERVICE_ID = "general_web"
SERVICE_VERSION = "0.2.0"


class GeneralWebKnowledgeService:
    id = SERVICE_ID
    name = "General web"
    description = "DuckDuckGo discovery with optional page fetch and section-ranked evidence."
    version = SERVICE_VERSION

    def __init__(self) -> None:
        self._pipeline = EvidencePipeline()

    def default_budget(self) -> RetrievalBudget:
        return RetrievalBudget(max_results=3, max_adapter_calls=1)

    def retrieve(self, ctx: RetrievalContext):
        budget = ctx.budget or self.default_budget()
        merged = RetrievalContext(
            query=ctx.query,
            semantic_query=ctx.semantic_query,
            knowledge_service=SERVICE_ID,
            query_vector=ctx.query_vector,
            embed_fn=ctx.embed_fn,
            budget=budget,
            adapter_filter=ctx.adapter_filter,
            library_store=ctx.library_store,
            source_filter=ctx.source_filter,
            preset_id=ctx.preset_id,
            retrieval_profile=ctx.retrieval_profile,
            composer_tool=ctx.composer_tool,
            fetch_url_count=ctx.fetch_url_count,
            site_bias=ctx.site_bias,
        )
        return self._pipeline.run(merged)
