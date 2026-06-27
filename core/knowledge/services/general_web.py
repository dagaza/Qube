"""General web knowledge service (Phase 0)."""

from __future__ import annotations

from core.knowledge.pipeline import EvidencePipeline
from core.knowledge.types import RetrievalBudget, RetrievalContext

SERVICE_ID = "general_web"
SERVICE_VERSION = "0.1.0"


class GeneralWebKnowledgeService:
    id = SERVICE_ID
    name = "General web"
    description = "DuckDuckGo SERP snippets with relevance gating."
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
        )
        return self._pipeline.run(merged)
