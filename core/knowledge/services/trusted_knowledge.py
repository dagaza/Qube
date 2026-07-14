"""Trusted knowledge service (Phase 1: Wikipedia + allowlisted web)."""

from __future__ import annotations

from core.knowledge.pipeline_trusted import TrustedEvidencePipeline
from core.knowledge.types import RetrievalBudget, RetrievalContext, SERVICE_TRUSTED_KNOWLEDGE

SERVICE_ID = SERVICE_TRUSTED_KNOWLEDGE
SERVICE_VERSION = "0.1.0"


class TrustedKnowledgeService:
    id = SERVICE_ID
    name = "Trusted knowledge"
    description = "Wikipedia extracts with gov/edu/wikipedia DDG fallback."
    version = SERVICE_VERSION

    def __init__(self) -> None:
        self._pipeline = TrustedEvidencePipeline()

    def default_budget(self) -> RetrievalBudget:
        return RetrievalBudget(max_results=3, max_adapter_calls=2)

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
