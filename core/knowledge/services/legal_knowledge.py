"""Legal knowledge service (@legal — CourtListener and jurisdiction adapters)."""

from __future__ import annotations

from core.knowledge.pipeline_legal import LegalEvidencePipeline
from core.knowledge.types import RetrievalBudget, RetrievalContext, SERVICE_LEGAL_KNOWLEDGE

SERVICE_ID = SERVICE_LEGAL_KNOWLEDGE
SERVICE_VERSION = "0.1.0"


class LegalKnowledgeService:
    id = SERVICE_ID
    name = "Legal knowledge"
    description = "U.S. case law opinions via CourtListener."
    version = SERVICE_VERSION

    def __init__(self) -> None:
        self._pipeline = LegalEvidencePipeline()

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
        )
        return self._pipeline.run(merged)
