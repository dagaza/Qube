"""Finance knowledge service (@finance — SEC EDGAR filings)."""

from __future__ import annotations

from core.knowledge.pipeline_finance import FinanceEvidencePipeline
from core.knowledge.types import RetrievalBudget, RetrievalContext, SERVICE_FINANCE_KNOWLEDGE

SERVICE_ID = SERVICE_FINANCE_KNOWLEDGE
SERVICE_VERSION = "0.1.0"


class FinanceKnowledgeService:
    id = SERVICE_ID
    name = "Finance knowledge"
    description = "SEC EDGAR company filings (10-K, 10-Q, 8-K)."
    version = SERVICE_VERSION

    def __init__(self) -> None:
        self._pipeline = FinanceEvidencePipeline()

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
