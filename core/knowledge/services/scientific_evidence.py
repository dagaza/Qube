"""Scientific evidence knowledge service (Phase 2)."""

from __future__ import annotations

from core.knowledge.pipeline_scientific import ScientificEvidencePipeline
from core.knowledge.types import RetrievalBudget, RetrievalContext, SERVICE_SCIENTIFIC_EVIDENCE

SERVICE_ID = SERVICE_SCIENTIFIC_EVIDENCE
SERVICE_VERSION = "0.1.0"


class ScientificEvidenceService:
    id = SERVICE_ID
    name = "Scientific evidence"
    description = "PubMed, OpenAlex, and arXiv abstracts with metadata."
    version = SERVICE_VERSION

    def __init__(self) -> None:
        self._pipeline = ScientificEvidencePipeline()

    def default_budget(self) -> RetrievalBudget:
        return RetrievalBudget(max_results=3, max_adapter_calls=3)

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
