"""Internal corpus knowledge service (user library via LanceDB)."""

from __future__ import annotations

from core.knowledge.pipeline_internal_corpus import InternalCorpusEvidencePipeline
from core.knowledge.types import RetrievalBudget, RetrievalContext, SERVICE_INTERNAL_CORPUS

SERVICE_ID = SERVICE_INTERNAL_CORPUS
SERVICE_VERSION = "0.1.0"


class InternalCorpusKnowledgeService:
    id = SERVICE_ID
    name = "Internal corpus"
    description = "Hybrid vector + FTS search over your indexed library documents."
    version = SERVICE_VERSION

    def __init__(self) -> None:
        self._pipeline = InternalCorpusEvidencePipeline()

    def default_budget(self) -> RetrievalBudget:
        return RetrievalBudget(max_results=5, max_adapter_calls=1)

    def retrieve(self, ctx: RetrievalContext):
        budget = ctx.budget or self.default_budget()
        merged = RetrievalContext(
            query=ctx.query,
            semantic_query=ctx.semantic_query,
            knowledge_service=SERVICE_ID,
            query_vector=ctx.query_vector,
            embed_fn=ctx.embed_fn,
            budget=budget,
            library_store=ctx.library_store,
            source_filter=ctx.source_filter,
            source_prefix_filter=ctx.source_prefix_filter,
        )
        return self._pipeline.run(merged)
