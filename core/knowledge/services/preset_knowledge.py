"""Preset-backed knowledge service."""

from __future__ import annotations

from core.knowledge.pipeline_generic import GenericOrchestrationPipeline
from core.knowledge.presets import load_preset
from core.knowledge.types import RetrievalBudget, RetrievalContext

SERVICE_ID = "preset_knowledge"


class PresetKnowledgeService:
    id = SERVICE_ID

    def default_budget(self) -> RetrievalBudget:
        return RetrievalBudget(max_results=3, max_adapter_calls=8, max_latency_ms=8000)

    def retrieve(self, ctx: RetrievalContext):
        preset_id = getattr(ctx, "preset_id", None) or ""
        preset = load_preset(str(preset_id))
        if preset is None:
            from core.knowledge.services.general_web import GeneralWebKnowledgeService

            return GeneralWebKnowledgeService().retrieve(ctx)

        budget = ctx.budget or self.default_budget()
        pipeline = GenericOrchestrationPipeline(
            base_service=preset.base_service,
            adapter_ids=tuple(preset.adapters),
            ranking_profile=preset.ranking_profile,
            query_planner=preset.query_planner,
            preset_id=preset.id,
            adapter_policy=preset.adapter_policy,
        )
        merged = RetrievalContext(
            query=ctx.query,
            semantic_query=ctx.semantic_query,
            knowledge_service=preset.base_service,
            query_vector=ctx.query_vector,
            embed_fn=ctx.embed_fn,
            budget=budget,
            adapter_filter=tuple(preset.adapters),
            library_store=ctx.library_store,
            source_filter=ctx.source_filter,
            preset_id=preset.id,
            retrieval_profile=ctx.retrieval_profile,
        )
        return pipeline.run(merged)
