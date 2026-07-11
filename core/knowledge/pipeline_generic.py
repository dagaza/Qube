"""Generic orchestration pipeline for presets and configured sources."""

from __future__ import annotations

import time
from typing import Any

from core.app_settings import get_knowledge_source_preferences
from core.knowledge.budget_enforcement import TurnBudgetTracker
from core.knowledge.bundle_builder import build_empty_bundle, build_generic_bundle
from core.knowledge.http_metrics import begin_turn_http_metrics, snapshot_turn_http_summary
from core.knowledge.orchestration_kernel import (
    PipelineStageTrace,
    StageTraceCollector,
    collect_adapter_rows,
    enforce_adapter_policy,
)
from core.knowledge.query_planners import plan_query
from core.knowledge.ranking_profiles import rank_rows
from core.knowledge.retrieval_profiles import get_profile_spec, normalize_profile_id
from core.knowledge.source_preferences import resolve_service_adapters
from core.knowledge.types import EvidenceBundle, RetrievalContext


class GenericOrchestrationPipeline:
    """Parallel adapter fan-out with generic ranking and bundle assembly."""

    def __init__(
        self,
        *,
        base_service: str,
        adapter_ids: tuple[str, ...] | None = None,
        ranking_profile: str = "generic",
        query_planner: str = "passthrough",
        retrieval_strategy: str | None = None,
        preset_id: str | None = None,
        adapter_policy: str | None = "fixed_order",
    ) -> None:
        self.base_service = base_service
        self.adapter_ids = adapter_ids
        self.ranking_profile = ranking_profile
        self.query_planner = query_planner
        self.preset_id = preset_id
        self.adapter_policy = adapter_policy
        if retrieval_strategy:
            self.retrieval_strategy = retrieval_strategy
        elif preset_id:
            self.retrieval_strategy = f"preset:{preset_id}"
        else:
            self.retrieval_strategy = "generic_parallel"

    def _resolve_adapters(self, ctx: RetrievalContext) -> tuple[str, ...]:
        if self.adapter_ids:
            return enforce_adapter_policy(
                tuple(self.adapter_ids),
                policy=self.adapter_policy,
            )
        if ctx.adapter_filter:
            return enforce_adapter_policy(
                tuple(ctx.adapter_filter),
                policy=self.adapter_policy,
            )
        resolved = resolve_service_adapters(
            self.base_service,
            query=ctx.query,
            composer_adapter_filter=ctx.adapter_filter,
            stored_preferences=get_knowledge_source_preferences(),
        )
        return enforce_adapter_policy(tuple(resolved), policy=self.adapter_policy)

    def run(
        self, ctx: RetrievalContext
    ) -> tuple[EvidenceBundle, dict[str, Any] | None, list[dict[str, Any]]]:
        t0 = time.time()
        begin_turn_http_metrics()
        stage_collector = StageTraceCollector()
        profile = get_profile_spec(normalize_profile_id(ctx.retrieval_profile))

        plan_t0 = time.time()
        plan = plan_query(
            ctx.query,
            semantic_query=ctx.semantic_query or ctx.query,
            planner=self.query_planner,
        )
        stage_collector.record(
            PipelineStageTrace(
                stage="plan",
                latency_ms=(time.time() - plan_t0) * 1000.0,
                outputs={"search_query": str(plan["search_query"])},
            )
        )

        effective_budget = profile.materialize_budget(ctx.budget)
        budget_tracker = TurnBudgetTracker(effective_budget)
        adapter_ids = self._resolve_adapters(ctx)
        per_adapter = max(1, min(3, effective_budget.max_results))
        adapter_calls, raw_audit, candidates, _attempts = collect_adapter_rows(
            adapter_ids,
            search_query=str(plan["search_query"]),
            per_adapter=per_adapter,
            budget_tracker=budget_tracker,
            max_parallel=profile.max_parallel_adapters,
            profile=profile,
            stage_collector=stage_collector,
        )

        rank_t0 = time.time()
        effective_ranking = self.ranking_profile
        if profile.ranking_profile_hint and effective_ranking == "generic":
            effective_ranking = profile.ranking_profile_hint
        ranked = rank_rows(
            candidates,
            query=str(plan["search_query"]),
            profile=effective_ranking,
            max_results=effective_budget.max_results,
        )
        stage_collector.record(
            PipelineStageTrace(
                stage="rank",
                latency_ms=(time.time() - rank_t0) * 1000.0,
                inputs_count=len(candidates),
                outputs_count=len(ranked),
                outputs={"ranking_profile": effective_ranking},
            )
        )

        latency_ms = (time.time() - t0) * 1000.0
        rel_diag: dict[str, Any] = {
            "web_results_raw_count": len(candidates),
            "ranking_profile": effective_ranking,
            "query_planner": self.query_planner,
            "preset_id": self.preset_id,
            "retrieval_profile": profile.id,
            "http_summary": snapshot_turn_http_summary(),
            "pipeline_stages": stage_collector.as_dicts(),
        }
        if not ranked:
            bundle = build_empty_bundle(
                query_raw=ctx.query,
                query_resolved=str(plan["search_query"]),
                latency_ms=latency_ms,
                rejected_count=max(0, len(candidates)),
                knowledge_service=self.base_service,
            )
            stage_collector.record(
                PipelineStageTrace(
                    stage="bundle",
                    latency_ms=0.0,
                    outputs_count=0,
                )
            )
            rel_diag["pipeline_stages"] = stage_collector.as_dicts()
            return bundle, rel_diag, raw_audit

        bundle = build_generic_bundle(
            query_raw=ctx.query,
            query_resolved=str(plan["search_query"]),
            kept_rows=ranked,
            rejected_count=max(0, len(candidates) - len(ranked)),
            latency_ms=latency_ms,
            knowledge_service=self.base_service,
            retrieval_strategy=self.retrieval_strategy,
            adapter_calls=tuple(adapter_calls),
            ranking_profile=effective_ranking,
            preset_id=self.preset_id,
        )
        stage_collector.record(
            PipelineStageTrace(
                stage="bundle",
                latency_ms=0.0,
                outputs_count=len(bundle.sources),
            )
        )
        rel_diag["pipeline_stages"] = stage_collector.as_dicts()
        return bundle, rel_diag, raw_audit
