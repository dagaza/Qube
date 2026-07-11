"""Shared orchestration primitives for knowledge pipelines."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

from core.knowledge.adapters.registry import get_search_function
from core.knowledge.budget_enforcement import BudgetExceededError, TurnBudgetTracker
from core.knowledge.retrieval_profiles import RetrievalProfileSpec, order_adapter_ids

SearchFn = Callable[..., list[dict[str, Any]]]


@dataclass
class AdapterAttempt:
    adapter_id: str
    status: str  # ok | skipped | failed | budget_exceeded
    outputs_count: int = 0
    error: str | None = None


@dataclass
class PipelineStageTrace:
    stage: str
    latency_ms: float
    status: str = "ok"
    adapter: str | None = None
    inputs_count: int = 0
    outputs_count: int = 0
    cache_hit: bool = False
    outputs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "stage": self.stage,
            "latency_ms": round(self.latency_ms, 2),
            "status": self.status,
        }
        if self.adapter:
            payload["adapter"] = self.adapter
        if self.inputs_count:
            payload["inputs_count"] = self.inputs_count
        if self.outputs_count:
            payload["outputs_count"] = self.outputs_count
        if self.cache_hit:
            payload["cache_hit"] = True
        if self.outputs:
            payload["outputs"] = dict(self.outputs)
        return payload


class StageTraceCollector:
    def __init__(self) -> None:
        self._stages: list[PipelineStageTrace] = []

    def record(self, stage: PipelineStageTrace) -> None:
        self._stages.append(stage)

    def as_dicts(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self._stages]


def enforce_adapter_policy(
    adapter_ids: tuple[str, ...],
    *,
    policy: str | None,
) -> tuple[str, ...]:
    """Apply preset adapter_policy (fixed_order keeps explicit list order)."""
    if not adapter_ids:
        return adapter_ids
    policy_name = (policy or "fixed_order").strip().lower()
    if policy_name == "fixed_order":
        return adapter_ids
    return adapter_ids


def collect_adapter_rows(
    adapter_ids: tuple[str, ...],
    *,
    search_query: str,
    per_adapter: int,
    budget_tracker: TurnBudgetTracker | None = None,
    max_parallel: int = 3,
    profile: RetrievalProfileSpec | None = None,
    stage_collector: StageTraceCollector | None = None,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]], list[AdapterAttempt]]:
    ordered = adapter_ids
    if profile is not None:
        ordered = order_adapter_ids(adapter_ids, profile=profile)

    adapter_calls: list[str] = []
    raw_audit: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    attempts: list[AdapterAttempt] = []
    if not ordered:
        return adapter_calls, raw_audit, candidates, attempts

    workers = min(max_parallel, len(ordered))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures: dict[Any, str] = {}
        for aid in ordered:
            if budget_tracker is not None:
                try:
                    budget_tracker.check_latency()
                    budget_tracker.record_adapter_call()
                except BudgetExceededError:
                    attempts.append(
                        AdapterAttempt(aid, "budget_exceeded", error="turn budget exceeded")
                    )
                    if stage_collector is not None:
                        stage_collector.record(
                            PipelineStageTrace(
                                stage="adapter_fetch",
                                latency_ms=0.0,
                                status="budget_exceeded",
                                adapter=aid,
                            )
                        )
                    break
            search_fn = get_search_function(aid)
            if search_fn is None:
                attempts.append(AdapterAttempt(aid, "skipped", error="unknown adapter"))
                if stage_collector is not None:
                    stage_collector.record(
                        PipelineStageTrace(
                            stage="adapter_fetch",
                            latency_ms=0.0,
                            status="skipped",
                            adapter=aid,
                        )
                    )
                continue
            futures[
                pool.submit(
                    search_fn,
                    search_query,
                    max_results=per_adapter,
                )
            ] = aid

        for future in as_completed(futures):
            aid = futures[future]
            t0 = time.time()
            try:
                rows = future.result()
            except Exception as exc:
                rows = []
                attempts.append(AdapterAttempt(aid, "failed", error=str(exc)))
                if stage_collector is not None:
                    stage_collector.record(
                        PipelineStageTrace(
                            stage="adapter_fetch",
                            latency_ms=(time.time() - t0) * 1000.0,
                            status="failed",
                            adapter=aid,
                            outputs_count=0,
                        )
                    )
                continue
            latency = (time.time() - t0) * 1000.0
            if rows:
                adapter_calls.append(aid)
                raw_audit.extend(dict(r) for r in rows)
                candidates.extend(dict(r) for r in rows)
                attempts.append(AdapterAttempt(aid, "ok", outputs_count=len(rows)))
            else:
                attempts.append(AdapterAttempt(aid, "ok", outputs_count=0))
            if stage_collector is not None:
                stage_collector.record(
                    PipelineStageTrace(
                        stage="adapter_fetch",
                        latency_ms=latency,
                        status="ok",
                        adapter=aid,
                        outputs_count=len(rows),
                    )
                )
    return adapter_calls, raw_audit, candidates, attempts
