"""Adaptive stopping heuristics for evidence pipelines."""

from __future__ import annotations

from core.knowledge.types import COVERAGE_ADEQUATE, COVERAGE_EXCELLENT, COVERAGE_POOR

MIN_RELEVANCE_FOR_STOP = 0.25
MIN_SOURCES_FOR_EXCELLENT = 3
CONFIDENCE_STOP_THRESHOLD = 0.72


def adaptive_stop_reason(
    *,
    kept_count: int,
    max_results: int,
    avg_relevance: float,
    adapter_count: int,
    abstract_count: int,
) -> str:
    """Choose stop_reason from coverage signals instead of fixed N."""
    if kept_count <= 0:
        return "no_evidence"
    if (
        kept_count >= MIN_SOURCES_FOR_EXCELLENT
        and avg_relevance >= MIN_RELEVANCE_FOR_STOP
        and adapter_count >= 2
        and abstract_count >= 2
    ):
        return "sufficient_evidence"
    if kept_count >= max_results and avg_relevance >= MIN_RELEVANCE_FOR_STOP:
        return "sufficient_evidence"
    if kept_count >= 2 and avg_relevance >= CONFIDENCE_STOP_THRESHOLD:
        return "sufficient_evidence"
    if kept_count >= max_results:
        return "budget_exhausted"
    return "budget_exhausted"


def coverage_from_signals(
    *,
    kept_count: int,
    avg_relevance: float,
    adapter_count: int,
    abstract_count: int,
) -> tuple[str, str]:
    if kept_count <= 0:
        return "none", "No sources retained after ranking."
    if (
        kept_count >= 3
        and avg_relevance >= 0.35
        and adapter_count >= 2
        and abstract_count >= 2
    ):
        return (
            COVERAGE_EXCELLENT,
            f"{kept_count} sources across {adapter_count} indexes with "
            f"{abstract_count} abstracts (avg relevance {avg_relevance:.2f}).",
        )
    if kept_count >= 2 and abstract_count >= 1:
        return (
            COVERAGE_ADEQUATE,
            f"{kept_count} ranked source(s); {abstract_count} with abstracts.",
        )
    return (
        COVERAGE_POOR,
        "Limited ranked coverage; corroboration may be insufficient.",
    )
