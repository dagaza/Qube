"""
Shadow-mode routing hysteresis simulation.

Post-hoc transformation of perturbation variant routes to estimate whether
enter/exit threshold buffers would reduce boundary-driven instability.
Does NOT modify CognitiveRouterV4 or production routing.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.router_evaluation import normalize_route

HYSTERESIS_SCHEMA = "qube.routing_hysteresis.v1"

_RETRIEVAL_ROUTES: frozenset[str] = frozenset({"memory", "rag", "hybrid", "web"})

_HYSTERESIS_CATEGORIES: frozenset[str] = frozenset({
    "general_knowledge_retrieval_tempting",
    "follow_up",
    "ambiguous",
})

RETRIEVAL_CONSISTENCY_GUARD_DELTA = 0.02


@dataclass
class HysteresisConfig:
    """Simulated enter/exit threshold offsets around a base margin threshold."""

    base_threshold: float = 0.10
    delta_enter: float = 0.05
    delta_exit: float = 0.05
    enter_margin_band: float = 0.05
    low_margin_band_high: float = 0.05

    @property
    def enter_retrieval_threshold(self) -> float:
        return self.base_threshold + self.delta_enter

    @property
    def exit_retrieval_threshold(self) -> float:
        return self.base_threshold - self.delta_exit


@dataclass
class HysteresisVariantComparison:
    case_id: str
    variant_id: str
    base_route: str
    hysteresis_route: str
    flipped: bool
    flip_type: str
    confidence_margin: float
    chat_score: float
    top_score: float
    second_best_score: float
    perturbation_type: str = ""


@dataclass
class HysteresisSimulationResult:
    summary: dict[str, Any]
    comparisons: list[HysteresisVariantComparison] = field(default_factory=list)


def apply_hysteresis_shadow_route(
    base_route: str,
    confidence_margin: float,
    chat_score: float,
    top_score: float,
    second_best_score: float,
    *,
    previous_route: str | None = None,
    config: HysteresisConfig | None = None,
) -> str:
    """
    Shadow hysteresis transform on a single route decision.

    Rules (applied in order):
    A. Block weak NONE→RETRIEVAL transitions when score gap is below enter band.
    B. Prevent RETRIEVAL→NONE collapse when margin is below exit threshold.
    C. Lock to previous/majority route inside the low-margin protection band.
    """
    cfg = config or HysteresisConfig()
    route = normalize_route(base_route)
    anchor = normalize_route(previous_route) if previous_route else route
    score_gap = top_score - second_best_score
    result = route

    # A. NONE → RETRIEVAL transition guard (prevent over-triggering)
    if route == "none" and score_gap < cfg.enter_margin_band:
        result = "none"
    elif route in _RETRIEVAL_ROUTES and score_gap < cfg.enter_margin_band:
        if anchor == "none":
            result = "none"

    # B. RETRIEVAL → NONE transition guard (prevent premature collapse)
    if route in _RETRIEVAL_ROUTES and confidence_margin < cfg.exit_retrieval_threshold:
        result = route

    # C. Low-margin protection band — lock to cluster majority / previous route
    if 0.0 <= confidence_margin <= cfg.low_margin_band_high and previous_route:
        result = anchor

    return result


def _majority_route(routes: list[str]) -> str:
    if not routes:
        return "none"
    counts = Counter(normalize_route(r) for r in routes)
    return counts.most_common(1)[0][0]


def _route_consistency(routes: list[str]) -> float:
    if not routes:
        return 1.0
    normalized = [normalize_route(r) for r in routes]
    unique = len(set(normalized))
    return 1.0 - (unique / len(normalized))


def _retrieval_consistency(hits_flags: list[int]) -> float:
    if not hits_flags or len(hits_flags) < 2:
        return 1.0
    return 1.0 - statistics.pvariance(hits_flags)


def _flip_type(base: str, other: str) -> str:
    a = normalize_route(base)
    b = normalize_route(other)
    if a == b:
        return "none"
    return f"{a}↔{b}"


def _count_hybrid_none_flips(routes: list[str], anchor: str) -> int:
    """Count variants whose route disagrees with anchor across the none/hybrid boundary."""
    anchor_n = normalize_route(anchor)
    count = 0
    for r in routes:
        rn = normalize_route(r)
        if {rn, anchor_n} == {"none", "hybrid"}:
            count += 1
    return count


def simulate_hysteresis_on_perturbation(
    perturbation_analysis: Any,
    *,
    config: HysteresisConfig | None = None,
) -> HysteresisSimulationResult:
    """
    Apply hysteresis shadow routing to an existing perturbation analysis.

    ``perturbation_analysis`` is a ``RoutePerturbationAnalysis`` from
    ``eval.routing_perturbation``.
    """
    cfg = config or HysteresisConfig()
    comparisons: list[HysteresisVariantComparison] = []

    baseline_consistency_sum = 0.0
    hysteresis_consistency_sum = 0.0
    baseline_retrieval_sum = 0.0
    hysteresis_retrieval_sum = 0.0
    original_flip_count = 0
    hysteresis_flip_count = 0
    baseline_hybrid_none = 0
    hysteresis_hybrid_none = 0
    low_margin_baseline_unstable = 0
    low_margin_hysteresis_unstable = 0
    low_margin_cases = 0

    by_category: dict[str, dict[str, float]] = {}

    for case_report in perturbation_analysis.cases:
        variants = case_report.variants
        if not variants:
            continue

        base_routes = [normalize_route(v.execution_route) for v in variants]
        majority = _majority_route(base_routes)
        anchor = _majority_route(
            [normalize_route(case_report.base_route), *base_routes]
        )

        hysteresis_routes: list[str] = []
        hits_flags = [
            1 if (v.memory_hits + v.rag_hits + v.web_hits) > 0 else 0
            for v in variants
        ]

        for v in variants:
            base = normalize_route(v.execution_route)
            hyst = apply_hysteresis_shadow_route(
                base,
                v.confidence_margin,
                v.chat_score,
                v.top_score,
                getattr(v, "second_best_score", 0.0),
                previous_route=anchor,
                config=cfg,
            )
            hysteresis_routes.append(hyst)

            flipped = base != hyst
            comparisons.append(
                HysteresisVariantComparison(
                    case_id=case_report.case_id,
                    variant_id=v.variant_id,
                    base_route=base,
                    hysteresis_route=hyst,
                    flipped=flipped,
                    flip_type=_flip_type(base, hyst) if flipped else "none",
                    confidence_margin=round(v.confidence_margin, 4),
                    chat_score=round(v.chat_score, 4),
                    top_score=round(v.top_score, 4),
                    second_best_score=round(getattr(v, "second_best_score", 0.0), 4),
                    perturbation_type=v.perturbation_type,
                )
            )

            if base != majority:
                original_flip_count += 1
            if hyst != majority:
                hysteresis_flip_count += 1

        baseline_hybrid_none += _count_hybrid_none_flips(base_routes, anchor)
        hysteresis_hybrid_none += _count_hybrid_none_flips(hysteresis_routes, anchor)

        b_cons = _route_consistency(base_routes)
        h_cons = _route_consistency(hysteresis_routes)
        baseline_consistency_sum += b_cons
        hysteresis_consistency_sum += h_cons

        b_retr = _retrieval_consistency(hits_flags)
        baseline_retrieval_sum += b_retr
        hysteresis_retrieval_sum += b_retr

        if any(0.0 <= v.confidence_margin <= cfg.low_margin_band_high for v in variants):
            low_margin_cases += 1
            if len(set(base_routes)) > 1:
                low_margin_baseline_unstable += 1
            if len(set(hysteresis_routes)) > 1:
                low_margin_hysteresis_unstable += 1

        cat = case_report.category
        if cat in _HYSTERESIS_CATEGORIES:
            bucket = by_category.setdefault(
                cat,
                {
                    "count": 0,
                    "baseline_consistency_sum": 0.0,
                    "hysteresis_consistency_sum": 0.0,
                    "baseline_hybrid_none": 0,
                    "hysteresis_hybrid_none": 0,
                },
            )
            bucket["count"] += 1
            bucket["baseline_consistency_sum"] += b_cons
            bucket["hysteresis_consistency_sum"] += h_cons
            bucket["baseline_hybrid_none"] += _count_hybrid_none_flips(base_routes, anchor)
            bucket["hysteresis_hybrid_none"] += _count_hybrid_none_flips(
                hysteresis_routes, anchor
            )

    n_cases = len(perturbation_analysis.cases)
    avg_baseline_cons = baseline_consistency_sum / n_cases if n_cases else 0.0
    avg_hysteresis_cons = hysteresis_consistency_sum / n_cases if n_cases else 0.0
    avg_baseline_retr = baseline_retrieval_sum / n_cases if n_cases else 0.0
    avg_hysteresis_retr = hysteresis_retrieval_sum / n_cases if n_cases else 0.0
    retr_delta = avg_hysteresis_retr - avg_baseline_retr

    flip_reduction = (
        (original_flip_count - hysteresis_flip_count) / original_flip_count
        if original_flip_count
        else 0.0
    )
    hybrid_none_prevented = baseline_hybrid_none - hysteresis_hybrid_none
    hybrid_none_reduction = (
        hybrid_none_prevented / baseline_hybrid_none if baseline_hybrid_none else 0.0
    )

    for cat, bucket in by_category.items():
        c = bucket["count"]
        bucket["avg_baseline_consistency"] = bucket["baseline_consistency_sum"] / c
        bucket["avg_hysteresis_consistency"] = bucket["hysteresis_consistency_sum"] / c
        bucket["stability_gain"] = (
            bucket["avg_hysteresis_consistency"] - bucket["avg_baseline_consistency"]
        )
        b_hn = bucket["baseline_hybrid_none"]
        h_hn = bucket["hysteresis_hybrid_none"]
        bucket["hybrid_none_suppression_rate"] = (
            (b_hn - h_hn) / b_hn if b_hn else 0.0
        )

    summary = {
        "cases_analyzed": n_cases,
        "variants_analyzed": len(comparisons),
        "flip_reduction_rate": round(flip_reduction, 4),
        "stability_gain": round(avg_hysteresis_cons - avg_baseline_cons, 4),
        "hybrid_none_flip_reduction": round(hybrid_none_reduction, 4),
        "hybrid_none_baseline_flips": baseline_hybrid_none,
        "hybrid_none_hysteresis_flips": hysteresis_hybrid_none,
        "hybrid_none_prevented": hybrid_none_prevented,
        "retrieval_consistency_delta": round(retr_delta, 4),
        "safety_flag": retr_delta < -RETRIEVAL_CONSISTENCY_GUARD_DELTA,
        "avg_baseline_route_consistency": round(avg_baseline_cons, 4),
        "avg_hysteresis_route_consistency": round(avg_hysteresis_cons, 4),
        "avg_baseline_retrieval_consistency": round(avg_baseline_retr, 4),
        "avg_hysteresis_retrieval_consistency": round(avg_hysteresis_retr, 4),
        "baseline_route_flips": original_flip_count,
        "hysteresis_route_flips": hysteresis_flip_count,
        "low_margin_instability_baseline": low_margin_baseline_unstable,
        "low_margin_instability_hysteresis": low_margin_hysteresis_unstable,
        "low_margin_cases": low_margin_cases,
        "config": asdict(cfg),
        "by_category": by_category,
        "comparison_table": {
            "route_flips": {
                "baseline": original_flip_count,
                "hysteresis": hysteresis_flip_count,
                "delta": original_flip_count - hysteresis_flip_count,
            },
            "hybrid_none_flips": {
                "baseline": baseline_hybrid_none,
                "hysteresis": hysteresis_hybrid_none,
                "delta": hybrid_none_prevented,
            },
            "retrieval_consistency": {
                "baseline": round(avg_baseline_retr, 4),
                "hysteresis": round(avg_hysteresis_retr, 4),
                "delta": round(retr_delta, 4),
            },
        },
    }

    return HysteresisSimulationResult(summary=summary, comparisons=comparisons)


def export_hysteresis_comparison_json(
    path: Path,
    result: HysteresisSimulationResult,
) -> None:
    payload = {
        "schema": HYSTERESIS_SCHEMA,
        "summary": result.summary,
        "comparisons": [asdict(c) for c in result.comparisons],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
