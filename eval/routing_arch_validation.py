"""
Continuous recall-fusion architectural validation layer.

End-to-end validation of the continuous pilot routing candidate with aggregated
metrics by cluster, category, and flip type. Shadow-only; no production changes.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.router_evaluation import normalize_route
from eval.routing_continuous_pilot import (
    PilotAnalysis,
    PilotClusterReport,
    PilotVariantRecord,
    analyze_continuous_pilot_routing,
)
from eval.routing_retrieval_propensity import (
    DELTA_GRID,
    T_NONE_GRID,
    PropensityThresholds,
    PropensityWeights,
    _evaluate_config,
)

VALIDATION_SCHEMA = "qube.routing_arch_validation.v1"


@dataclass
class ArchValidationAnalysis:
    summary: dict[str, Any]
    pilot: PilotAnalysis
    threshold_sweep: list[dict[str, Any]] = field(default_factory=list)
    top_unstable_clusters: list[dict[str, Any]] = field(default_factory=list)


def _flip_pattern(routes: list[str]) -> str:
    unique = sorted(set(normalize_route(r) for r in routes))
    if len(unique) <= 1:
        return unique[0] if unique else "none"
    return " ↔ ".join(unique)


def _aggregate_by_category(
    clusters: list[PilotClusterReport],
    variants: list[PilotVariantRecord],
) -> dict[str, dict[str, Any]]:
    by_cat: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "clusters": 0,
            "unstable_baseline": 0,
            "unstable_pilot": 0,
            "retrieval_loss_sum": 0.0,
            "hybrid_none_baseline": 0,
            "hybrid_none_pilot": 0,
            "memory_rag_baseline": 0,
            "memory_rag_pilot": 0,
            "pilot_agreement": 0,
            "canonical_agreement": 0,
            "variant_count": 0,
        }
    )
    for cr in clusters:
        b = by_cat[cr.category]
        b["clusters"] += 1
        if not cr.baseline_stable:
            b["unstable_baseline"] += 1
        if not cr.pilot_stable:
            b["unstable_pilot"] += 1
        b["retrieval_loss_sum"] += cr.retrieval_loss_estimate
        b["hybrid_none_baseline"] += cr.hybrid_none_flips_baseline
        b["hybrid_none_pilot"] += cr.hybrid_none_flips_pilot
        b["memory_rag_baseline"] += cr.memory_rag_flips_baseline
        b["memory_rag_pilot"] += cr.memory_rag_flips_pilot
    for v in variants:
        b = by_cat[v.category]
        b["variant_count"] += 1
        if v.pilot_vs_baseline_agreement:
            b["pilot_agreement"] += 1
        if v.pilot_vs_canonical_agreement:
            b["canonical_agreement"] += 1

    out: dict[str, dict[str, Any]] = {}
    for cat, b in by_cat.items():
        n_var = b["variant_count"] or 1
        n_cl = b["clusters"] or 1
        unstable_b = b["unstable_baseline"]
        out[cat] = {
            "clusters": b["clusters"],
            "instability_reduction_pct": (
                (unstable_b - b["unstable_pilot"]) / unstable_b if unstable_b else 0.0
            ),
            "avg_retrieval_loss_estimate": b["retrieval_loss_sum"] / n_cl,
            "hybrid_none_flip_delta": b["hybrid_none_baseline"] - b["hybrid_none_pilot"],
            "memory_rag_flip_delta": b["memory_rag_baseline"] - b["memory_rag_pilot"],
            "pilot_vs_baseline_agreement_rate": b["pilot_agreement"] / n_var,
            "pilot_vs_canonical_agreement_rate": b["canonical_agreement"] / n_var,
        }
    return out


def _aggregate_flip_types(clusters: list[PilotClusterReport]) -> dict[str, dict[str, int]]:
    return {
        "hybrid_none": {
            "baseline": sum(c.hybrid_none_flips_baseline for c in clusters),
            "pilot": sum(c.hybrid_none_flips_pilot for c in clusters),
            "hysteresis": sum(c.hybrid_none_flips_hysteresis for c in clusters),
            "canonical_shadow": sum(c.hybrid_none_flips_canonical_shadow for c in clusters),
        },
        "memory_rag": {
            "baseline": sum(c.memory_rag_flips_baseline for c in clusters),
            "pilot": sum(c.memory_rag_flips_pilot for c in clusters),
        },
    }


def _build_threshold_sweep(
    perturbation_analysis: Any,
    weights: PropensityWeights,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for t_none in T_NONE_GRID:
        for delta in DELTA_GRID:
            thresholds = PropensityThresholds(t_none=t_none, delta=delta)
            _variants, _clusters, metrics = _evaluate_config(
                perturbation_analysis,
                weights=weights,
                thresholds=thresholds,
            )
            rows.append({
                "T_none": t_none,
                "delta": delta,
                "instability_reduction_proxy": metrics.get("instability_reduction_proxy", 0.0),
                "retrieval_loss_proxy": metrics.get("retrieval_loss_proxy", 0.0),
                "recall_fusion_flip_reduction_pct": metrics.get(
                    "recall_fusion_flip_reduction_pct", 0.0
                ),
                "hybrid_none_flip_reduction_pct": metrics.get(
                    "hybrid_none_flip_reduction_pct", 0.0
                ),
            })
    return sorted(
        rows,
        key=lambda r: (
            r["instability_reduction_proxy"] - 2.0 * r["retrieval_loss_proxy"]
        ),
        reverse=True,
    )


def _top_unstable_clusters(
    pilot: PilotAnalysis,
    perturbation_analysis: Any,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    case_by_id = {c.case_id: c for c in perturbation_analysis.cases}
    unstable = [cr for cr in pilot.clusters if not cr.pilot_stable]
    ranked = sorted(
        unstable,
        key=lambda cr: (
            -cr.retrieval_loss_estimate,
            cr.hybrid_none_flips_pilot,
            cr.case_id,
        ),
    )[:limit]
    rows: list[dict[str, Any]] = []
    for cr in ranked:
        case = case_by_id.get(cr.case_id)
        base_routes = (
            [normalize_route(v.execution_route) for v in case.variants]
            if case
            else []
        )
        pilot_routes = [
            v.pilot_route for v in pilot.variants if v.case_id == cr.case_id
        ]
        rows.append({
            "case_id": cr.case_id,
            "category": cr.category,
            "canonical_route": cr.canonical_route_majority,
            "baseline_route_pattern": _flip_pattern(base_routes),
            "pilot_route_pattern": _flip_pattern(pilot_routes),
            "retrieval_loss_estimate": cr.retrieval_loss_estimate,
            "hybrid_none_flips_pilot": cr.hybrid_none_flips_pilot,
            "memory_rag_flips_pilot": cr.memory_rag_flips_pilot,
            "pilot_instability_reduction": cr.pilot_instability_reduction,
        })
    return rows


def analyze_continuous_arch_validation(
    perturbation_analysis: Any,
    *,
    pilot_analysis: Optional[PilotAnalysis] = None,
    weights: PropensityWeights | None = None,
    hysteresis_summary: dict[str, Any] | None = None,
    canonicalization_summary: dict[str, Any] | None = None,
    propensity_summary: dict[str, Any] | None = None,
) -> ArchValidationAnalysis:
    """
    Run full architectural validation on the continuous pilot routing candidate.
    """
    w = weights or PropensityWeights()
    pilot = pilot_analysis or analyze_continuous_pilot_routing(
        perturbation_analysis,
        weights=w,
        hysteresis_summary=hysteresis_summary,
        canonicalization_summary=canonicalization_summary,
        propensity_summary=propensity_summary,
    )
    ps = pilot.summary

    by_category = _aggregate_by_category(pilot.clusters, pilot.variants)
    flip_types = _aggregate_flip_types(pilot.clusters)
    threshold_sweep = _build_threshold_sweep(perturbation_analysis, w)
    top_unstable = _top_unstable_clusters(pilot, perturbation_analysis)

    comparison_matrix = {
        "unstable_clusters": {
            "baseline": ps.get("clusters_unstable_baseline", 0),
            "pilot": ps.get("clusters_unstable_pilot", 0),
            "hysteresis": ps.get("clusters_unstable_hysteresis", 0),
            "canonical_shadow": ps.get("clusters_unstable_canonical_shadow", 0),
        },
        "instability_reduction": {
            "pilot_vs_baseline": ps.get("instability_reduction_pct", 0.0),
            "hysteresis_stability_gain": (ps.get("hysteresis_comparison") or {}).get(
                "hysteresis_instability_reduction"
            ),
            "canonicalization_reduction": (ps.get("canonicalization_comparison") or {}).get(
                "canonicalization_instability_reduction"
            ),
        },
        "retrieval_loss_proxy": {
            "pilot": ps.get("retrieval_loss_proxy", 0.0),
            "canonicalization": (ps.get("canonicalization_comparison") or {}).get(
                "canonicalization_retrieval_loss"
            ),
        },
    }

    validation_passed = (
        ps.get("instability_reduction_pct", 0.0) >= 0.5
        and ps.get("retrieval_loss_proxy", 0.0) < 0.10
    )

    summary = {
        "avg_propensity_score": ps.get("avg_propensity_score"),
        "instability_reduction_pct": ps.get("instability_reduction_pct"),
        "retrieval_loss_proxy": ps.get("retrieval_loss_proxy"),
        "flip_reduction_vs_canonical": ps.get("flip_reduction_vs_canonical"),
        "hybrid_none_flip_reduction_pct": ps.get("hybrid_none_flip_reduction_pct"),
        "retrieval_continuity_score": ps.get("retrieval_continuity_score"),
        "best_thresholds": ps.get("best_thresholds"),
        "best_weight_set": ps.get("best_weight_set"),
        "pilot_vs_baseline_agreement_rate": ps.get("pilot_vs_baseline_agreement_rate"),
        "pilot_vs_canonical_agreement_rate": ps.get("pilot_vs_canonical_agreement_rate"),
        "pilot_resolves_all_unstable": ps.get("pilot_resolves_all_unstable"),
        "validation_passed": validation_passed,
        "by_category": by_category,
        "flip_type_summary": flip_types,
        "comparison_matrix": comparison_matrix,
        "interpretation": ps.get("interpretation"),
        "clusters_total": len(pilot.clusters),
        "variants_total": len(pilot.variants),
    }

    return ArchValidationAnalysis(
        summary=summary,
        pilot=pilot,
        threshold_sweep=threshold_sweep,
        top_unstable_clusters=top_unstable,
    )


def export_arch_validation_json(path: Path, analysis: ArchValidationAnalysis) -> None:
    payload = {
        "schema": VALIDATION_SCHEMA,
        "summary": analysis.summary,
        "threshold_sweep": analysis.threshold_sweep,
        "top_unstable_clusters": analysis.top_unstable_clusters,
        "variants": [asdict(v) for v in analysis.pilot.variants],
        "clusters": [
            {
                **asdict(c),
                "cluster_flip_patterns": {
                    "baseline_hybrid_none": c.hybrid_none_flips_baseline,
                    "pilot_hybrid_none": c.hybrid_none_flips_pilot,
                    "baseline_memory_rag": c.memory_rag_flips_baseline,
                    "pilot_memory_rag": c.memory_rag_flips_pilot,
                },
            }
            for c in analysis.pilot.clusters
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
