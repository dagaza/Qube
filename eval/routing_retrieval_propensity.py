"""
Shadow continuous retrieval propensity model.

Replaces binary recall_fusion_triggered with a smooth propensity score and
simulates continuous retrieval routing offline. Does NOT modify production routing.
"""
from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.router_evaluation import normalize_route

PROPENSITY_SCHEMA = "qube.routing_retrieval_propensity.v1"

DEFAULT_WEIGHTS: dict[str, float] = {
    "w1": 0.30,
    "w2": 0.25,
    "w3": 0.20,
    "w4": 0.15,
    "w5": 0.10,
}
DEFAULT_A1 = 5.0
DEFAULT_T_NONE = 0.35
DEFAULT_DELTA = 0.10
OVERLAP_PENALTY_FACTOR = 0.5

T_NONE_GRID: tuple[float, ...] = (0.30, 0.35, 0.40)
DELTA_GRID: tuple[float, ...] = (0.08, 0.10, 0.12)


@dataclass
class PropensityWeights:
    w1: float = 0.30
    w2: float = 0.25
    w3: float = 0.20
    w4: float = 0.15
    w5: float = 0.10
    a1: float = DEFAULT_A1

    @classmethod
    def from_dict(cls, data: dict[str, float] | None) -> PropensityWeights:
        if not data:
            return cls()
        return cls(
            w1=float(data.get("w1", 0.30)),
            w2=float(data.get("w2", 0.25)),
            w3=float(data.get("w3", 0.20)),
            w4=float(data.get("w4", 0.15)),
            w5=float(data.get("w5", 0.10)),
            a1=float(data.get("a1", DEFAULT_A1)),
        )


@dataclass
class PropensityThresholds:
    t_none: float = DEFAULT_T_NONE
    delta: float = DEFAULT_DELTA


@dataclass
class VariantPropensityRecord:
    case_id: str
    variant_id: str
    category: str
    original_route: str
    shadow_route: str
    recall_fusion_triggered: bool
    retrieval_propensity_score: float
    p_memory: float
    p_rag: float
    p_hybrid: float
    route_agreement: bool
    retrieval_agreement: bool
    chat_score: float
    confidence_margin: float
    score_separation: float
    embedding_similarity: float = 0.0


@dataclass
class ClusterPropensityReport:
    cluster_id: str
    case_id: str
    category: str
    propensity_variance: float
    binary_vs_continuous_flip_reduction: float
    retrieval_continuity_score: float
    canonical_alignment_improvement: float
    recall_fusion_flip_count_baseline: int
    recall_fusion_flip_count_shadow: int
    hybrid_none_flips_baseline: int
    hybrid_none_flips_shadow: int
    instability_type_proxy: str = ""


@dataclass
class PropensityAnalysis:
    summary: dict[str, Any]
    variants: list[VariantPropensityRecord] = field(default_factory=list)
    clusters: list[ClusterPropensityReport] = field(default_factory=list)


def _sigmoid(x: float, a: float = 1.0) -> float:
    z = max(-20.0, min(20.0, a * x))
    return 1.0 / (1.0 + math.exp(-z))


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _token_jaccard(a: str, b: str) -> float:
    ta = set((a or "").lower().split())
    tb = set((b or "").lower().split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _follow_up_boost(category: str, perturbation_type: str) -> float:
    if perturbation_type == "deixis":
        return 0.9
    if category == "follow_up":
        return 0.8
    return 0.0


def _discourse_continuation_signal(category: str, perturbation_type: str) -> float:
    if perturbation_type == "deixis":
        return 0.7
    if category == "follow_up":
        return 0.5
    return 0.0


def _retrieval_affinities(
    route: str,
    memory_hits: int,
    rag_hits: int,
) -> tuple[float, float]:
    route_n = normalize_route(route)
    memory_aff = 0.35
    rag_aff = 0.35
    if route_n == "memory":
        memory_aff = 0.85
    elif route_n == "rag":
        rag_aff = 0.85
    elif route_n == "hybrid":
        memory_aff = 0.75
        rag_aff = 0.75
    elif route_n == "web":
        memory_aff = 0.20
        rag_aff = 0.20
    if memory_hits > 0:
        memory_aff = max(memory_aff, 0.55 + 0.1 * min(memory_hits, 3))
    if rag_hits > 0:
        rag_aff = max(rag_aff, 0.55 + 0.1 * min(rag_hits, 3))
    return _clamp(memory_aff), _clamp(rag_aff)


def compute_retrieval_propensity_score(
    *,
    top_score: float,
    second_best_score: float,
    chat_score: float,
    confidence_margin: float,
    follow_up_boost: float,
    discourse_signal: float,
    weights: PropensityWeights,
) -> float:
    sep = top_score - second_best_score
    margin_term = _clamp(1.0 - confidence_margin)
    raw = (
        weights.w1 * _sigmoid(sep, weights.a1)
        + weights.w2 * _clamp(1.0 - chat_score)
        + weights.w3 * margin_term
        + weights.w4 * _clamp(follow_up_boost)
        + weights.w5 * _clamp(discourse_signal)
    )
    return round(_clamp(raw), 4)


def compute_retrieval_probabilities(
    propensity: float,
    *,
    memory_affinity: float,
    rag_affinity: float,
) -> tuple[float, float, float]:
    p_memory = _clamp(propensity * memory_affinity)
    p_rag = _clamp(propensity * rag_affinity)
    overlap = p_memory * p_rag * OVERLAP_PENALTY_FACTOR
    p_hybrid = _clamp(p_memory + p_rag - overlap)
    return round(p_memory, 4), round(p_rag, 4), round(p_hybrid, 4)


def decide_shadow_route(
    propensity: float,
    p_memory: float,
    p_rag: float,
    *,
    thresholds: PropensityThresholds,
    original_route: str,
) -> str:
    if propensity < thresholds.t_none:
        return "none"
    if p_memory > p_rag + thresholds.delta:
        return "memory"
    if p_rag > p_memory + thresholds.delta:
        return "rag"
    if p_memory > 0.05 or p_rag > 0.05:
        return "hybrid"
    return normalize_route(original_route)


def _majority_route(routes: list[str]) -> str:
    if not routes:
        return "none"
    return Counter(normalize_route(r) for r in routes).most_common(1)[0][0]


def _is_cluster_stable(routes: list[str]) -> bool:
    return len(set(normalize_route(r) for r in routes)) <= 1 if routes else True


def _count_hybrid_none(routes: list[str], anchor: str) -> int:
    anchor_n = normalize_route(anchor)
    return sum(
        1
        for r in routes
        if {normalize_route(r), anchor_n} == {"none", "hybrid"}
    )


def _retrieval_continuity(variants: list[Any], shadow_routes: list[str]) -> float:
    if not variants:
        return 1.0
    base_hits = [v.memory_hits + v.rag_hits + v.web_hits > 0 for v in variants]
    shadow_retrieval = [normalize_route(r) != "none" for r in shadow_routes]
    agree = sum(1 for b, s in zip(base_hits, shadow_retrieval) if b == s)
    return agree / len(variants)


def _propensity_for_variant(
    variant: Any,
    case_report: Any,
    *,
    weights: PropensityWeights,
    thresholds: PropensityThresholds,
) -> VariantPropensityRecord:
    category = case_report.category
    follow_up = _follow_up_boost(category, variant.perturbation_type)
    discourse = _discourse_continuation_signal(category, variant.perturbation_type)
    embed_sim = _token_jaccard(variant.text, case_report.base_prompt)

    propensity = compute_retrieval_propensity_score(
        top_score=variant.top_score,
        second_best_score=getattr(variant, "second_best_score", 0.0),
        chat_score=variant.chat_score,
        confidence_margin=variant.confidence_margin,
        follow_up_boost=follow_up,
        discourse_signal=discourse,
        weights=weights,
    )
    mem_aff, rag_aff = _retrieval_affinities(
        variant.execution_route,
        variant.memory_hits,
        variant.rag_hits,
    )
    p_mem, p_rag, p_hyb = compute_retrieval_probabilities(
        propensity,
        memory_affinity=mem_aff,
        rag_affinity=rag_aff,
    )
    original = normalize_route(variant.execution_route)
    shadow = decide_shadow_route(
        propensity,
        p_mem,
        p_rag,
        thresholds=thresholds,
        original_route=original,
    )
    base_hits = variant.memory_hits + variant.rag_hits + variant.web_hits > 0
    shadow_hits = shadow != "none"

    return VariantPropensityRecord(
        case_id=case_report.case_id,
        variant_id=variant.variant_id,
        category=category,
        original_route=original,
        shadow_route=shadow,
        recall_fusion_triggered=variant.recall_fusion_triggered,
        retrieval_propensity_score=propensity,
        p_memory=p_mem,
        p_rag=p_rag,
        p_hybrid=p_hyb,
        route_agreement=original == shadow,
        retrieval_agreement=base_hits == shadow_hits,
        chat_score=round(variant.chat_score, 4),
        confidence_margin=round(variant.confidence_margin, 4),
        score_separation=round(variant.top_score - getattr(variant, "second_best_score", 0.0), 4),
        embedding_similarity=round(embed_sim, 4),
    )


def _evaluate_config(
    perturbation_analysis: Any,
    *,
    weights: PropensityWeights,
    thresholds: PropensityThresholds,
) -> tuple[list[VariantPropensityRecord], list[ClusterPropensityReport], dict[str, float]]:
    all_variants: list[VariantPropensityRecord] = []
    cluster_reports: list[ClusterPropensityReport] = []

    fusion_unstable_baseline = 0
    fusion_resolved_shadow = 0
    baseline_hybrid_none = 0
    shadow_hybrid_none = 0
    baseline_route_flips = 0
    shadow_route_flips = 0
    retrieval_loss = 0
    total_with_hits = 0
    propensity_scores: list[float] = []
    propensity_variances: list[float] = []
    instability_scores: list[float] = []

    for case_report in perturbation_analysis.cases:
        records = [
            _propensity_for_variant(v, case_report, weights=weights, thresholds=thresholds)
            for v in case_report.variants
        ]
        all_variants.extend(records)

        base_routes = [r.original_route for r in records]
        shadow_routes = [r.shadow_route for r in records]
        anchor = _majority_route(base_routes)
        canon = anchor

        propensity_vals = [r.retrieval_propensity_score for r in records]
        prop_var = statistics.pvariance(propensity_vals) if len(propensity_vals) > 1 else 0.0
        propensity_variances.append(prop_var)
        propensity_scores.extend(propensity_vals)
        instability_scores.append(1.0 - case_report.route_consistency_score)

        fusion_flags = {r.recall_fusion_triggered for r in records}
        fusion_flip_baseline = 1 if len(fusion_flags) > 1 else 0
        fusion_flip_shadow = 0 if _is_cluster_stable(shadow_routes) else 1
        if fusion_flip_baseline:
            fusion_unstable_baseline += 1
            if fusion_flip_shadow == 0:
                fusion_resolved_shadow += 1

        hn_b = _count_hybrid_none(base_routes, anchor)
        hn_s = _count_hybrid_none(shadow_routes, anchor)
        baseline_hybrid_none += hn_b
        shadow_hybrid_none += hn_s

        for r in records:
            if r.original_route != canon:
                baseline_route_flips += 1
            if r.shadow_route != canon:
                shadow_route_flips += 1
            if r.original_route != "none" and (r.p_memory + r.p_rag) > 0:
                total_with_hits += 1
                if r.shadow_route == "none":
                    retrieval_loss += 1

        base_agree = sum(1 for r in records if r.original_route == canon) / len(records)
        shadow_agree = sum(1 for r in records if r.shadow_route == canon) / len(records)

        flip_red = 0.0
        if fusion_flip_baseline and fusion_flip_shadow == 0:
            flip_red = 1.0

        cluster_reports.append(
            ClusterPropensityReport(
                cluster_id=f"pert_{case_report.case_id}",
                case_id=case_report.case_id,
                category=case_report.category,
                propensity_variance=round(prop_var, 4),
                binary_vs_continuous_flip_reduction=flip_red,
                retrieval_continuity_score=round(
                    _retrieval_continuity(case_report.variants, shadow_routes), 4
                ),
                canonical_alignment_improvement=round(shadow_agree - base_agree, 4),
                recall_fusion_flip_count_baseline=fusion_flip_baseline,
                recall_fusion_flip_count_shadow=fusion_flip_shadow,
                hybrid_none_flips_baseline=hn_b,
                hybrid_none_flips_shadow=hn_s,
                instability_type_proxy=case_report.stability_label,
            )
        )

    unstable_baseline = 0
    unstable_shadow = 0
    for case_report in perturbation_analysis.cases:
        base_routes = [normalize_route(v.execution_route) for v in case_report.variants]
        shadow_routes = [
            r.shadow_route
            for r in all_variants
            if r.case_id == case_report.case_id
        ]
        if not _is_cluster_stable(base_routes):
            unstable_baseline += 1
        if not _is_cluster_stable(shadow_routes):
            unstable_shadow += 1

    fusion_reduction = (
        fusion_resolved_shadow / fusion_unstable_baseline if fusion_unstable_baseline else 0.0
    )

    flip_reduction = (
        (baseline_route_flips - shadow_route_flips) / baseline_route_flips
        if baseline_route_flips
        else 0.0
    )
    cluster_instability_reduction = (
        (unstable_baseline - unstable_shadow) / unstable_baseline
        if unstable_baseline
        else 0.0
    )
    retrieval_loss_proxy = retrieval_loss / total_with_hits if total_with_hits else 0.0

    corr = 0.0
    if len(propensity_variances) > 1 and len(instability_scores) > 1:
        try:
            corr = statistics.correlation(propensity_variances, instability_scores)
        except statistics.StatisticsError:
            corr = 0.0

    metrics = {
        "binary_vs_continuous_flip_reduction_pct": round(flip_reduction, 4),
        "recall_fusion_flip_reduction_pct": round(fusion_reduction, 4),
        "hybrid_none_flip_reduction_pct": round(
            (baseline_hybrid_none - shadow_hybrid_none) / baseline_hybrid_none
            if baseline_hybrid_none
            else 0.0,
            4,
        ),
        "instability_reduction_proxy": round(cluster_instability_reduction, 4),
        "retrieval_loss_proxy": round(retrieval_loss_proxy, 4),
        "route_agreement_rate": round(
            sum(1 for v in all_variants if v.route_agreement) / len(all_variants)
            if all_variants
            else 0.0,
            4,
        ),
        "retrieval_agreement_rate": round(
            sum(1 for v in all_variants if v.retrieval_agreement) / len(all_variants)
            if all_variants
            else 0.0,
            4,
        ),
        "propensity_instability_correlation": round(corr, 4),
        "fusion_unstable_clusters_baseline": fusion_unstable_baseline,
        "fusion_unstable_clusters_resolved": fusion_resolved_shadow,
    }
    return all_variants, cluster_reports, metrics


def _sweep_thresholds(
    perturbation_analysis: Any,
    weights: PropensityWeights,
) -> tuple[PropensityThresholds, dict[str, float], list[VariantPropensityRecord], list[ClusterPropensityReport]]:
    best_thresholds = PropensityThresholds()
    best_metrics: dict[str, float] = {}
    best_variants: list[VariantPropensityRecord] = []
    best_clusters: list[ClusterPropensityReport] = []
    best_score = float("-inf")

    for t_none in T_NONE_GRID:
        for delta in DELTA_GRID:
            thresholds = PropensityThresholds(t_none=t_none, delta=delta)
            variants, clusters, metrics = _evaluate_config(
                perturbation_analysis,
                weights=weights,
                thresholds=thresholds,
            )
            score = (
                metrics["recall_fusion_flip_reduction_pct"]
                + metrics["instability_reduction_proxy"]
                - 2.0 * metrics["retrieval_loss_proxy"]
            )
            if score > best_score:
                best_score = score
                best_thresholds = thresholds
                best_metrics = metrics
                best_variants = variants
                best_clusters = clusters

    return best_thresholds, best_metrics, best_variants, best_clusters


def analyze_retrieval_propensity(
    perturbation_analysis: Any,
    *,
    weights: PropensityWeights | None = None,
    hysteresis_summary: dict[str, Any] | None = None,
) -> PropensityAnalysis:
    """Run shadow continuous retrieval propensity analysis."""
    w = weights or PropensityWeights()
    best_thresholds, metrics, variants, clusters = _sweep_thresholds(
        perturbation_analysis, w
    )

    propensity_scores = [v.retrieval_propensity_score for v in variants]
    variance_by_category: dict[str, list[float]] = defaultdict(list)
    for v in variants:
        variance_by_category[v.category].append(v.retrieval_propensity_score)

    category_variance = {
        cat: round(statistics.pvariance(vals) if len(vals) > 1 else 0.0, 4)
        for cat, vals in variance_by_category.items()
    }

    failed_categories: dict[str, int] = defaultdict(int)
    for cr in clusters:
        if cr.canonical_alignment_improvement < 0 and cr.propensity_variance > 0.01:
            failed_categories[cr.category] += 1

    hysteresis_cmp = {}
    if hysteresis_summary:
        hysteresis_cmp = {
            "hysteresis_flip_reduction_rate": hysteresis_summary.get("flip_reduction_rate"),
            "hysteresis_stability_gain": hysteresis_summary.get("stability_gain"),
            "hysteresis_hybrid_none_reduction": hysteresis_summary.get(
                "hybrid_none_flip_reduction"
            ),
            "hysteresis_retrieval_loss_delta": hysteresis_summary.get(
                "retrieval_consistency_delta"
            ),
            "propensity_vs_hysteresis_instability_reduction": (
                metrics["instability_reduction_proxy"]
                - float(hysteresis_summary.get("stability_gain", 0.0) or 0.0)
            ),
        }

    interpretation = _build_interpretation(metrics, hysteresis_cmp)

    summary = {
        "avg_propensity_score": round(
            statistics.mean(propensity_scores) if propensity_scores else 0.0, 4
        ),
        "variance_by_category": category_variance,
        "binary_vs_continuous_flip_reduction_pct": metrics["binary_vs_continuous_flip_reduction_pct"],
        "recall_fusion_flip_reduction_pct": metrics["recall_fusion_flip_reduction_pct"],
        "hybrid_none_flip_reduction_pct": metrics["hybrid_none_flip_reduction_pct"],
        "retrieval_loss_proxy": metrics["retrieval_loss_proxy"],
        "instability_reduction_proxy": metrics["instability_reduction_proxy"],
        "route_agreement_rate": metrics["route_agreement_rate"],
        "retrieval_agreement_rate": metrics["retrieval_agreement_rate"],
        "propensity_instability_correlation": metrics["propensity_instability_correlation"],
        "best_weight_set": asdict(w),
        "best_thresholds": {
            "T_none": best_thresholds.t_none,
            "delta": best_thresholds.delta,
        },
        "fusion_unstable_clusters_baseline": metrics["fusion_unstable_clusters_baseline"],
        "fusion_unstable_clusters_resolved": metrics["fusion_unstable_clusters_resolved"],
        "hysteresis_comparison": hysteresis_cmp,
        "failed_categories": dict(failed_categories),
        "interpretation": interpretation,
    }

    return PropensityAnalysis(summary=summary, variants=variants, clusters=clusters)


def _build_interpretation(
    metrics: dict[str, float],
    hysteresis_cmp: dict[str, Any],
) -> str:
    fusion_red = metrics.get("recall_fusion_flip_reduction_pct", 0.0)
    retr_loss = metrics.get("retrieval_loss_proxy", 0.0)
    inst_red = metrics.get("instability_reduction_proxy", 0.0)

    if fusion_red > 0.5 and retr_loss < 0.15:
        return (
            "Yes — continuous propensity modeling largely eliminates recall-fusion flip "
            "instability while preserving retrieval coverage better than binary gating."
        )
    if fusion_red > 0.3 and inst_red > 0.3:
        if retr_loss < float(hysteresis_cmp.get("hysteresis_retrieval_loss_delta", 1.0) or 1.0):
            return (
                "Partially — continuous modeling reduces binary recall-fusion instability and "
                "preserves retrieval better than threshold/hysteresis approaches. "
                "Instability is fundamentally tied to discrete gating."
            )
        return (
            "Continuous modeling stabilizes routes but retrieval tradeoffs remain. "
            "Binary recall-fusion gating is a major structural contributor."
        )
    return (
        "Limited benefit — recall-fusion flips persist under continuous modeling, suggesting "
        "instability is not solely caused by binary gating discreteness."
    )


def export_propensity_json(path: Path, analysis: PropensityAnalysis) -> None:
    payload = {
        "schema": PROPENSITY_SCHEMA,
        "summary": analysis.summary,
        "clusters": [asdict(c) for c in analysis.clusters],
        "variants": [asdict(v) for v in analysis.variants],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
