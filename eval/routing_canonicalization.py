"""
Shadow routing canonicalization learner.

Learns stable canonical routes per perturbation cluster and sweeps simple
decision-boundary thresholds to estimate repairability of routing instability.
Does NOT modify production routing.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.router_evaluation import normalize_route

CANONICALIZATION_SCHEMA = "qube.routing_canonicalization.v1"

DOMINANT_ROUTE_PCT = 0.60
BOUNDARY_MARGIN_MAX = 0.10
BOUNDARY_SEP_MIN = 0.05

T_CHAT_GRID: tuple[float, ...] = (0.60, 0.65, 0.70, 0.75, 0.80)
T_MARGIN_LOW_GRID: tuple[float, ...] = (0.05, 0.08, 0.10, 0.12, 0.15)
T_SEP_GRID: tuple[float, ...] = (0.03, 0.05, 0.07, 0.08, 0.10)

_INSTABILITY_TYPES: tuple[str, ...] = (
    "stable",
    "purely_ambiguous",
    "boundary_instability",
    "retrieval_noise_instability",
    "recall_fusion_instability",
)


@dataclass
class VariantRecord:
    variant_id: str
    case_id: str
    execution_route: str
    chat_score: float
    confidence_margin: float
    top_score: float
    second_best_score: float
    memory_hits: int = 0
    rag_hits: int = 0
    web_hits: int = 0
    recall_fusion_triggered: bool = False
    perturbation_type: str = ""

    @property
    def score_separation(self) -> float:
        return self.top_score - self.second_best_score

    @property
    def has_retrieval_hits(self) -> bool:
        return (self.memory_hits + self.rag_hits + self.web_hits) > 0


@dataclass
class PerturbationCluster:
    cluster_id: str
    case_id: str
    base_prompt: str
    category: str
    variants: list[VariantRecord]
    stability_cluster_id: str = ""
    stability_cluster_cases: list[str] = field(default_factory=list)


@dataclass
class ClusterCanonicalReport:
    cluster_id: str
    case_id: str
    category: str
    canonical_route_majority: str
    canonical_route_weighted: str
    instability_type: str
    variant_count: int
    route_distribution: dict[str, int]
    canonical_agreement_baseline: float
    canonical_inconsistent: bool


@dataclass
class CanonicalizationAnalysis:
    summary: dict[str, Any]
    clusters: list[ClusterCanonicalReport] = field(default_factory=list)
    tradeoff_curve: list[dict[str, Any]] = field(default_factory=list)
    ambiguous_clusters: list[dict[str, Any]] = field(default_factory=list)


def canonical_route_majority(variants: list[VariantRecord]) -> str:
    if not variants:
        return "none"
    counts = Counter(normalize_route(v.execution_route) for v in variants)
    return counts.most_common(1)[0][0]


def canonical_route_weighted(variants: list[VariantRecord]) -> str:
    if not variants:
        return "none"
    weights: dict[str, float] = defaultdict(float)
    for v in variants:
        route = normalize_route(v.execution_route)
        weights[route] += max(v.score_separation, 0.0)
    if not weights or max(weights.values()) <= 0.0:
        return canonical_route_majority(variants)
    return max(weights.items(), key=lambda item: item[1])[0]


def apply_shadow_boundary(
    variant: VariantRecord,
    *,
    t_chat: float,
    t_margin_low: float,
    t_sep: float,
) -> str:
    """Shadow decision boundary — may force CHAT when scores are borderline."""
    router_route = normalize_route(variant.execution_route)
    if variant.chat_score > t_chat:
        return "none"
    if variant.confidence_margin < t_margin_low:
        return "none"
    if variant.score_separation < t_sep:
        return "none"
    return router_route


def classify_instability(
    variants: list[VariantRecord],
    canonical_majority: str,
) -> str:
    if not variants:
        return "stable"

    routes = [normalize_route(v.execution_route) for v in variants]
    route_counts = Counter(routes)
    total = len(routes)
    dominant_pct = route_counts.most_common(1)[0][1] / total

    fusion_flags = {v.recall_fusion_triggered for v in variants}
    if len(fusion_flags) > 1:
        return "recall_fusion_instability"

    hits_flags = [v.has_retrieval_hits for v in variants]
    if len(set(routes)) == 1 and len(set(hits_flags)) > 1:
        return "retrieval_noise_instability"

    if len(set(routes)) >= 2 and dominant_pct <= DOMINANT_ROUTE_PCT:
        return "purely_ambiguous"

    if len(set(routes)) >= 2 and dominant_pct > DOMINANT_ROUTE_PCT:
        for v in variants:
            if normalize_route(v.execution_route) != canonical_majority:
                if (
                    v.confidence_margin < BOUNDARY_MARGIN_MAX
                    or v.score_separation < BOUNDARY_SEP_MIN
                ):
                    return "boundary_instability"

    return "stable"


def _cluster_route_consistency(routes: list[str]) -> float:
    if not routes:
        return 1.0
    unique = len(set(routes))
    return 1.0 - (unique / len(routes))


def _canonical_agreement(variants: list[VariantRecord], canonical: str) -> float:
    if not variants:
        return 1.0
    matches = sum(1 for v in variants if normalize_route(v.execution_route) == canonical)
    return matches / len(variants)


def _is_cluster_stable(routes: list[str]) -> bool:
    return len(set(routes)) <= 1 if routes else True


def _evaluate_thresholds(
    clusters: list[PerturbationCluster],
    *,
    t_chat: float,
    t_margin_low: float,
    t_sep: float,
) -> dict[str, Any]:
    baseline_flips = 0
    shadow_flips = 0
    baseline_canonical_agree = 0.0
    shadow_canonical_agree = 0.0
    retrieval_loss = 0
    total_variants = 0
    baseline_stable = 0
    shadow_stable = 0

    for cluster in clusters:
        canon = canonical_route_majority(cluster.variants)
        base_routes = [normalize_route(v.execution_route) for v in cluster.variants]
        shadow_routes = [
            apply_shadow_boundary(
                v, t_chat=t_chat, t_margin_low=t_margin_low, t_sep=t_sep
            )
            for v in cluster.variants
        ]

        if _is_cluster_stable(base_routes):
            baseline_stable += 1
        if _is_cluster_stable(shadow_routes):
            shadow_stable += 1

        for v, br, sr in zip(cluster.variants, base_routes, shadow_routes):
            total_variants += 1
            if br != canon:
                baseline_flips += 1
            if sr != canon:
                shadow_flips += 1
            if br == canon:
                baseline_canonical_agree += 1
            if sr == canon:
                shadow_canonical_agree += 1
            if v.has_retrieval_hits and sr == "none" and br != "none":
                retrieval_loss += 1

    n = len(clusters)
    unstable_baseline = n - baseline_stable
    unstable_shadow = n - shadow_stable
    flip_reduction = (
        (baseline_flips - shadow_flips) / baseline_flips if baseline_flips else 0.0
    )
    cluster_instability_reduction = (
        (unstable_baseline - unstable_shadow) / unstable_baseline
        if unstable_baseline
        else 0.0
    )
    retrieval_loss_pct = retrieval_loss / total_variants if total_variants else 0.0
    baseline_agree_pct = baseline_canonical_agree / total_variants if total_variants else 0.0
    shadow_agree_pct = shadow_canonical_agree / total_variants if total_variants else 0.0

    return {
        "T_chat": t_chat,
        "T_margin_low": t_margin_low,
        "T_sep": t_sep,
        "flip_reduction_pct": round(flip_reduction, 4),
        "cluster_instability_reduction_pct": round(cluster_instability_reduction, 4),
        "retrieval_loss_pct": round(retrieval_loss_pct, 4),
        "canonical_agreement_baseline": round(baseline_agree_pct, 4),
        "canonical_agreement_shadow": round(shadow_agree_pct, 4),
        "canonical_agreement_gain": round(shadow_agree_pct - baseline_agree_pct, 4),
        "clusters_stable_baseline": baseline_stable,
        "clusters_stable_shadow": shadow_stable,
        "stability_gain_clusters": shadow_stable - baseline_stable,
        "baseline_flips": baseline_flips,
        "shadow_flips": shadow_flips,
        "retrieval_loss_count": retrieval_loss,
        "score": round(
            cluster_instability_reduction
            - 2.0 * retrieval_loss_pct,
            4,
        ),
    }


def _sweep_thresholds(clusters: list[PerturbationCluster]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    curve: list[dict[str, Any]] = []
    best: Optional[dict[str, Any]] = None
    for t_chat in T_CHAT_GRID:
        for t_margin in T_MARGIN_LOW_GRID:
            for t_sep in T_SEP_GRID:
                row = _evaluate_thresholds(
                    clusters,
                    t_chat=t_chat,
                    t_margin_low=t_margin,
                    t_sep=t_sep,
                )
                curve.append(row)
                if best is None or row["score"] > best["score"]:
                    best = row
    assert best is not None
    return best, curve


def _instability_reclassification(
    cluster_reports: list[ClusterCanonicalReport],
    clusters: list[PerturbationCluster],
    best: dict[str, Any],
) -> dict[str, float]:
    case_to_cluster = {c.case_id: c for c in clusters}
    boundary_total = 0
    boundary_resolved = 0
    retrieval_noise_total = 0
    retrieval_noise_unchanged = 0

    for report in cluster_reports:
        cluster = case_to_cluster.get(report.case_id)
        if cluster is None:
            continue

        if report.instability_type == "boundary_instability":
            boundary_total += 1
            shadow_routes = [
                apply_shadow_boundary(
                    v,
                    t_chat=best["T_chat"],
                    t_margin_low=best["T_margin_low"],
                    t_sep=best["T_sep"],
                )
                for v in cluster.variants
            ]
            if _is_cluster_stable(shadow_routes):
                boundary_resolved += 1

        if report.instability_type == "retrieval_noise_instability":
            retrieval_noise_total += 1
            base_routes = [normalize_route(v.execution_route) for v in cluster.variants]
            if _is_cluster_stable(base_routes):
                retrieval_noise_unchanged += 1

    return {
        "boundary_instability_resolved_pct": (
            boundary_resolved / boundary_total if boundary_total else 0.0
        ),
        "retrieval_noise_unchanged_pct": (
            retrieval_noise_unchanged / retrieval_noise_total if retrieval_noise_total else 1.0
        ),
        "boundary_instability_total": boundary_total,
        "boundary_instability_resolved": boundary_resolved,
        "retrieval_noise_total": retrieval_noise_total,
    }


def build_clusters_from_perturbation(
    perturbation_analysis: Any,
    *,
    stability_clusters: Optional[dict[str, Any]] = None,
) -> list[PerturbationCluster]:
    """Build perturbation clusters from a ``RoutePerturbationAnalysis`` object."""
    case_to_stab: dict[str, dict[str, Any]] = {}
    if stability_clusters:
        for sc in stability_clusters.get("clusters") or []:
            for case_id in sc.get("cases") or []:
                case_to_stab[case_id] = sc

    out: list[PerturbationCluster] = []
    for case_report in perturbation_analysis.cases:
        variants = [
            VariantRecord(
                variant_id=v.variant_id,
                case_id=case_report.case_id,
                execution_route=v.execution_route,
                chat_score=v.chat_score,
                confidence_margin=v.confidence_margin,
                top_score=v.top_score,
                second_best_score=getattr(v, "second_best_score", 0.0),
                memory_hits=v.memory_hits,
                rag_hits=v.rag_hits,
                web_hits=v.web_hits,
                recall_fusion_triggered=v.recall_fusion_triggered,
                perturbation_type=v.perturbation_type,
            )
            for v in case_report.variants
        ]
        stab = case_to_stab.get(case_report.case_id, {})
        out.append(
            PerturbationCluster(
                cluster_id=f"pert_{case_report.case_id}",
                case_id=case_report.case_id,
                base_prompt=case_report.base_prompt,
                category=case_report.category,
                variants=variants,
                stability_cluster_id=str(stab.get("cluster_id") or ""),
                stability_cluster_cases=list(stab.get("cases") or []),
            )
        )
    return out


def load_perturbation_from_json(path: Path) -> Any:
    """Load perturbation cases from ``route_perturbation_cases.json``."""
    from eval.routing_perturbation import (
        CasePerturbationReport,
        RoutePerturbationAnalysis,
        VariantRunResult,
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    cases = []
    for c in data.get("cases") or []:
        variants = [VariantRunResult(**v) for v in c.get("variants") or []]
        base = {k: v for k, v in c.items() if k != "variants"}
        cases.append(CasePerturbationReport(variants=variants, **base))
    return RoutePerturbationAnalysis(summary=data.get("summary") or {}, cases=cases)


def analyze_routing_canonicalization(
    perturbation_analysis: Any,
    *,
    stability_clusters: Optional[dict[str, Any]] = None,
) -> CanonicalizationAnalysis:
    """Run full canonicalization learner analysis (shadow mode)."""
    clusters = build_clusters_from_perturbation(
        perturbation_analysis,
        stability_clusters=stability_clusters,
    )

    cluster_reports: list[ClusterCanonicalReport] = []
    instability_counts: dict[str, int] = {t: 0 for t in _INSTABILITY_TYPES}

    for cluster in clusters:
        canon_maj = canonical_route_majority(cluster.variants)
        canon_wgt = canonical_route_weighted(cluster.variants)
        inst_type = classify_instability(cluster.variants, canon_maj)
        instability_counts[inst_type] = instability_counts.get(inst_type, 0) + 1

        routes = [normalize_route(v.execution_route) for v in cluster.variants]
        route_dist = dict(Counter(routes))
        agree = _canonical_agreement(cluster.variants, canon_maj)

        cluster_reports.append(
            ClusterCanonicalReport(
                cluster_id=cluster.cluster_id,
                case_id=cluster.case_id,
                category=cluster.category,
                canonical_route_majority=canon_maj,
                canonical_route_weighted=canon_wgt,
                instability_type=inst_type,
                variant_count=len(cluster.variants),
                route_distribution=route_dist,
                canonical_agreement_baseline=round(agree, 4),
                canonical_inconsistent=canon_maj != canon_wgt,
            )
        )

    best, tradeoff_curve = _sweep_thresholds(clusters)
    reclass = _instability_reclassification(cluster_reports, clusters, best)

    baseline_stable = sum(
        1
        for c in clusters
        if _is_cluster_stable([normalize_route(v.execution_route) for v in c.variants])
    )

    ambiguous = [
        {
            "cluster_id": r.cluster_id,
            "case_id": r.case_id,
            "category": r.category,
            "canonical_route_majority": r.canonical_route_majority,
            "canonical_route_weighted": r.canonical_route_weighted,
            "route_distribution": r.route_distribution,
            "instability_type": r.instability_type,
        }
        for r in cluster_reports
        if r.instability_type == "purely_ambiguous" or r.canonical_inconsistent
    ]

    n = len(clusters)
    boundary_noise_pct = (
        instability_counts.get("boundary_instability", 0) / n if n else 0.0
    )
    ambiguous_pct = instability_counts.get("purely_ambiguous", 0) / n if n else 0.0
    repairable_pct = boundary_noise_pct
    recall_fusion_pct = instability_counts.get("recall_fusion_instability", 0) / n if n else 0.0
    unfixable_pct = ambiguous_pct + recall_fusion_pct

    summary = {
        "clusters_total": n,
        "clusters_stable_baseline": baseline_stable,
        "clusters_stable_shadow_best": best["clusters_stable_shadow"],
        "best_threshold_set": {
            "T_chat": best["T_chat"],
            "T_margin_low": best["T_margin_low"],
            "T_sep": best["T_sep"],
        },
        "metrics": {
            "flip_reduction_pct": best["flip_reduction_pct"],
            "cluster_instability_reduction_pct": best["cluster_instability_reduction_pct"],
            "retrieval_loss_pct": best["retrieval_loss_pct"],
            "canonical_agreement_gain": best["canonical_agreement_gain"],
            "stability_gain_clusters": best["stability_gain_clusters"],
            "boundary_noise_pct": round(boundary_noise_pct, 4),
            "semantic_ambiguity_pct": round(ambiguous_pct, 4),
            "recall_fusion_instability_pct": round(recall_fusion_pct, 4),
            "repairable_instability_pct": round(repairable_pct, 4),
            "unfixable_instability_pct": round(unfixable_pct, 4),
            "instability_reclassification": reclass,
        },
        "instability_type_breakdown": instability_counts,
        "avg_canonical_agreement_baseline": round(
            statistics.mean(r.canonical_agreement_baseline for r in cluster_reports)
            if cluster_reports
            else 0.0,
            4,
        ),
        "interpretation": _build_interpretation(
            repairable_pct,
            ambiguous_pct,
            best["cluster_instability_reduction_pct"],
            recall_fusion_pct,
            best["retrieval_loss_pct"],
        ),
    }

    return CanonicalizationAnalysis(
        summary=summary,
        clusters=cluster_reports,
        tradeoff_curve=tradeoff_curve,
        ambiguous_clusters=ambiguous,
    )


def _build_interpretation(
    repairable_pct: float,
    ambiguous_pct: float,
    cluster_instability_reduction: float,
    recall_fusion_pct: float,
    retrieval_loss: float,
) -> str:
    if cluster_instability_reduction > 0.5 and retrieval_loss < 0.10:
        if recall_fusion_pct < 0.3 and repairable_pct > ambiguous_pct:
            return (
                "Yes — instability is mostly boundary jitter (threshold noise). "
                "Simple boundary tuning repairs most cluster instability with low retrieval loss."
            )
        return (
            "Partially repairable — boundary tuning stabilizes routes, but recall-fusion "
            "variance across paraphrases is a major root cause."
        )
    if cluster_instability_reduction > 0.5 and retrieval_loss >= 0.10:
        return (
            "Boundary tuning can stabilize cluster routes, but at significant retrieval cost. "
            "Root cause is likely recall-fusion/threshold interaction, not pure semantic ambiguity."
        )
    if recall_fusion_pct > 0.5 or ambiguous_pct > repairable_pct:
        return (
            "No — simple threshold tuning does not sufficiently stabilize clusters. "
            "Recall-fusion sensitivity and/or semantic ambiguity dominate."
        )
    return (
        "Mixed instability profile: boundary noise and recall-fusion/ambiguity both contribute. "
        "Threshold tuning helps partially but deeper router redesign may be needed."
    )


def export_canonicalization_json(
    path: Path,
    analysis: CanonicalizationAnalysis,
) -> None:
    payload = {
        "schema": CANONICALIZATION_SCHEMA,
        "summary": analysis.summary,
        "clusters": [asdict(c) for c in analysis.clusters],
        "tradeoff_curve": analysis.tradeoff_curve,
        "ambiguous_clusters": analysis.ambiguous_clusters,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
