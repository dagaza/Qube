"""
Continuous recall-fusion architectural pilot (shadow → routing candidate).

Elevates the continuous retrieval propensity model to a measurable routing
candidate and compares it against baseline, canonicalization, and hysteresis.
Does NOT modify CognitiveRouterV4 or production routing.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.router_evaluation import normalize_route
from eval.routing_canonicalization import (
    VariantRecord,
    apply_shadow_boundary,
    canonical_route_majority,
)
from eval.routing_hysteresis import HysteresisConfig, apply_hysteresis_shadow_route
from eval.routing_retrieval_propensity import (
    PropensityThresholds,
    PropensityWeights,
    _count_hybrid_none,
    _is_cluster_stable,
    _majority_route,
    _propensity_for_variant,
    _retrieval_continuity,
    _sweep_thresholds,
    decide_shadow_route,
)

PILOT_SCHEMA = "qube.routing_continuous_pilot.v1"


@dataclass
class PilotVariantRecord:
    case_id: str
    variant_id: str
    category: str
    baseline_route: str
    pilot_route: str
    propensity_score: float
    p_memory: float
    p_rag: float
    p_hybrid: float
    canonical_route: str
    hysteresis_route: str
    canonical_shadow_route: str
    recall_fusion_triggered: bool
    pilot_vs_baseline_agreement: bool
    pilot_vs_canonical_agreement: bool
    pilot_vs_hysteresis_agreement: bool
    pilot_vs_canonical_shadow_agreement: bool


@dataclass
class PilotClusterReport:
    cluster_id: str
    case_id: str
    category: str
    canonical_route_majority: str
    pilot_instability_reduction: float
    retrieval_loss_estimate: float
    canonical_alignment_improvement: float
    hysteresis_alignment_delta: float
    canonical_shadow_alignment_delta: float
    hybrid_none_flips_baseline: int
    hybrid_none_flips_pilot: int
    hybrid_none_flips_hysteresis: int
    hybrid_none_flips_canonical_shadow: int
    memory_rag_flips_baseline: int
    memory_rag_flips_pilot: int
    pilot_stable: bool
    baseline_stable: bool


@dataclass
class PilotAnalysis:
    summary: dict[str, Any]
    variants: list[PilotVariantRecord] = field(default_factory=list)
    clusters: list[PilotClusterReport] = field(default_factory=list)


def decide_pilot_route(
    propensity: float,
    p_memory: float,
    p_rag: float,
    *,
    thresholds: PropensityThresholds,
    baseline_route: str,
) -> str:
    """Pilot routing candidate — same decision surface as shadow propensity layer."""
    return decide_shadow_route(
        propensity,
        p_memory,
        p_rag,
        thresholds=thresholds,
        original_route=baseline_route,
    )


def _count_memory_rag_flips(routes: list[str], anchor: str) -> int:
    anchor_n = normalize_route(anchor)
    return sum(
        1
        for r in routes
        if {normalize_route(r), anchor_n} == {"memory", "rag"}
    )


def _retrieval_loss_estimate(
    variants: list[Any],
    pilot_routes: list[str],
) -> float:
    total = 0
    loss = 0
    for v, pr in zip(variants, pilot_routes):
        had_hits = v.memory_hits + v.rag_hits + v.web_hits > 0
        if had_hits:
            total += 1
            if normalize_route(pr) == "none":
                loss += 1
    return loss / total if total else 0.0


def _category_retrieval_coverage(
    variants: list[PilotVariantRecord],
) -> dict[str, dict[str, float]]:
    by_cat: dict[str, dict[str, int]] = defaultdict(
        lambda: {"baseline_retrieval": 0, "pilot_retrieval": 0, "total": 0}
    )
    for v in variants:
        bucket = by_cat[v.category]
        bucket["total"] += 1
        if v.baseline_route != "none":
            bucket["baseline_retrieval"] += 1
        if v.pilot_route != "none":
            bucket["pilot_retrieval"] += 1
    return {
        cat: {
            "baseline_coverage": b["baseline_retrieval"] / b["total"] if b["total"] else 0.0,
            "pilot_coverage": b["pilot_retrieval"] / b["total"] if b["total"] else 0.0,
            "delta": (
                (b["pilot_retrieval"] - b["baseline_retrieval"]) / b["total"]
                if b["total"]
                else 0.0
            ),
        }
        for cat, b in by_cat.items()
    }


def analyze_continuous_pilot_routing(
    perturbation_analysis: Any,
    *,
    weights: PropensityWeights | None = None,
    hysteresis_summary: dict[str, Any] | None = None,
    canonicalization_summary: dict[str, Any] | None = None,
    propensity_summary: dict[str, Any] | None = None,
) -> PilotAnalysis:
    """
    Run continuous recall-fusion pilot analysis on perturbation clusters.

    Reuses propensity sweep for best thresholds unless ``propensity_summary``
    already contains ``best_thresholds``.
    """
    w = weights or PropensityWeights()
    best_thresholds, _metrics, _prop_variants, _prop_clusters = _sweep_thresholds(
        perturbation_analysis, w
    )

    if propensity_summary and propensity_summary.get("best_thresholds"):
        bt = propensity_summary["best_thresholds"]
        best_thresholds = PropensityThresholds(
            t_none=float(bt.get("T_none", best_thresholds.t_none)),
            delta=float(bt.get("delta", best_thresholds.delta)),
        )

    canon_thresholds = None
    if canonicalization_summary and canonicalization_summary.get("best_threshold_set"):
        cbt = canonicalization_summary["best_threshold_set"]
        canon_thresholds = (
            float(cbt.get("T_chat", 0.70)),
            float(cbt.get("T_margin_low", 0.10)),
            float(cbt.get("T_sep", 0.05)),
        )

    hyst_cfg = HysteresisConfig()
    all_variants: list[PilotVariantRecord] = []
    cluster_reports: list[PilotClusterReport] = []

    unstable_baseline = 0
    unstable_pilot = 0
    unstable_hysteresis = 0
    unstable_canon_shadow = 0
    canonical_flips = 0
    pilot_flips = 0
    baseline_hybrid_none = 0
    pilot_hybrid_none = 0

    for case_report in perturbation_analysis.cases:
        variant_objs = case_report.variants
        base_routes = [normalize_route(v.execution_route) for v in variant_objs]
        anchor = _majority_route(base_routes)
        canon_route = canonical_route_majority(
            [
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
                )
                for v in variant_objs
            ]
        )

        prop_records = [
            _propensity_for_variant(v, case_report, weights=w, thresholds=best_thresholds)
            for v in variant_objs
        ]

        pilot_routes: list[str] = []
        hyst_routes: list[str] = []
        canon_shadow_routes: list[str] = []
        pilot_records: list[PilotVariantRecord] = []

        for v, pr in zip(variant_objs, prop_records):
            pilot = pr.shadow_route
            pilot_routes.append(pilot)

            hyst = apply_hysteresis_shadow_route(
                v.execution_route,
                v.confidence_margin,
                v.chat_score,
                v.top_score,
                getattr(v, "second_best_score", 0.0),
                previous_route=anchor,
                config=hyst_cfg,
            )
            hyst_routes.append(hyst)

            if canon_thresholds:
                t_chat, t_margin, t_sep = canon_thresholds
                canon_shadow = apply_shadow_boundary(
                    VariantRecord(
                        variant_id=v.variant_id,
                        case_id=case_report.case_id,
                        execution_route=v.execution_route,
                        chat_score=v.chat_score,
                        confidence_margin=v.confidence_margin,
                        top_score=v.top_score,
                        second_best_score=getattr(v, "second_best_score", 0.0),
                    ),
                    t_chat=t_chat,
                    t_margin_low=t_margin,
                    t_sep=t_sep,
                )
            else:
                canon_shadow = normalize_route(v.execution_route)
            canon_shadow_routes.append(canon_shadow)

            pilot_records.append(
                PilotVariantRecord(
                    case_id=case_report.case_id,
                    variant_id=v.variant_id,
                    category=case_report.category,
                    baseline_route=pr.original_route,
                    pilot_route=pilot,
                    propensity_score=pr.retrieval_propensity_score,
                    p_memory=pr.p_memory,
                    p_rag=pr.p_rag,
                    p_hybrid=pr.p_hybrid,
                    canonical_route=canon_route,
                    hysteresis_route=hyst,
                    canonical_shadow_route=canon_shadow,
                    recall_fusion_triggered=v.recall_fusion_triggered,
                    pilot_vs_baseline_agreement=pr.original_route == pilot,
                    pilot_vs_canonical_agreement=pilot == canon_route,
                    pilot_vs_hysteresis_agreement=pilot == hyst,
                    pilot_vs_canonical_shadow_agreement=pilot == canon_shadow,
                )
            )

        all_variants.extend(pilot_records)

        base_stable = _is_cluster_stable(base_routes)
        pilot_stable = _is_cluster_stable(pilot_routes)
        if not base_stable:
            unstable_baseline += 1
        if not pilot_stable:
            unstable_pilot += 1
        if not _is_cluster_stable(hyst_routes):
            unstable_hysteresis += 1
        if not _is_cluster_stable(canon_shadow_routes):
            unstable_canon_shadow += 1

        for r in pilot_records:
            if r.baseline_route != canon_route:
                canonical_flips += 1
            if r.pilot_route != canon_route:
                pilot_flips += 1

        hn_b = _count_hybrid_none(base_routes, anchor)
        hn_p = _count_hybrid_none(pilot_routes, anchor)
        baseline_hybrid_none += hn_b
        pilot_hybrid_none += hn_p

        base_agree = sum(1 for r in pilot_records if r.baseline_route == canon_route) / len(
            pilot_records
        )
        pilot_agree = sum(1 for r in pilot_records if r.pilot_route == canon_route) / len(
            pilot_records
        )
        hyst_agree = sum(1 for r in pilot_records if r.hysteresis_route == canon_route) / len(
            pilot_records
        )
        canon_shadow_agree = sum(
            1 for r in pilot_records if r.canonical_shadow_route == canon_route
        ) / len(pilot_records)

        pilot_instability_red = 1.0 if (not base_stable and pilot_stable) else 0.0

        cluster_reports.append(
            PilotClusterReport(
                cluster_id=f"pilot_{case_report.case_id}",
                case_id=case_report.case_id,
                category=case_report.category,
                canonical_route_majority=canon_route,
                pilot_instability_reduction=pilot_instability_red,
                retrieval_loss_estimate=round(
                    _retrieval_loss_estimate(variant_objs, pilot_routes), 4
                ),
                canonical_alignment_improvement=round(pilot_agree - base_agree, 4),
                hysteresis_alignment_delta=round(pilot_agree - hyst_agree, 4),
                canonical_shadow_alignment_delta=round(pilot_agree - canon_shadow_agree, 4),
                hybrid_none_flips_baseline=hn_b,
                hybrid_none_flips_pilot=hn_p,
                hybrid_none_flips_hysteresis=_count_hybrid_none(hyst_routes, anchor),
                hybrid_none_flips_canonical_shadow=_count_hybrid_none(
                    canon_shadow_routes, anchor
                ),
                memory_rag_flips_baseline=_count_memory_rag_flips(base_routes, anchor),
                memory_rag_flips_pilot=_count_memory_rag_flips(pilot_routes, anchor),
                pilot_stable=pilot_stable,
                baseline_stable=base_stable,
            )
        )

    n_cases = len(perturbation_analysis.cases)
    instability_reduction = (
        (unstable_baseline - unstable_pilot) / unstable_baseline
        if unstable_baseline
        else 0.0
    )
    flip_reduction_vs_canonical = (
        (canonical_flips - pilot_flips) / canonical_flips if canonical_flips else 0.0
    )
    retrieval_loss = statistics.mean(
        [c.retrieval_loss_estimate for c in cluster_reports]
    ) if cluster_reports else 0.0

    propensity_scores = [v.propensity_score for v in all_variants]
    category_coverage = _category_retrieval_coverage(all_variants)

    hyst_cmp = {}
    if hysteresis_summary:
        hyst_cmp = {
            "hysteresis_instability_reduction": hysteresis_summary.get("stability_gain"),
            "hysteresis_hybrid_none_reduction": hysteresis_summary.get(
                "hybrid_none_flip_reduction"
            ),
            "hysteresis_retrieval_loss_delta": hysteresis_summary.get(
                "retrieval_consistency_delta"
            ),
        }

    canon_cmp = {}
    if canonicalization_summary:
        metrics = canonicalization_summary.get("metrics") or {}
        canon_cmp = {
            "canonicalization_instability_reduction": metrics.get(
                "cluster_instability_reduction_pct"
            ),
            "canonicalization_retrieval_loss": metrics.get("retrieval_loss_pct"),
            "canonicalization_flip_reduction": metrics.get("flip_reduction_pct"),
        }

    interpretation = _build_interpretation(
        instability_reduction,
        retrieval_loss,
        unstable_baseline,
        unstable_pilot,
        hysteresis_summary,
        canonicalization_summary,
    )

    summary = {
        "avg_propensity_score": round(
            statistics.mean(propensity_scores) if propensity_scores else 0.0, 4
        ),
        "instability_reduction_pct": round(instability_reduction, 4),
        "retrieval_loss_proxy": round(retrieval_loss, 4),
        "flip_reduction_vs_canonical": round(flip_reduction_vs_canonical, 4),
        "hybrid_none_flip_reduction_pct": round(
            (baseline_hybrid_none - pilot_hybrid_none) / baseline_hybrid_none
            if baseline_hybrid_none
            else 0.0,
            4,
        ),
        "clusters_unstable_baseline": unstable_baseline,
        "clusters_unstable_pilot": unstable_pilot,
        "clusters_unstable_hysteresis": unstable_hysteresis,
        "clusters_unstable_canonical_shadow": unstable_canon_shadow,
        "pilot_vs_baseline_agreement_rate": round(
            sum(1 for v in all_variants if v.pilot_vs_baseline_agreement) / len(all_variants)
            if all_variants
            else 0.0,
            4,
        ),
        "pilot_vs_canonical_agreement_rate": round(
            sum(1 for v in all_variants if v.pilot_vs_canonical_agreement) / len(all_variants)
            if all_variants
            else 0.0,
            4,
        ),
        "retrieval_continuity_score": round(
            statistics.mean(
                [
                    _retrieval_continuity(
                        next(
                            c.variants
                            for c in perturbation_analysis.cases
                            if c.case_id == cr.case_id
                        ),
                        [v.pilot_route for v in all_variants if v.case_id == cr.case_id],
                    )
                    for cr in cluster_reports
                ]
            )
            if cluster_reports
            else 0.0,
            4,
        ),
        "best_weight_set": asdict(w),
        "best_thresholds": {
            "T_none": best_thresholds.t_none,
            "delta": best_thresholds.delta,
        },
        "per_category_retrieval_coverage": category_coverage,
        "hysteresis_comparison": hyst_cmp,
        "canonicalization_comparison": canon_cmp,
        "interpretation": interpretation,
        "pilot_resolves_all_unstable": (
            unstable_baseline > 0 and unstable_pilot == 0
        ),
    }

    return PilotAnalysis(summary=summary, variants=all_variants, clusters=cluster_reports)


def _build_interpretation(
    instability_reduction: float,
    retrieval_loss: float,
    unstable_baseline: int,
    unstable_pilot: int,
    hysteresis_summary: dict[str, Any] | None,
    canonicalization_summary: dict[str, Any] | None,
) -> str:
    resolves_all = unstable_baseline > 0 and unstable_pilot == 0
    low_loss = retrieval_loss < 0.05

    if resolves_all and low_loss:
        return (
            "Pilot routing stabilizes all previously unstable clusters with minimal "
            "retrieval loss — continuous recall-fusion is a viable architectural candidate."
        )

    canon_loss = None
    if canonicalization_summary:
        canon_loss = (canonicalization_summary.get("metrics") or {}).get("retrieval_loss_pct")

    if instability_reduction > 0.5 and low_loss:
        if canon_loss is not None and retrieval_loss < float(canon_loss):
            return (
                "Pilot routing reduces instability while preserving retrieval coverage "
                "better than canonicalization/hysteresis threshold approaches."
            )
        return (
            "Pilot routing materially reduces instability with acceptable retrieval cost. "
            "Binary recall-fusion gating is the structural oscillation source."
        )

    if hysteresis_summary and instability_reduction > float(
        hysteresis_summary.get("stability_gain", 0.0) or 0.0
    ):
        return (
            "Continuous pilot outperforms hysteresis on cluster stability — supports "
            "probabilistic routing over discrete boundary buffers."
        )

    return (
        "Pilot shows partial benefit — further weight/threshold tuning or deeper "
        "integration may be needed before production pilot."
    )


def export_pilot_json(path: Path, analysis: PilotAnalysis) -> None:
    payload = {
        "schema": PILOT_SCHEMA,
        "summary": analysis.summary,
        "clusters": [asdict(c) for c in analysis.clusters],
        "variants": [asdict(v) for v in analysis.variants],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
