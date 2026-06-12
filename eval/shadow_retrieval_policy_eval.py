"""
Offline aggregation of shadow retrieval policy vs perturbation artifacts.

Mirrors LLMWorker shadow policy for run.json / report without live execution.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.shadow_retrieval_policy import (
    PolicyThresholds,
    PolicyWeights,
    ShadowRetrievalState,
    compute_retrieval_policy,
    _is_retrieval_route,
    _normalize_route,
)

EVAL_SCHEMA = "qube.shadow_retrieval_policy_eval.v1"


@dataclass
class ShadowPolicyEvalAnalysis:
    summary: dict[str, Any]
    variant_records: list[dict[str, Any]] = field(default_factory=list)
    regression_cases: list[dict[str, Any]] = field(default_factory=list)
    improvement_cases: list[dict[str, Any]] = field(default_factory=list)


def _follow_up_strength(category: str, perturbation_type: str) -> float:
    if perturbation_type == "deixis":
        return 0.9
    if category == "follow_up":
        return float(0.8)
    return 0.0


def _discourse_signal(category: str, perturbation_type: str) -> float:
    if perturbation_type == "deixis":
        return 0.7
    if category == "follow_up":
        return 0.5
    return 0.0


def analyze_shadow_retrieval_policy(
    perturbation_analysis: Any,
    *,
    weights: PolicyWeights | None = None,
    thresholds: PolicyThresholds | None = None,
    pilot_summary: dict[str, Any] | None = None,
) -> ShadowPolicyEvalAnalysis:
    """Run shadow policy analysis on perturbation variant artifacts."""
    w = weights or PolicyWeights()
    t = thresholds or PolicyThresholds()
    if pilot_summary and pilot_summary.get("best_thresholds"):
        bt = pilot_summary["best_thresholds"]
        t = PolicyThresholds(
            t_none=float(bt.get("T_none", t.t_none)),
            delta=float(bt.get("delta", t.delta)),
        )

    records: list[dict[str, Any]] = []
    regression_cases: list[dict[str, Any]] = []
    improvement_cases: list[dict[str, Any]] = []

    fusion_flip_clusters = 0
    fusion_resolved = 0
    hybrid_none_baseline = 0
    hybrid_none_shadow = 0
    diverged = 0
    suppress_regression = 0
    stability_improve = 0

    by_category: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"variants": 0, "diverged": 0, "fusion_flips": 0, "suppress": 0}
    )

    for case_report in perturbation_analysis.cases:
        fusion_flags = set()
        routes_baseline = []
        routes_shadow = []

        for variant in case_report.variants:
            baseline = _normalize_route(variant.execution_route)
            state = ShadowRetrievalState(
                baseline_route=baseline,
                decision={
                    "chat_score": variant.chat_score,
                    "confidence_margin": variant.confidence_margin,
                    "top_score": variant.top_score,
                    "second_best_score": getattr(variant, "second_best_score", 0.0),
                    "recall_fusion": variant.recall_fusion_triggered,
                },
                prompt=variant.text,
                chat_score=variant.chat_score,
                confidence_margin=variant.confidence_margin,
                top_score=variant.top_score,
                second_best_score=getattr(variant, "second_best_score", 0.0),
                follow_up_strength=_follow_up_strength(
                    case_report.category, variant.perturbation_type
                ),
                discourse_continuation=_discourse_signal(
                    case_report.category, variant.perturbation_type
                ),
                baseline_recall_fusion=variant.recall_fusion_triggered,
                weights=w,
                thresholds=t,
            )
            policy = compute_retrieval_policy(state)
            shadow = policy["shadow_decision"]
            routes_baseline.append(baseline)
            routes_shadow.append(shadow)
            fusion_flags.add(variant.recall_fusion_triggered)

            route_div = baseline != shadow
            if route_div:
                diverged += 1

            had_hits = variant.memory_hits + variant.rag_hits + variant.web_hits > 0
            shadow_suppresses = _is_retrieval_route(baseline) and shadow == "none" and had_hits
            shadow_improves_case = (
                variant.recall_fusion_triggered and shadow == "none" and baseline == "hybrid"
            )

            rec = {
                "case_id": case_report.case_id,
                "variant_id": variant.variant_id,
                "category": case_report.category,
                "baseline_route": baseline,
                "shadow_route": shadow,
                "pilot_route": shadow,
                "propensity_score": policy["retrieval_propensity_score"],
                "P_memory": policy["P_memory"],
                "P_rag": policy["P_rag"],
                "P_hybrid": policy["P_hybrid"],
                "agreement_vs_baseline": not route_div,
                "agreement_vs_canonical": False,
                "baseline_recall_fusion": policy["baseline_recall_fusion"],
                "route_divergence": route_div,
                "retrieval_hits": had_hits,
            }
            records.append(rec)

            cat = by_category[case_report.category]
            cat["variants"] += 1
            if route_div:
                cat["diverged"] += 1
            if shadow_suppresses:
                suppress_regression += 1
                cat["suppress"] += 1
                regression_cases.append({
                    **rec,
                    "reason": "shadow_suppresses_baseline_retrieval_with_hits",
                })
            if shadow_improves_case:
                stability_improve += 1
                improvement_cases.append({
                    **rec,
                    "reason": "shadow_stabilizes_recall_fusion_hybrid",
                })

        if len(fusion_flags) > 1:
            fusion_flip_clusters += 1
            if len(set(routes_shadow)) == 1:
                fusion_resolved += 1

        from collections import Counter

        anchor = (
            Counter(routes_baseline).most_common(1)[0][0] if routes_baseline else "none"
        )
        for b, s in zip(routes_baseline, routes_shadow):
            if {b, anchor} == {"none", "hybrid"} or {s, anchor} == {"none", "hybrid"}:
                if b != anchor:
                    hybrid_none_baseline += 1
                if s != anchor:
                    hybrid_none_shadow += 1

    n = len(records)
    propensity_scores = [r["propensity_score"] for r in records]
    unstable_baseline = sum(
        1
        for c in perturbation_analysis.cases
        if len({ _normalize_route(v.execution_route) for v in c.variants }) > 1
    )
    unstable_shadow = 0
    for case_report in perturbation_analysis.cases:
        shadows = [
            r["shadow_route"]
            for r in records
            if r["case_id"] == case_report.case_id
        ]
        if len(set(shadows)) > 1:
            unstable_shadow += 1

    for cat, stats in by_category.items():
        v = stats["variants"] or 1
        stats["divergence_rate"] = stats["diverged"] / v
        stats["suppression_rate"] = stats["suppress"] / v

    summary = {
        "avg_propensity_score": round(
            statistics.mean(propensity_scores) if propensity_scores else 0.0, 4
        ),
        "divergence_rate": round(diverged / n if n else 0.0, 4),
        "recall_fusion_eliminated_rate": round(
            fusion_resolved / fusion_flip_clusters if fusion_flip_clusters else 0.0, 4
        ),
        "recall_fusion_flip_rate": round(
            fusion_flip_clusters / len(perturbation_analysis.cases)
            if perturbation_analysis.cases
            else 0.0,
            4,
        ),
        "shadow_replacement_rate": round(diverged / n if n else 0.0, 4),
        "hybrid_stability_gain": round(
            (hybrid_none_baseline - hybrid_none_shadow) / hybrid_none_baseline
            if hybrid_none_baseline
            else 0.0,
            4,
        ),
        "hybrid_none_suppression_rate": round(
            (hybrid_none_baseline - hybrid_none_shadow) / hybrid_none_baseline
            if hybrid_none_baseline
            else 0.0,
            4,
        ),
        "none_reduction_rate": round(
            sum(1 for r in records if r["baseline_route"] == "none" and r["shadow_route"] != "none")
            / max(1, sum(1 for r in records if r["baseline_route"] == "none")),
            4,
        ),
        "retrieval_coverage_delta": round(
            (stability_improve - suppress_regression) / n if n else 0.0, 4
        ),
        "retrieval_stability_gain_estimate": round(
            (unstable_baseline - unstable_shadow) / unstable_baseline
            if unstable_baseline
            else 0.0,
            4,
        ),
        "instability_reduction_pct": round(
            (unstable_baseline - unstable_shadow) / unstable_baseline
            if unstable_baseline
            else 0.0,
            4,
        ),
        "regression_suppression_count": suppress_regression,
        "stability_improvement_count": stability_improve,
        "best_thresholds": {
            "T_none": t.t_none,
            "delta": t.delta,
            "weights": asdict(w),
        },
        "by_category": dict(by_category),
        "interpretation": _interpret({
            "fusion_resolved": fusion_resolved,
            "fusion_flip_clusters": fusion_flip_clusters,
            "suppress_regression": suppress_regression,
            "unstable_reduction": unstable_baseline - unstable_shadow,
        }),
    }

    return ShadowPolicyEvalAnalysis(
        summary=summary,
        variant_records=records,
        regression_cases=regression_cases[:20],
        improvement_cases=improvement_cases[:20],
    )


def _interpret(stats: dict[str, Any]) -> str:
    if stats["fusion_flip_clusters"] and stats["fusion_resolved"] == stats["fusion_flip_clusters"]:
        return (
            "Shadow policy eliminates recall-fusion cluster instability in offline replay — "
            "binary fusion appears redundant for retrieval activation."
        )
    if stats["unstable_reduction"] > 0 and stats["suppress_regression"] == 0:
        return (
            "Shadow policy reduces execution instability without suppressing baseline retrieval hits."
        )
    if stats["suppress_regression"] > stats["unstable_reduction"]:
        return (
            "Shadow policy diverges from baseline with retrieval suppression risk — "
            "tune thresholds before production pilot."
        )
    return (
        "Mixed shadow divergence — recall-fusion replacement shows partial benefit; "
        "review regression and improvement case lists."
    )


def export_shadow_policy_eval_json(path: Path, analysis: ShadowPolicyEvalAnalysis) -> None:
    payload = {
        "schema": EVAL_SCHEMA,
        "summary": analysis.summary,
        "regression_cases": analysis.regression_cases,
        "improvement_cases": analysis.improvement_cases,
        "variant_sample": analysis.variant_records[:50],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
