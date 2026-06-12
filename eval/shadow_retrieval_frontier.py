"""
2D retrieval frontier analysis for decomposed shadow propensity axes.

Maps semantic vs contextual threshold space with perturbation invariance overlays.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

from core.shadow_retrieval_policy import (
    PolicyThresholds,
    PolicyWeights,
    PropensityAxes,
    ShadowRetrievalState,
    axes_activate_retrieval,
    compute_retrieval_policy,
    decompose_propensity_axes,
    _is_retrieval_route,
    _normalize_route,
)
from eval.routing_perturbation import CasePerturbationReport, VariantRunResult
from eval.shadow_retrieval_policy_eval import _discourse_signal, _follow_up_strength

FRONTIER_SCHEMA = "qube.shadow_retrieval_frontier_2d.v1"


@dataclass
class FrontierVariant:
    case_id: str
    category: str
    baseline_route: str
    baseline_had_hits: bool
    expected_retrieval: bool
    recall_fusion: bool
    axes: PropensityAxes
    state: ShadowRetrievalState


@dataclass
class FrontierCell:
    t_semantic: float
    t_contextual: float
    recall_hits: float
    precision_hits: float
    recall_label: float
    precision_label: float
    shadow_retrieval_rate: float
    suppression_count: int
    false_activation_count: int
    unstable_clusters: int
    unstable_rate: float
    route_consistency_mean: float
    retrieval_consistency_mean: float
    instability_reduction: float
    fusion_flip_resolved_rate: float
    semantic_only_rate: float
    contextual_only_rate: float
    both_axes_rate: float


@dataclass
class Frontier2DAnalysis:
    summary: dict[str, Any]
    grid: list[FrontierCell] = field(default_factory=list)
    reference_points: list[dict[str, Any]] = field(default_factory=list)
    axis_distribution: dict[str, Any] = field(default_factory=dict)


def load_frontier_variants(
    perturbation_cases_path: Path,
    *,
    expected_by_id: Optional[dict[str, str]] = None,
) -> tuple[list[FrontierVariant], list[CasePerturbationReport]]:
    """Load perturbation artifacts into frontier evaluation records."""
    pert = json.loads(perturbation_cases_path.read_text(encoding="utf-8"))
    cases: list[CasePerturbationReport] = []
    variants: list[FrontierVariant] = []

    for c in pert["cases"]:
        vlist = [
            VariantRunResult(**{k: v[k] for k in VariantRunResult.__dataclass_fields__ if k in v})
            for v in c["variants"]
        ]
        case = CasePerturbationReport(
            **{k: c[k] for k in CasePerturbationReport.__dataclass_fields__ if k in c}
            | {"variants": vlist}
        )
        cases.append(case)
        exp = _normalize_route((expected_by_id or {}).get(case.case_id, "none"))
        for variant in vlist:
            state = ShadowRetrievalState(
                baseline_route=_normalize_route(variant.execution_route),
                decision={
                    "chat_score": variant.chat_score,
                    "confidence_margin": variant.confidence_margin,
                    "top_score": variant.top_score,
                    "second_best_score": variant.second_best_score,
                    "recall_fusion": variant.recall_fusion_triggered,
                },
                prompt=variant.text,
                chat_score=variant.chat_score,
                confidence_margin=variant.confidence_margin,
                top_score=variant.top_score,
                second_best_score=variant.second_best_score,
                follow_up_strength=_follow_up_strength(case.category, variant.perturbation_type),
                discourse_continuation=_discourse_signal(case.category, variant.perturbation_type),
                baseline_recall_fusion=variant.recall_fusion_triggered,
            )
            variants.append(
                FrontierVariant(
                    case_id=case.case_id,
                    category=case.category,
                    baseline_route=_normalize_route(variant.execution_route),
                    baseline_had_hits=(
                        variant.memory_hits + variant.rag_hits + variant.web_hits > 0
                    ),
                    expected_retrieval=_is_retrieval_route(exp),
                    recall_fusion=variant.recall_fusion_triggered,
                    axes=decompose_propensity_axes(state),
                    state=state,
                )
            )
    return variants, cases


def _route_consistency(routes: list[str]) -> float:
    if not routes:
        return 1.0
    return 1.0 - (len(set(routes)) / len(routes))


def _retrieval_consistency(flags: list[bool]) -> float:
    if not flags:
        return 1.0
    if len(flags) < 2:
        return 1.0
    return 1.0 - statistics.pvariance([1 if f else 0 for f in flags])


def evaluate_2d_cell(
    variants: list[FrontierVariant],
    cases: list[CasePerturbationReport],
    *,
    t_semantic: float,
    t_contextual: float,
) -> FrontierCell:
    """Evaluate one operating point in semantic×contextual threshold space."""
    n = len(variants)
    thresholds = PolicyThresholds(
        t_semantic=t_semantic,
        t_contextual=t_contextual,
    )

    records: list[dict[str, Any]] = []
    for v in variants:
        activated = axes_activate_retrieval(
            v.axes, t_semantic=t_semantic, t_contextual=t_contextual
        )
        sem_gate = t_semantic > 0.0 and v.axes.semantic_norm >= t_semantic
        ctx_gate = t_contextual > 0.0 and v.axes.contextual_norm >= t_contextual
        sem_only = sem_gate and not ctx_gate
        ctx_only = ctx_gate and not sem_gate
        both = sem_gate and ctx_gate

        if activated:
            st = replace(v.state, thresholds=thresholds)
            shadow_route = compute_retrieval_policy(st)["shadow_decision"]
        else:
            shadow_route = "none"

        shadow_ret = _is_retrieval_route(shadow_route)
        records.append({
            "case_id": v.case_id,
            "baseline_had_hits": v.baseline_had_hits,
            "expected_retrieval": v.expected_retrieval,
            "shadow_route": shadow_route,
            "shadow_ret": shadow_ret,
            "recall_fusion": v.recall_fusion,
            "sem_only": sem_only,
            "ctx_only": ctx_only,
            "both": both,
            "activated": activated,
        })

    warranted = [r for r in records if r["baseline_had_hits"]]
    shadow_retrieved = [r for r in records if r["shadow_ret"]]
    label_pos = [r for r in records if r["expected_retrieval"]]

    recall_hits = (
        sum(1 for r in warranted if r["shadow_ret"]) / len(warranted)
        if warranted
        else 0.0
    )
    precision_hits = (
        sum(1 for r in shadow_retrieved if r["baseline_had_hits"]) / len(shadow_retrieved)
        if shadow_retrieved
        else 0.0
    )
    recall_label = (
        sum(1 for r in label_pos if r["shadow_ret"]) / len(label_pos)
        if label_pos
        else 0.0
    )
    precision_label = (
        sum(1 for r in shadow_retrieved if r["expected_retrieval"]) / len(shadow_retrieved)
        if shadow_retrieved
        else 0.0
    )

    suppress = sum(1 for r in records if r["baseline_had_hits"] and not r["shadow_ret"])
    false_act = sum(1 for r in records if r["shadow_ret"] and not r["baseline_had_hits"])

    by_case_routes: dict[str, list[str]] = defaultdict(list)
    by_case_ret: dict[str, list[bool]] = defaultdict(list)
    for r in records:
        by_case_routes[r["case_id"]].append(r["shadow_route"])
        by_case_ret[r["case_id"]].append(r["shadow_ret"])

    baseline_unstable = sum(
        1
        for c in cases
        if len({_normalize_route(v.execution_route) for v in c.variants}) > 1
    )
    unstable = sum(1 for routes in by_case_routes.values() if len(set(routes)) > 1)
    route_cons = [
        _route_consistency(by_case_routes[cid]) for cid in by_case_routes
    ]
    retr_cons = [
        _retrieval_consistency(by_case_ret[cid]) for cid in by_case_ret
    ]

    fusion_flip_cases = 0
    fusion_resolved = 0
    for case in cases:
        flags = {v.recall_fusion_triggered for v in case.variants}
        if len(flags) <= 1:
            continue
        fusion_flip_cases += 1
        shadows = by_case_routes.get(case.case_id, [])
        if shadows and len(set(shadows)) == 1:
            fusion_resolved += 1

    return FrontierCell(
        t_semantic=round(t_semantic, 3),
        t_contextual=round(t_contextual, 3),
        recall_hits=round(recall_hits, 4),
        precision_hits=round(precision_hits, 4),
        recall_label=round(recall_label, 4),
        precision_label=round(precision_label, 4),
        shadow_retrieval_rate=round(len(shadow_retrieved) / n, 4),
        suppression_count=suppress,
        false_activation_count=false_act,
        unstable_clusters=unstable,
        unstable_rate=round(unstable / len(cases), 4),
        route_consistency_mean=round(statistics.mean(route_cons), 4),
        retrieval_consistency_mean=round(statistics.mean(retr_cons), 4),
        instability_reduction=round(
            (baseline_unstable - unstable) / baseline_unstable if baseline_unstable else 0.0,
            4,
        ),
        fusion_flip_resolved_rate=round(
            fusion_resolved / fusion_flip_cases if fusion_flip_cases else 0.0,
            4,
        ),
        semantic_only_rate=round(sum(1 for r in records if r["sem_only"]) / n, 4),
        contextual_only_rate=round(sum(1 for r in records if r["ctx_only"]) / n, 4),
        both_axes_rate=round(sum(1 for r in records if r["both"]) / n, 4),
    )


def sweep_2d_frontier(
    variants: list[FrontierVariant],
    cases: list[CasePerturbationReport],
    *,
    semantic_steps: int = 11,
    contextual_steps: int = 11,
) -> Frontier2DAnalysis:
    """Sweep semantic×contextual threshold grid."""
    sem_vals = [round(i / (semantic_steps - 1), 2) for i in range(semantic_steps)]
    ctx_vals = [round(i / (contextual_steps - 1), 2) for i in range(contextual_steps)]

    grid = [
        evaluate_2d_cell(variants, cases, t_semantic=ts, t_contextual=tc)
        for ts in sem_vals
        for tc in ctx_vals
    ]

    sem_scores = [v.axes.semantic_norm for v in variants]
    ctx_scores = [v.axes.contextual_norm for v in variants]

    # Reference operating points
    refs = [
        ("permissive_both_disabled", evaluate_2d_cell(
            variants, cases, t_semantic=0.0, t_contextual=0.0
        )),
        ("semantic_only_T0.60", evaluate_2d_cell(
            variants, cases, t_semantic=0.60, t_contextual=0.0
        )),
        ("contextual_only_T0.50", evaluate_2d_cell(
            variants, cases, t_semantic=0.0, t_contextual=0.50
        )),
        ("or_gate_T0.60_T0.50", evaluate_2d_cell(
            variants, cases, t_semantic=0.60, t_contextual=0.50
        )),
    ]

    baseline_unstable = sum(
        1
        for c in cases
        if len({_normalize_route(v.execution_route) for v in c.variants}) > 1
    )
    baseline_route_cons = statistics.mean([
        1.0 - len(set(_normalize_route(v.execution_route) for v in c.variants)) / len(c.variants)
        for c in cases
    ])

    # Pareto-ish frontier on hit recall vs precision with stability overlay
    candidates = [
        c for c in grid if c.suppression_count == 0 and c.unstable_clusters <= 5
    ]
    candidates.sort(key=lambda c: (-c.recall_hits, -c.precision_hits, -c.route_consistency_mean))

    summary = {
        "n_variants": len(variants),
        "n_cases": len(cases),
        "baseline_unstable_clusters": baseline_unstable,
        "baseline_route_consistency_mean": round(baseline_route_cons, 4),
        "semantic_axis": {
            "min": round(min(sem_scores), 4),
            "p25": round(sorted(sem_scores)[len(sem_scores) // 4], 4),
            "median": round(statistics.median(sem_scores), 4),
            "p75": round(sorted(sem_scores)[3 * len(sem_scores) // 4], 4),
            "max": round(max(sem_scores), 4),
        },
        "contextual_axis": {
            "min": round(min(ctx_scores), 4),
            "p25": round(sorted(ctx_scores)[len(ctx_scores) // 4], 4),
            "median": round(statistics.median(ctx_scores), 4),
            "p75": round(sorted(ctx_scores)[3 * len(ctx_scores) // 4], 4),
            "max": round(max(ctx_scores), 4),
        },
        "zero_suppression_stable_band": [
            {
                "t_semantic": c.t_semantic,
                "t_contextual": c.t_contextual,
                "recall_hits": c.recall_hits,
                "precision_hits": c.precision_hits,
                "unstable_clusters": c.unstable_clusters,
                "route_consistency_mean": c.route_consistency_mean,
            }
            for c in candidates[:12]
        ],
        "interpretation_axes": (
            "Semantic axis = router margin + chat avoidance + confidence margin. "
            "Contextual axis = follow-up + discourse continuation. "
            "Activation OR-gate: retrieve if either normalized axis clears its threshold."
        ),
    }

    return Frontier2DAnalysis(
        summary=summary,
        grid=grid,
        reference_points=[
            {"label": label, **asdict(cell)} for label, cell in refs
        ],
        axis_distribution=summary["semantic_axis"] | {
            "contextual": summary["contextual_axis"]
        },
    )


def export_frontier_json(path: Path, analysis: Frontier2DAnalysis) -> None:
    payload = {
        "schema": FRONTIER_SCHEMA,
        "summary": analysis.summary,
        "reference_points": analysis.reference_points,
        "grid": [asdict(c) for c in analysis.grid],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_frontier_from_run_dir(
    run_dir: Path,
    *,
    corpus_path: Optional[Path] = None,
) -> Frontier2DAnalysis:
    from core.router_evaluation import load_corpus

    pert_path = run_dir / "route_perturbation_cases.json"
    if corpus_path is None:
        corpus_path = Path("eval/router_corpus/v1_baseline.json")
    _, corpus_cases = load_corpus(corpus_path)
    expected_by_id = {c.id: c.expected_route for c in corpus_cases}
    variants, cases = load_frontier_variants(pert_path, expected_by_id=expected_by_id)
    return sweep_2d_frontier(variants, cases)
