"""
Human-readable evaluation reports for the router eval harness.
"""
from __future__ import annotations

from typing import Any

from core.router_evaluation import RouterEvalSummary


def format_evaluation_report(
    summary: RouterEvalSummary,
    *,
    routing_stability: dict[str, Any] | None = None,
    route_perturbation: dict[str, Any] | None = None,
    routing_hysteresis: dict[str, Any] | None = None,
    routing_canonicalization: dict[str, Any] | None = None,
    retrieval_propensity: dict[str, Any] | None = None,
    continuous_pilot_routing: dict[str, Any] | None = None,
    continuous_arch_validation: dict[str, Any] | None = None,
    shadow_retrieval_policy: dict[str, Any] | None = None,
) -> str:
    """Full markdown-style report for ``--report`` mode."""
    rc = summary.retrieval_calibration or {}
    lines = [
        "# Router Evaluation Report",
        "",
        "## Accuracy",
        f"- Strict accuracy (final route): **{summary.strict_accuracy:.1%}**",
        f"- Family accuracy (final route): **{summary.family_accuracy:.1%}**",
        f"- Router accuracy: {summary.router_accuracy:.1%}",
        f"- Pre-retrieval accuracy: {summary.execution_pre_accuracy:.1%}",
        f"- Downgrade rate: {summary.downgrade_rate:.1%} ({summary.downgrade_count} cases)",
        "",
        "## Retrieval Calibration Summary",
        "",
        f"- Strict accuracy: **{summary.strict_accuracy:.1%}**",
        f"- Family accuracy: **{summary.family_accuracy:.1%}**",
        (
            "- Over-retrieval rate (CHAT leakage into retrieval): "
            f"**{rc.get('over_retrieval_rate', 0):.1%}** "
            f"({rc.get('over_retrieval_count', 0)}/{rc.get('chat_labeled_total', 0)} CHAT-labeled)"
        ),
        (
            "- Under-retrieval rate (missed retrieval): "
            f"**{rc.get('under_retrieval_rate', 0):.1%}** "
            f"({rc.get('retrieval_necessity_error_count', 0)}/"
            f"{rc.get('retrieval_expected_total', 0)} retrieval-expected)"
        ),
        (
            "- Recall-fusion share of over-retrieval: "
            f"**{rc.get('recall_fusion_over_retrieval_share', 0):.1%}** "
            f"({rc.get('recall_fusion_over_retrieval_count', 0)}/"
            f"{rc.get('over_retrieval_count', 0)})"
        ),
        (
            "- Avg chat_score (correct CHAT cases): "
            f"{rc.get('avg_chat_score_correct_chat_cases', 0):.3f}"
        ),
        (
            "- Avg chat_score (over-retrieval cases): "
            f"{rc.get('avg_chat_score_over_retrieval_cases', 0):.3f}"
        ),
        (
            "- Potential chat guard threshold (median − ε, not enforced): "
            f"{rc.get('potential_chat_guard_threshold_candidate', 0):.3f}"
        ),
        "",
        _format_over_retrieval_by_category(rc),
        "",
        _format_chat_margin_histogram(rc),
        "",
        _format_suppression_candidates(rc),
        "",
    ]
    if routing_stability:
        lines.extend([
            "## Routing Stability Analysis",
            "",
            _format_routing_stability(routing_stability),
            "",
        ])
    if route_perturbation:
        lines.extend([
            "## ROUTE PERTURBATION INVARIANCE REPORT",
            "",
            _format_route_perturbation(route_perturbation),
            "",
        ])
    if routing_hysteresis:
        lines.extend([
            "## ROUTING HYSTERESIS SIMULATION REPORT",
            "",
            _format_routing_hysteresis(routing_hysteresis),
            "",
        ])
    if routing_canonicalization:
        lines.extend([
            "## ROUTING CANONICALIZATION LEARNER REPORT",
            "",
            _format_routing_canonicalization(routing_canonicalization),
            "",
        ])
    if retrieval_propensity:
        lines.extend([
            "## RETRIEVAL PROPENSITY MODEL ANALYSIS (SHADOW CONTINUOUS LAYER)",
            "",
            _format_retrieval_propensity(retrieval_propensity),
            "",
        ])
    if continuous_pilot_routing:
        lines.extend([
            "## CONTINUOUS RECALL-FUSION PILOT REPORT",
            "",
            _format_continuous_pilot_routing(continuous_pilot_routing),
            "",
        ])
    if continuous_arch_validation:
        lines.extend([
            "## CONTINUOUS RECALL-FUSION ARCHITECTURAL VALIDATION REPORT",
            "",
            _format_continuous_arch_validation(continuous_arch_validation),
            "",
        ])
    if shadow_retrieval_policy:
        lines.extend([
            "## SHADOW LLMWORKER RETRIEVAL POLICY ANALYSIS",
            "",
            _format_shadow_retrieval_policy(shadow_retrieval_policy),
            "",
        ])
    lines.extend([
        "## Retrieval Hit Rates",
    ])

    if summary.retrieval_hit_rates:
        for cat, rate in sorted(summary.retrieval_hit_rates.items()):
            lines.append(f"- {cat}: {rate:.1%}")
    else:
        lines.append("_No retrieval hit rates (use `--with-retrieval`)._")

    lines.extend([
        "",
        "## Failure Causes",
        "",
        _format_failure_causes(summary.failure_causes),
        "",
        "## Category Breakdown",
        "",
        _format_category_table(summary.by_category),
        "",
        "## Rewrite Impact",
        "",
        _format_rewrite_impact(summary.rewrite_impact or {}),
        "",
        "## Memory Recall Analysis",
        "",
        _format_memory_analysis(summary.memory_analysis),
        "",
        "## Confusion Matrix (strict, expected → final)",
        "",
    ])
    for exp, acts in summary.confusion_matrix.items():
        act_str = ", ".join(f"{k}={v}" for k, v in sorted(acts.items()))
        lines.append(f"- {exp}: {act_str}")

    lines.append("")
    lines.append("## Interpretation Guide")
    lines.append("")
    lines.append(
        "1. **Over-retrieval rate** — CHAT-labeled prompts that activated retrieval with hits."
    )
    lines.append(
        "2. **Recall-fusion share** — fraction of over-retrieval driven by recall fusion."
    )
    lines.append(
        "3. **Chat score gap** — if over-retrieval cases have low chat_score, a margin "
        "guard may help; high chat_score suggests override/fusion not router scoring."
    )
    lines.append(
        "4. **Under-retrieval** — true missed retrieval (not label mismatch)."
    )
    lines.append(
        "5. **Suppression candidates** — high chat_score but still retrieved; best "
        "targets for guard tuning."
    )

    if summary.errors:
        lines.append("")
        lines.append(f"## Errors ({len(summary.errors)})")
        for err in summary.errors[:20]:
            lines.append(f"- {err}")

    return "\n".join(lines)


def _format_over_retrieval_by_category(rc: dict[str, Any]) -> str:
    by_cat = rc.get("over_retrieval_by_category") or {}
    if not by_cat:
        return "### Over-retrieval by category\n_No CHAT calibration categories in run._"
    lines = ["### Over-retrieval by category"]
    for cat, stats in sorted(by_cat.items()):
        lines.append(
            f"- {cat}: **{stats.get('over_retrieval_rate', 0):.1%}** "
            f"({stats.get('over_retrieval_count', 0)}/"
            f"{stats.get('chat_labeled_total', 0)})"
        )
    return "\n".join(lines)


def _format_chat_margin_histogram(rc: dict[str, Any]) -> str:
    hist = rc.get("chat_margin_histogram") or {}
    if not hist:
        return "### CHAT confidence margin (CHAT-labeled prompts)\n_No data._"
    lines = ["### CHAT confidence margin (CHAT-labeled prompts)"]
    for bucket in ("0-0.05", "0.05-0.10", "0.10-0.20", "0.20+"):
        lines.append(f"- {bucket}: {hist.get(bucket, 0)} cases")
    return "\n".join(lines)


def _format_shadow_retrieval_policy(sp: dict[str, Any]) -> str:
    if not sp:
        return "_Shadow retrieval policy not enabled (use `--shadow-retrieval-policy-analysis`)._"
    best_t = sp.get("best_thresholds") or {}
    weights = best_t.get("weights") or {}
    lines = [
        f"- Avg propensity score: **{sp.get('avg_propensity_score', 0):.3f}**",
        f"- Route divergence rate: **{sp.get('divergence_rate', 0):.1%}**",
        f"- Recall-fusion eliminated rate: **{sp.get('recall_fusion_eliminated_rate', 0):.1%}**",
        f"- Hybrid stability gain: **{sp.get('hybrid_stability_gain', 0):.1%}**",
        f"- Instability reduction: **{sp.get('instability_reduction_pct', 0):.1%}**",
        f"- Retrieval coverage delta: **{sp.get('retrieval_coverage_delta', 0):+.3f}**",
        (
            f"- Regression suppressions: {sp.get('regression_suppression_count', 0)} | "
            f"Stability improvements: {sp.get('stability_improvement_count', 0)}"
        ),
        (
            f"- Thresholds: T_none={best_t.get('T_none', 0):.2f}, "
            f"delta={best_t.get('delta', 0):.2f} | "
            f"w1={weights.get('w1', 0):.2f}, w2={weights.get('w2', 0):.2f}"
        ),
    ]
    if sp.get("interpretation"):
        lines.extend(["", f"**Interpretation:** {sp['interpretation']}"])
    by_cat = sp.get("by_category") or {}
    if by_cat:
        lines.extend(["", "### By category", ""])
        lines.append("| Category | Divergence | Suppression |")
        lines.append("|----------|------------|-------------|")
        for cat, stats in sorted(by_cat.items()):
            lines.append(
                f"| {cat} | {stats.get('divergence_rate', 0):.1%} | "
                f"{stats.get('suppression_rate', 0):.1%} |"
            )
    regress = sp.get("regression_cases") or []
    if regress:
        lines.extend(["", "### Regression check (shadow suppresses baseline retrieval)", ""])
        for row in regress[:5]:
            lines.append(
                f"- {row.get('case_id')}/{row.get('variant_id')}: "
                f"{row.get('baseline_route')} → {row.get('shadow_route')} ({row.get('reason', '')})"
            )
    improve = sp.get("improvement_cases") or []
    if improve:
        lines.extend(["", "### Stability improvements (shadow vs recall-fusion)", ""])
        for row in improve[:5]:
            lines.append(
                f"- {row.get('case_id')}/{row.get('variant_id')}: "
                f"{row.get('baseline_route')} → {row.get('shadow_route')}"
            )
    return "\n".join(lines)


def _format_continuous_arch_validation(av: dict[str, Any]) -> str:
    if not av:
        return "_Arch validation not enabled (use `--continuous-arch-validation`)._"
    best_t = av.get("best_thresholds") or {}
    matrix = av.get("comparison_matrix") or {}
    unstable = matrix.get("unstable_clusters") or {}
    inst_red = matrix.get("instability_reduction") or {}
    retr = matrix.get("retrieval_loss_proxy") or {}
    lines = [
        f"- Validation passed: **{av.get('validation_passed', False)}**",
        f"- Avg propensity score: **{av.get('avg_propensity_score', 0):.3f}**",
        f"- Instability reduction: **{av.get('instability_reduction_pct', 0):.1%}**",
        f"- Retrieval loss proxy: **{av.get('retrieval_loss_proxy', 0):.1%}**",
        f"- Retrieval continuity: **{av.get('retrieval_continuity_score', 0):.3f}**",
        (
            f"- Best thresholds: T_none={best_t.get('T_none', 0):.2f}, "
            f"delta={best_t.get('delta', 0):.2f}"
        ),
    ]
    if av.get("interpretation"):
        lines.extend(["", f"**Interpretation:** {av['interpretation']}"])
    lines.extend([
        "",
        "### Comparison matrix (unstable clusters)",
        "",
        "| Method | Unstable clusters |",
        "|--------|-------------------|",
        f"| Baseline | {unstable.get('baseline', 0)} |",
        f"| Pilot | {unstable.get('pilot', 0)} |",
        f"| Hysteresis | {unstable.get('hysteresis', 0)} |",
        f"| Canonical shadow | {unstable.get('canonical_shadow', 0)} |",
        "",
        "### Instability reduction vs alternatives",
        f"- Pilot vs baseline: **{inst_red.get('pilot_vs_baseline', 0):.1%}**",
        f"- Hysteresis stability gain: {inst_red.get('hysteresis_stability_gain', 'n/a')}",
        f"- Canonicalization reduction: {inst_red.get('canonicalization_reduction', 'n/a')}",
        "",
        "### Retrieval loss proxy",
        f"- Pilot: **{retr.get('pilot', 0):.1%}** | Canonicalization: {retr.get('canonicalization', 'n/a')}",
    ])
    flip = av.get("flip_type_summary") or {}
    if flip:
        hn = flip.get("hybrid_none") or {}
        mr = flip.get("memory_rag") or {}
        lines.extend([
            "",
            "### Flip patterns",
            "",
            "| Flip type | Baseline | Pilot | Hysteresis | Canon shadow |",
            "|-----------|----------|-------|------------|--------------|",
            (
                f"| hybrid↔none | {hn.get('baseline', 0)} | {hn.get('pilot', 0)} | "
                f"{hn.get('hysteresis', 0)} | {hn.get('canonical_shadow', 0)} |"
            ),
            f"| memory↔rag | {mr.get('baseline', 0)} | {mr.get('pilot', 0)} | — | — |",
        ])
    by_cat = av.get("by_category") or {}
    if by_cat:
        lines.extend([
            "",
            "### By corpus category",
            "",
            "| Category | Instability Δ | Retrieval loss | hybrid↔none Δ | Agreement |",
            "|----------|---------------|----------------|---------------|-----------|",
        ])
        for cat, stats in sorted(by_cat.items()):
            lines.append(
                f"| {cat} | {stats.get('instability_reduction_pct', 0):.1%} | "
                f"{stats.get('avg_retrieval_loss_estimate', 0):.3f} | "
                f"{stats.get('hybrid_none_flip_delta', 0)} | "
                f"{stats.get('pilot_vs_baseline_agreement_rate', 0):.1%} |"
            )
    sweep = av.get("threshold_sweep_top") or []
    if sweep:
        lines.extend([
            "",
            "### Threshold sweep (top candidates)",
            "",
            "| T_none | delta | instability Δ | retrieval loss |",
            "|--------|-------|-----------------|----------------|",
        ])
        for row in sweep[:5]:
            lines.append(
                f"| {row.get('T_none', 0):.2f} | {row.get('delta', 0):.2f} | "
                f"{row.get('instability_reduction_proxy', 0):.1%} | "
                f"{row.get('retrieval_loss_proxy', 0):.1%} |"
            )
    top = av.get("top_unstable_clusters") or []
    if top:
        lines.extend([
            "",
            "### Top unstable clusters (pilot)",
            "",
            "| case_id | category | baseline pattern | pilot pattern | retrieval loss |",
            "|---------|----------|------------------|---------------|----------------|",
        ])
        for row in top[:10]:
            lines.append(
                f"| {row.get('case_id', '')} | {row.get('category', '')} | "
                f"{row.get('baseline_route_pattern', '')} | "
                f"{row.get('pilot_route_pattern', '')} | "
                f"{row.get('retrieval_loss_estimate', 0):.3f} |"
            )
    return "\n".join(lines)


def _format_continuous_pilot_routing(cp: dict[str, Any]) -> str:
    if not cp:
        return "_Continuous pilot not enabled (use `--continuous-pilot-routing`)._"
    best_t = cp.get("best_thresholds") or {}
    hyst = cp.get("hysteresis_comparison") or {}
    canon = cp.get("canonicalization_comparison") or {}
    lines = [
        f"- Avg propensity score: **{cp.get('avg_propensity_score', 0):.3f}**",
        f"- Instability reduction: **{cp.get('instability_reduction_pct', 0):.1%}**",
        f"- Retrieval loss proxy: **{cp.get('retrieval_loss_proxy', 0):.1%}**",
        f"- Retrieval continuity: **{cp.get('retrieval_continuity_score', 0):.3f}**",
        (
            f"- Hybrid↔none flip reduction: "
            f"**{cp.get('hybrid_none_flip_reduction_pct', 0):.1%}**"
        ),
        (
            f"- Flip reduction vs canonical: "
            f"**{cp.get('flip_reduction_vs_canonical', 0):.1%}**"
        ),
        (
            f"- Best thresholds: T_none={best_t.get('T_none', 0):.2f}, "
            f"delta={best_t.get('delta', 0):.2f}"
        ),
        (
            f"- Resolves all unstable clusters: "
            f"**{cp.get('pilot_resolves_all_unstable', False)}**"
        ),
    ]
    if cp.get("interpretation"):
        lines.extend(["", f"**Interpretation:** {cp['interpretation']}"])
    if hyst or canon:
        lines.extend(["", "### Comparison vs hysteresis / canonicalization"])
        if hyst:
            lines.append(
                f"- Hysteresis stability gain: {hyst.get('hysteresis_instability_reduction', 0)}"
            )
            lines.append(
                f"- Hysteresis hybrid↔none reduction: "
                f"{hyst.get('hysteresis_hybrid_none_reduction', 0):.1%}"
                if isinstance(hyst.get("hysteresis_hybrid_none_reduction"), float)
                else f"- Hysteresis hybrid↔none reduction: {hyst.get('hysteresis_hybrid_none_reduction')}"
            )
        if canon:
            lines.append(
                f"- Canonicalization instability reduction: "
                f"{canon.get('canonicalization_instability_reduction', 0):.1%}"
                if isinstance(canon.get("canonicalization_instability_reduction"), float)
                else f"- Canonicalization instability reduction: {canon.get('canonicalization_instability_reduction')}"
            )
            lines.append(
                f"- Canonicalization retrieval loss: "
                f"{canon.get('canonicalization_retrieval_loss', 0):.1%}"
                if isinstance(canon.get("canonicalization_retrieval_loss"), float)
                else f"- Canonicalization retrieval loss: {canon.get('canonicalization_retrieval_loss')}"
            )
    coverage = cp.get("per_category_retrieval_coverage") or {}
    if coverage:
        lines.extend(["", "### Per-category retrieval coverage (pilot vs baseline)"])
        for cat, stats in sorted(coverage.items()):
            lines.append(
                f"- {cat}: baseline={stats.get('baseline_coverage', 0):.1%} "
                f"pilot={stats.get('pilot_coverage', 0):.1%} "
                f"delta={stats.get('delta', 0):+.1%}"
            )
    lines.extend([
        "",
        "### Flip patterns",
        "- hybrid↔none: see cluster-level `hybrid_none_flips_*` in `continuous_pilot_routing.json`",
        "- memory↔rag: see cluster-level `memory_rag_flips_*` in export",
    ])
    return "\n".join(lines)


def _format_retrieval_propensity(rp: dict[str, Any]) -> str:
    if not rp:
        return "_Retrieval propensity analysis not enabled (use `--retrieval-propensity-analysis`)._"
    best_w = rp.get("best_weight_set") or {}
    best_t = rp.get("best_thresholds") or {}
    hyst = rp.get("hysteresis_comparison") or {}
    lines = [
        f"- Avg propensity score: **{rp.get('avg_propensity_score', 0):.3f}**",
        (
            "- Recall-fusion flip reduction: "
            f"**{rp.get('recall_fusion_flip_reduction_pct', 0):.1%}**"
        ),
        (
            "- Hybrid↔none oscillation reduction: "
            f"**{rp.get('hybrid_none_flip_reduction_pct', 0):.1%}**"
        ),
        (
            "- Binary→continuous flip reduction: "
            f"**{rp.get('binary_vs_continuous_flip_reduction_pct', 0):.1%}**"
        ),
        f"- Instability reduction proxy: **{rp.get('instability_reduction_proxy', 0):.1%}**",
        f"- Retrieval loss proxy: **{rp.get('retrieval_loss_proxy', 0):.1%}**",
        (
            "- Best weights: "
            f"w1={best_w.get('w1', 0):.2f}, w2={best_w.get('w2', 0):.2f}, "
            f"w3={best_w.get('w3', 0):.2f}, w4={best_w.get('w4', 0):.2f}, "
            f"w5={best_w.get('w5', 0):.2f}"
        ),
        (
            f"- Best thresholds: T_none={best_t.get('T_none', 0):.2f}, "
            f"delta={best_t.get('delta', 0):.2f}"
        ),
    ]
    if rp.get("interpretation"):
        lines.extend(["", f"**Interpretation:** {rp['interpretation']}"])
    if hyst:
        lines.extend([
            "",
            "### Comparison vs hysteresis",
            (
                f"- Hysteresis flip reduction: "
                f"{hyst.get('hysteresis_flip_reduction_rate', 0):.1%}"
            ),
            (
                f"- Hysteresis stability gain: "
                f"{hyst.get('hysteresis_stability_gain', 0):+.3f}"
            ),
            (
                f"- Propensity instability reduction delta: "
                f"{hyst.get('propensity_vs_hysteresis_instability_reduction', 0):+.3f}"
            ),
        ])
    variance = rp.get("variance_by_category") or {}
    if variance:
        lines.extend(["", "### Propensity variance by category"])
        for cat, var in sorted(variance.items()):
            lines.append(f"- {cat}: {var:.4f}")
    failed = rp.get("failed_categories") or {}
    if failed:
        lines.extend(["", "### Categories where continuous model fails (true ambiguity proxy)"])
        for cat, count in sorted(failed.items()):
            lines.append(f"- {cat}: {count} clusters")
    return "\n".join(lines)


def _format_routing_canonicalization(rc: dict[str, Any]) -> str:
    if not rc:
        return "_Canonicalization analysis not enabled (use `--canonicalization-analysis`)._"
    best = rc.get("best_threshold_set") or {}
    metrics = rc.get("metrics") or {}
    reclass = metrics.get("instability_reclassification") or {}
    breakdown = rc.get("instability_type_breakdown") or {}
    lines = [
        f"- Clusters analyzed: **{rc.get('clusters_total', 0)}**",
        (
            f"- Stable clusters: baseline={rc.get('clusters_stable_baseline', 0)}, "
            f"shadow best={rc.get('clusters_stable_shadow_best', 0)}"
        ),
        (
            "- Best threshold set: "
            f"T_chat={best.get('T_chat', 0):.2f}, "
            f"T_margin_low={best.get('T_margin_low', 0):.2f}, "
            f"T_sep={best.get('T_sep', 0):.2f}"
        ),
        (
            f"- Cluster instability reduction: "
            f"**{metrics.get('cluster_instability_reduction_pct', 0):.1%}**"
        ),
        f"- Variant canonical flip reduction: **{metrics.get('flip_reduction_pct', 0):.1%}**",
        f"- Canonical agreement gain: **{metrics.get('canonical_agreement_gain', 0):+.3f}**",
        f"- Retrieval loss: **{metrics.get('retrieval_loss_pct', 0):.1%}**",
        (
            f"- Boundary noise share: **{metrics.get('boundary_noise_pct', 0):.1%}** | "
            f"Semantic ambiguity: **{metrics.get('semantic_ambiguity_pct', 0):.1%}**"
        ),
    ]
    if rc.get("interpretation"):
        lines.extend(["", f"**Interpretation:** {rc['interpretation']}"])
    if breakdown:
        lines.extend(["", "### Instability type breakdown"])
        for itype, count in sorted(breakdown.items()):
            lines.append(f"- {itype}: {count}")
    if reclass:
        lines.extend([
            "",
            "### Instability reclassification (best shadow thresholds)",
            (
                f"- Boundary instability resolved: "
                f"{reclass.get('boundary_instability_resolved_pct', 0):.1%} "
                f"({reclass.get('boundary_instability_resolved', 0)}/"
                f"{reclass.get('boundary_instability_total', 0)})"
            ),
            (
                f"- Retrieval noise unchanged: "
                f"{reclass.get('retrieval_noise_unchanged_pct', 0):.1%}"
            ),
        ])
    if metrics.get("recall_fusion_instability_pct") is not None:
        lines.append(
            f"- Recall-fusion instability share: "
            f"{metrics.get('recall_fusion_instability_pct', 0):.1%}"
        )
    tradeoff = rc.get("tradeoff_curve_top") or []
    if tradeoff:
        lines.extend([
            "",
            "### Tradeoff curve (top threshold sets)",
            "",
            "| T_chat | T_margin | T_sep | cluster Δ | retrieval loss | score |",
            "|--------|----------|-------|-----------|----------------|-------|",
        ])
        for row in tradeoff[:6]:
            lines.append(
                f"| {row.get('T_chat', 0):.2f} | {row.get('T_margin_low', 0):.2f} | "
                f"{row.get('T_sep', 0):.2f} | "
                f"{row.get('cluster_instability_reduction_pct', 0):.1%} | "
                f"{row.get('retrieval_loss_pct', 0):.1%} | {row.get('score', 0):.3f} |"
            )
    if rc.get("ambiguous_cluster_count"):
        lines.append(
            f"\n- True ambiguity / inconsistent canonical clusters: "
            f"{rc['ambiguous_cluster_count']}"
        )
    return "\n".join(lines)


def _format_routing_hysteresis(rh: dict[str, Any]) -> str:
    if not rh:
        return "_Hysteresis simulation not enabled (use `--simulate-hysteresis`)._"
    table = rh.get("comparison_table") or {}
    rf = table.get("route_flips") or {}
    hn = table.get("hybrid_none_flips") or {}
    retr = table.get("retrieval_consistency") or {}
    lines = [
        f"- Flip reduction: **{rh.get('flip_reduction_rate', 0):.1%}**",
        f"- Stability gain: **{rh.get('stability_gain', 0):+.3f}**",
        (
            f"- Hybrid↔none suppression: "
            f"**{rh.get('hybrid_none_flip_reduction', 0):.1%}** "
            f"({hn.get('baseline', 0)} → {hn.get('hysteresis', 0)})"
        ),
        f"- Retrieval consistency delta: **{rh.get('retrieval_consistency_delta', 0):+.3f}**",
    ]
    if rh.get("safety_flag"):
        lines.append(
            "- ⚠ **Safety flag**: retrieval consistency dropped >2% (shadow only; no routing change)"
        )
    lines.extend([
        "",
        "### Comparison table",
        "",
        "| Metric | Baseline | Hysteresis | Delta |",
        "|--------|----------|------------|-------|",
        (
            f"| Route flips | {rf.get('baseline', 0)} | {rf.get('hysteresis', 0)} | "
            f"↓{rf.get('delta', 0)} |"
        ),
        (
            f"| Hybrid↔none flips | {hn.get('baseline', 0)} | {hn.get('hysteresis', 0)} | "
            f"↓{hn.get('delta', 0)} |"
        ),
        (
            f"| Retrieval consistency | {retr.get('baseline', 0):.3f} | "
            f"{retr.get('hysteresis', 0):.3f} | {retr.get('delta', 0):+.3f} |"
        ),
    ])
    low_b = rh.get("low_margin_instability_baseline")
    low_h = rh.get("low_margin_instability_hysteresis")
    if low_b is not None and low_h is not None:
        lines.extend([
            "",
            "### Low-margin band (0–0.05) instability",
            f"- Baseline unstable cases: {low_b}",
            f"- Hysteresis unstable cases: {low_h}",
            f"- Reduction: {low_b - low_h}",
        ])
    by_cat = rh.get("by_category") or {}
    if by_cat:
        lines.extend(["", "### By category"])
        for cat, stats in sorted(by_cat.items()):
            lines.append(
                f"- {cat}: stability_gain={stats.get('stability_gain', 0):+.3f} "
                f"hybrid↔none_suppression={stats.get('hybrid_none_suppression_rate', 0):.1%}"
            )
    return "\n".join(lines)


def _format_route_perturbation(rp: dict[str, Any]) -> str:
    if not rp:
        return "_Route perturbation analysis not enabled (use `--route-perturbation-analysis`)._"
    lines = [
        f"- Cases analyzed: **{rp.get('cases_analyzed', 0)}**",
        f"- Avg route consistency: **{rp.get('avg_route_consistency', 0):.3f}**",
        f"- Avg retrieval consistency: **{rp.get('avg_retrieval_consistency', 0):.3f}**",
        (
            "- Stability: "
            f"stable={rp.get('stable_rate', 0):.1%}, "
            f"moderate={rp.get('moderately_unstable_rate', 0):.1%}, "
            f"highly_unstable={rp.get('highly_unstable_rate', 0):.1%}"
        ),
        f"- Web trigger stability (avg): {rp.get('avg_web_trigger_stability', 0):.3f}",
    ]
    by_cat = rp.get("by_category") or {}
    if by_cat:
        lines.append("")
        lines.append("### By category")
        for cat, stats in sorted(by_cat.items()):
            lines.append(
                f"- {cat}: route_cons={stats.get('avg_route_consistency', 0):.3f} "
                f"retr_cons={stats.get('avg_retrieval_consistency', 0):.3f} "
                f"unstable={stats.get('unstable_rate', 0):.1%}"
            )
    vbrt = rp.get("variance_by_route_type") or {}
    if vbrt:
        lines.append("")
        lines.append("### Variance by base route type")
        for route, stats in sorted(vbrt.items()):
            lines.append(
                f"- {route}: avg_consistency={stats.get('avg_route_consistency', 0):.3f} "
                f"unstable_rate={stats.get('unstable_rate', 0):.1%} (n={stats.get('count', 0)})"
            )
    heat = rp.get("instability_heatmap") or {}
    if heat:
        lines.append("")
        lines.append("### Instability heatmap (unstable cases)")
        lines.append("")
        lines.append("| chat_score \\ margin | 0-0.05 | 0.05-0.10 | 0.10-0.20 | 0.20+ |")
        lines.append("|---------------------|--------|-----------|-----------|-------|")
        for cs in ("0.0-0.3", "0.3-0.5", "0.5-0.7", "0.7-1.0"):
            row = heat.get(cs) or {}
            lines.append(
                f"| {cs} | {row.get('0-0.05', 0)} | {row.get('0.05-0.10', 0)} | "
                f"{row.get('0.10-0.20', 0)} | {row.get('0.20+', 0)} |"
            )
    top = rp.get("top_unstable_cases") or []
    if top:
        lines.append("")
        lines.append("### Top unstable cases")
        lines.append("")
        lines.append(
            "| case_id | category | route pattern | retrieval | route_cons | label |"
        )
        lines.append(
            "|---------|----------|---------------|-----------|------------|-------|"
        )
        for row in top[:10]:
            lines.append(
                f"| {row.get('case_id', '')} | {row.get('category', '')} | "
                f"{row.get('route_variance_pattern', '')} | "
                f"{row.get('retrieval_variance_pattern', '')} | "
                f"{row.get('route_consistency_score', 0):.2f} | "
                f"{row.get('stability_label', '')} |"
            )
    return "\n".join(lines)


def _format_routing_stability(rs: dict[str, Any]) -> str:
    if not rs:
        return "_Routing stability analysis not enabled (use `--routing-stability-analysis`)._"
    lines = [
        f"- Total clusters: **{rs.get('total_clusters', 0)}**",
        (
            "- Oscillating clusters: "
            f"**{rs.get('oscillating_clusters', 0)}** "
            f"({rs.get('oscillation_rate', 0):.1%})"
        ),
        f"- Avg entropy: {rs.get('avg_entropy', 0):.3f}",
        f"- Avg instability score: {rs.get('avg_instability_score', 0):.3f}",
        f"- Similarity method: {rs.get('similarity_method', 'unknown')} "
        f"(threshold {rs.get('similarity_threshold', 0.85)})",
    ]
    inst_hist = rs.get("instability_histogram") or {}
    if inst_hist:
        lines.append("")
        lines.append("### Instability score distribution")
        for bucket in ("0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"):
            lines.append(f"- {bucket}: {inst_hist.get(bucket, 0)} clusters")
    ent_hist = rs.get("entropy_histogram") or {}
    if ent_hist:
        lines.append("")
        lines.append("### Route entropy distribution")
        for bucket in ("0.0", "0.0-0.5", "0.5-1.0", "1.0-1.5", "1.5+"):
            lines.append(f"- {bucket}: {ent_hist.get(bucket, 0)} clusters")
    top = rs.get("max_instability_clusters") or []
    if top:
        lines.append("")
        lines.append("### Top unstable clusters")
        lines.append("")
        lines.append(
            "| cluster | size | dominant | instability | entropy | oscillating | reason |"
        )
        lines.append(
            "|---------|------|----------|-------------|---------|-------------|--------|"
        )
        for row in top[:10]:
            lines.append(
                f"| {row.get('cluster_id', '')} | {len(row.get('cases') or [])} | "
                f"{row.get('dominant_route', '')} | "
                f"{row.get('instability_score', 0):.2f} | "
                f"{row.get('entropy', 0):.2f} | "
                f"{row.get('is_oscillating', False)} | "
                f"{row.get('oscillation_reason', '') or '-'} |"
            )
    return "\n".join(lines)


def _format_suppression_candidates(rc: dict[str, Any]) -> str:
    candidates = rc.get("retrieval_suppression_candidates") or []
    if not candidates:
        return "### Retrieval suppression candidates\n_No over-retrieval cases._"
    lines = [
        "### Retrieval suppression candidates",
        "",
        "CHAT-labeled prompts that retrieved anyway (sorted by chat_score desc):",
        "",
        "| case_id | chat_score | route | type | hits | recall_fusion |",
        "|---------|------------|-------|------|------|---------------|",
    ]
    for row in candidates[:25]:
        lines.append(
            f"| {row['case_id']} | {row['chat_score']:.3f} | "
            f"{row['route_taken']} | {row['retrieval_type']} | "
            f"{row['retrieval_hits']} | {row['recall_fusion_triggered']} |"
        )
    if len(candidates) > 25:
        lines.append(f"\n_…and {len(candidates) - 25} more (see run.json)._")
    return "\n".join(lines)


def _format_failure_causes(causes: dict[str, int]) -> str:
    if not causes:
        return "_No failure classifications recorded._"
    lines = []
    for reason, count in sorted(causes.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- {reason}: {count}")
    return "\n".join(lines)


def _format_category_table(by_category: dict[str, dict[str, Any]]) -> str:
    if not by_category:
        return "_No categories._"
    lines = [
        "| Category | n | Strict | Family | Downgrades |",
        "|----------|---|--------|--------|------------|",
    ]
    for cat, stats in sorted(by_category.items()):
        lines.append(
            f"| {cat} | {stats.get('total', 0)} | "
            f"{stats.get('strict_accuracy', 0):.1%} | "
            f"{stats.get('family_accuracy', 0):.1%} | "
            f"{stats.get('downgrade_count', 0)} |"
        )
    return "\n".join(lines)


def _format_rewrite_impact(ri: dict[str, Any]) -> str:
    if not ri or not ri.get("attempted_count"):
        return "_Rewrite analysis not enabled (use `--with-sidecar`)._"
    lines = [
        f"- Attempt rate: {ri.get('attempt_rate', 0):.1%} ({ri.get('attempted_count', 0)} cases)",
        f"- Acceptance rate: {ri.get('acceptance_rate', 0):.1%} ({ri.get('applied_count', 0)} applied)",
        f"- Avg extra memory hits when applied: {ri.get('avg_extra_memory_hits', 0):.2f}",
        f"- Avg extra RAG hits when applied: {ri.get('avg_extra_rag_hits', 0):.2f}",
    ]
    return "\n".join(lines)


def _format_memory_analysis(ma: dict[str, Any]) -> str:
    if not ma or not ma.get("total"):
        return "_No memory_recall cases in corpus._"
    lines = [
        f"- Total memory cases: {ma.get('total', 0)}",
        f"- With memory hits: {ma.get('with_hits', 0)}",
        f"- Without memory hits: {ma.get('without_hits', 0)}",
        f"- Strict success: {ma.get('strict_success', 0)}",
        f"- Family success: {ma.get('family_success', 0)}",
    ]
    by_type = ma.get("by_memory_type") or {}
    if by_type:
        lines.append("")
        lines.append("### By memory type (misses)")
        for mtype, stats in sorted(by_type.items()):
            lines.append(
                f"- {mtype}: {stats.get('misses', 0)} misses / {stats.get('total', 0)} total "
                f"({stats.get('hit_rate', 0):.1%} hit rate)"
            )
    return "\n".join(lines)
