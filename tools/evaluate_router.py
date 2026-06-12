#!/usr/bin/env python3
"""
Offline evaluation harness for CognitiveRouterV4.

Examples:
  python3 tools/evaluate_router.py
  python3 tools/evaluate_router.py --corpus eval/router_corpus/v1_baseline.json
  python3 tools/evaluate_router.py --no-embeddings --output-dir eval/runs/smoke
  python3 tools/evaluate_router.py --with-retrieval
  venv/bin/python tools/evaluate_router.py --eval-fixtures --run-id fixtures_smoke
  python3 tools/evaluate_router.py --baseline eval/runs/baseline.json --fail-on-regression
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _default_corpus() -> Path:
    return _ROOT / "eval" / "router_corpus" / "v1_baseline.json"


def _default_runs_dir() -> Path:
    path = _ROOT / "eval" / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_embed_fn(*, use_embeddings: bool):
    if not use_embeddings:
        return None
    try:
        from rag.embedder import EmbeddingModel

        model = EmbeddingModel()

        def _embed(text: str):
            return model.embed_query(text)

        return _embed
    except Exception as exc:
        logging.warning("Embedding model unavailable (%s); running substring-only router", exc)
        return None


def _embedder_adapter(embed_fn):
    """Wrap a plain callable for ``build_centroid`` (expects ``.embed_query``)."""

    class _Adapter:
        def embed_query(self, text: str):
            return embed_fn(text)

    return _Adapter()


def _build_store(*, with_retrieval: bool, lancedb_dir: Path | None = None):
    if not with_retrieval:
        return None
    try:
        from rag.store import DocumentStore

        if lancedb_dir is not None:
            return DocumentStore(lancedb_dir.resolve(), quiet=True)
        return DocumentStore()
    except Exception as exc:
        logging.warning("RAG store unavailable (%s); retrieval hits will be zero", exc)
        return None


def _build_seed_embedder():
    from rag.embedder import EmbeddingModel

    return EmbeddingModel()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | [%(name)s] %(message)s",
    )

    from core.router_evaluation import (
        RouterEvalConfig,
        build_summary,
        compare_runs,
        corpus_fingerprint,
        evaluate_case,
        format_summary_text,
        install_router_centroids,
        load_corpus,
        load_run_json,
        write_csv,
        write_run_json,
    )
    from mcp.cognitive_router import CognitiveRouterV4

    parser = argparse.ArgumentParser(description="Evaluate CognitiveRouterV4 against a corpus")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=_default_corpus(),
        help="Path to router corpus JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for CSV + JSON run artifacts (default: eval/runs/<run-id>)",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Run identifier for artifacts (default: UTC timestamp)",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip embedder; router uses substring triggers only (Tier 1)",
    )
    parser.add_argument(
        "--with-retrieval",
        action="store_true",
        help="Run memory/RAG search against local LanceDB (hits may be zero on empty DB)",
    )
    parser.add_argument(
        "--lancedb-dir",
        type=Path,
        default=None,
        help="LanceDB directory for --with-retrieval (default: ~/.qube/data/lancedb)",
    )
    parser.add_argument(
        "--seed-eval-library",
        action="store_true",
        help="Index eval/fixtures into --lancedb-dir before running (implies --with-retrieval)",
    )
    parser.add_argument(
        "--force-seed",
        action="store_true",
        help="Re-index eval fixtures even when seed manifest matches",
    )
    parser.add_argument(
        "--eval-fixtures",
        action="store_true",
        help="Shortcut: --seed-eval-library --with-retrieval --lancedb-dir eval/.lancedb",
    )
    parser.add_argument(
        "--with-sidecar",
        action="store_true",
        help="Attempt sidecar query rewrite (requires sidecar model)",
    )
    parser.add_argument(
        "--internet-enabled",
        action="store_true",
        help="Simulate internet tool enabled (affects web veto logic)",
    )
    parser.add_argument(
        "--internet-hybrid-auto",
        action="store_true",
        help="Simulate USE_COGNITIVE_ROUTER_INTERNET auto-web",
    )
    parser.add_argument(
        "--no-discourse",
        action="store_true",
        help="Disable discourse grounding simulation",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Prior run JSON for regression comparison",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit 1 when final accuracy regresses vs --baseline",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Allowed accuracy drop vs baseline before --fail-on-regression trips",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print summary JSON to stdout only",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Write report.md with failure analysis and family accuracy",
    )
    parser.add_argument(
        "--routing-stability-analysis",
        action="store_true",
        help="Post-hoc cluster stability analysis (shadow mode; no routing changes)",
    )
    parser.add_argument(
        "--route-perturbation-analysis",
        action="store_true",
        help="Shadow paraphrase invariance stress test (no routing changes)",
    )
    parser.add_argument(
        "--simulate-hysteresis",
        action="store_true",
        help="Shadow hysteresis simulation on perturbation variants (requires --route-perturbation-analysis)",
    )
    parser.add_argument(
        "--canonicalization-analysis",
        action="store_true",
        help="Shadow canonical route learner + boundary sweep (requires --route-perturbation-analysis)",
    )
    parser.add_argument(
        "--retrieval-propensity-analysis",
        action="store_true",
        help="Shadow continuous retrieval propensity model (requires --route-perturbation-analysis)",
    )
    parser.add_argument(
        "--continuous-pilot-routing",
        action="store_true",
        help="Continuous recall-fusion pilot candidate (requires --route-perturbation-analysis)",
    )
    parser.add_argument(
        "--continuous-arch-validation",
        action="store_true",
        help="Full architectural validation of continuous pilot (requires --route-perturbation-analysis)",
    )
    parser.add_argument(
        "--shadow-retrieval-policy-analysis",
        action="store_true",
        help="Shadow LLMWorker retrieval policy replay on perturbation artifacts",
    )
    parser.add_argument(
        "--propensity-w1",
        type=float,
        default=None,
        help="Override propensity weight w1 (score separation)",
    )
    parser.add_argument(
        "--propensity-w2",
        type=float,
        default=None,
        help="Override propensity weight w2 (1 - chat_score)",
    )
    parser.add_argument(
        "--propensity-w3",
        type=float,
        default=None,
        help="Override propensity weight w3 (1 - confidence_margin)",
    )
    parser.add_argument(
        "--stability-threshold",
        type=float,
        default=0.85,
        help="Similarity threshold for stability clustering (default: 0.85)",
    )
    args = parser.parse_args()

    if args.simulate_hysteresis and not args.route_perturbation_analysis:
        logging.error("--simulate-hysteresis requires --route-perturbation-analysis")
        return 2
    if args.canonicalization_analysis and not args.route_perturbation_analysis:
        logging.error("--canonicalization-analysis requires --route-perturbation-analysis")
        return 2
    if args.retrieval_propensity_analysis and not args.route_perturbation_analysis:
        logging.error("--retrieval-propensity-analysis requires --route-perturbation-analysis")
        return 2
    if args.continuous_pilot_routing and not args.route_perturbation_analysis:
        logging.error("--continuous-pilot-routing requires --route-perturbation-analysis")
        return 2
    if args.continuous_arch_validation and not args.route_perturbation_analysis:
        logging.error("--continuous-arch-validation requires --route-perturbation-analysis")
        return 2
    if args.shadow_retrieval_policy_analysis and not args.route_perturbation_analysis:
        logging.error("--shadow-retrieval-policy-analysis requires --route-perturbation-analysis")
        return 2

    from core.router_eval_seed import default_eval_lancedb_dir, seed_router_eval_library

    if args.eval_fixtures:
        args.seed_eval_library = True
        args.with_retrieval = True
        if args.lancedb_dir is None:
            args.lancedb_dir = default_eval_lancedb_dir()

    lancedb_dir = args.lancedb_dir.resolve() if args.lancedb_dir else None
    if args.seed_eval_library:
        args.with_retrieval = True
        if lancedb_dir is None:
            lancedb_dir = default_eval_lancedb_dir()

    corpus_path = args.corpus.resolve()
    if not corpus_path.is_file():
        logging.error("Corpus not found: %s", corpus_path)
        return 2

    corpus_meta, cases = load_corpus(corpus_path)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else _default_runs_dir() / run_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    config = RouterEvalConfig(
        discourse_enabled=not args.no_discourse,
        internet_enabled=args.internet_enabled,
        internet_hybrid_auto=args.internet_hybrid_auto,
        install_centroids=not args.no_embeddings,
        with_retrieval=args.with_retrieval,
        with_sidecar_rewrite=args.with_sidecar,
    )

    embed_fn = _build_embed_fn(use_embeddings=not args.no_embeddings)
    store = _build_store(with_retrieval=args.with_retrieval, lancedb_dir=lancedb_dir)

    if args.seed_eval_library and store is not None:
        try:
            seed_embedder = _build_seed_embedder()
            seed_summary = seed_router_eval_library(
                store,
                seed_embedder,
                force=args.force_seed,
            )
            logging.info("Eval library seed: %s", seed_summary)
        except Exception as exc:
            logging.error("Eval library seed failed: %s", exc)
            return 2

    sidecar_client = None
    if args.with_sidecar:
        try:
            from core.sidecar_llm import SidecarLlmClient

            sidecar_client = SidecarLlmClient()
        except Exception as exc:
            logging.warning("Sidecar unavailable: %s", exc)

    router = CognitiveRouterV4()
    if config.install_centroids and embed_fn is not None:
        install_router_centroids(router, _embedder_adapter(embed_fn))

    results = []
    for case in cases:
        router_case = CognitiveRouterV4()
        if config.install_centroids and embed_fn is not None:
            install_router_centroids(router_case, _embedder_adapter(embed_fn))
        results.append(
            evaluate_case(
                case,
                router=router_case,
                embed_fn=embed_fn,
                config=config,
                store=store,
                sidecar_client=sidecar_client,
            )
        )

    summary = build_summary(results)

    routing_stability_payload = None
    route_perturbation_payload = None
    routing_hysteresis_payload = None
    routing_canonicalization_payload = None
    retrieval_propensity_payload = None
    continuous_pilot_payload = None
    continuous_arch_validation_payload = None
    shadow_retrieval_policy_payload = None
    pilot_analysis_obj = None
    perturbation_analysis = None
    stability_clusters_path = out_dir / "routing_stability_clusters.json"
    perturbation_detail_path = out_dir / "route_perturbation_cases.json"
    hysteresis_comparison_path = out_dir / "hysteresis_comparison.json"
    canonicalization_path = out_dir / "routing_canonicalization.json"
    propensity_path = out_dir / "retrieval_propensity.json"
    pilot_path = out_dir / "continuous_pilot_routing.json"
    arch_validation_path = out_dir / "continuous_arch_validation.json"
    shadow_policy_path = out_dir / "shadow_retrieval_policy.json"
    if args.routing_stability_analysis:
        from eval.routing_stability import (
            analyze_routing_stability,
            annotate_results_with_stability,
            export_stability_clusters_json,
        )

        stability = analyze_routing_stability(
            results,
            embed_fn=embed_fn,
            threshold=args.stability_threshold,
        )
        results = annotate_results_with_stability(results, stability)
        routing_stability_payload = stability.summary
        export_stability_clusters_json(stability_clusters_path, stability)
        logging.info(
            "Routing stability: %d clusters, %d oscillating (%.1f%%)",
            stability.summary.get("total_clusters", 0),
            stability.summary.get("oscillating_clusters", 0),
            100.0 * float(stability.summary.get("oscillation_rate", 0.0)),
        )

    if args.route_perturbation_analysis:
        from eval.routing_perturbation import (
            analyze_route_perturbation,
            export_perturbation_json,
        )

        perturbation_analysis = analyze_route_perturbation(
            cases,
            results,
            embed_fn=embed_fn,
            config=config,
            store=store,
            run_id=run_id,
            cache_dir=_ROOT / "eval" / "cache",
            corpus_fingerprint=corpus_fingerprint(corpus_path),
        )
        route_perturbation_payload = perturbation_analysis.summary
        export_perturbation_json(perturbation_detail_path, perturbation_analysis)
        logging.info(
            "Route perturbation: %d cases, avg route consistency %.3f, unstable rate %.1f%%",
            perturbation_analysis.summary.get("cases_analyzed", 0),
            float(perturbation_analysis.summary.get("avg_route_consistency", 0.0)),
            100.0 * float(perturbation_analysis.summary.get("unstable_rate", 0.0)),
        )

    if args.simulate_hysteresis and perturbation_analysis is not None:
        from eval.routing_hysteresis import (
            export_hysteresis_comparison_json,
            simulate_hysteresis_on_perturbation,
        )

        hysteresis = simulate_hysteresis_on_perturbation(perturbation_analysis)
        export_hysteresis_comparison_json(hysteresis_comparison_path, hysteresis)
        hs = hysteresis.summary
        routing_hysteresis_payload = {
            "flip_reduction_rate": hs.get("flip_reduction_rate", 0.0),
            "stability_gain": hs.get("stability_gain", 0.0),
            "hybrid_none_flip_reduction": hs.get("hybrid_none_flip_reduction", 0.0),
            "retrieval_consistency_delta": hs.get("retrieval_consistency_delta", 0.0),
            "safety_flag": hs.get("safety_flag", False),
            "comparison_table": hs.get("comparison_table"),
            "by_category": hs.get("by_category"),
            "low_margin_instability_baseline": hs.get("low_margin_instability_baseline"),
            "low_margin_instability_hysteresis": hs.get("low_margin_instability_hysteresis"),
        }
        logging.info(
            "Hysteresis simulation: flip reduction %.1f%%, stability gain %+.3f, "
            "hybrid↔none suppression %.1f%%",
            100.0 * float(hs.get("flip_reduction_rate", 0.0)),
            float(hs.get("stability_gain", 0.0)),
            100.0 * float(hs.get("hybrid_none_flip_reduction", 0.0)),
        )

    if args.canonicalization_analysis and perturbation_analysis is not None:
        import json as _json

        from eval.routing_canonicalization import (
            analyze_routing_canonicalization,
            export_canonicalization_json,
        )

        stability_data = None
        if stability_clusters_path.is_file():
            stability_data = _json.loads(stability_clusters_path.read_text(encoding="utf-8"))

        canonicalization = analyze_routing_canonicalization(
            perturbation_analysis,
            stability_clusters=stability_data,
        )
        export_canonicalization_json(canonicalization_path, canonicalization)
        cs = canonicalization.summary
        top_tradeoff = sorted(
            canonicalization.tradeoff_curve,
            key=lambda r: r.get("score", 0.0),
            reverse=True,
        )[:8]
        routing_canonicalization_payload = {
            "clusters_total": cs.get("clusters_total", 0),
            "clusters_stable_baseline": cs.get("clusters_stable_baseline", 0),
            "clusters_stable_shadow_best": cs.get("clusters_stable_shadow_best", 0),
            "best_threshold_set": cs.get("best_threshold_set"),
            "metrics": cs.get("metrics"),
            "instability_type_breakdown": cs.get("instability_type_breakdown"),
            "interpretation": cs.get("interpretation"),
            "tradeoff_curve_top": top_tradeoff,
            "ambiguous_cluster_count": len(canonicalization.ambiguous_clusters),
        }
        metrics = cs.get("metrics") or {}
        logging.info(
            "Canonicalization: %d clusters, flip reduction %.1f%%, "
            "boundary noise %.1f%%",
            cs.get("clusters_total", 0),
            100.0 * float(metrics.get("flip_reduction_pct", 0.0)),
            100.0 * float(metrics.get("boundary_noise_pct", 0.0)),
        )

    if args.retrieval_propensity_analysis and perturbation_analysis is not None:
        from eval.routing_retrieval_propensity import (
            PropensityWeights,
            analyze_retrieval_propensity,
            export_propensity_json,
        )

        weight_overrides = {
            k: v
            for k, v in {
                "w1": args.propensity_w1,
                "w2": args.propensity_w2,
                "w3": args.propensity_w3,
            }.items()
            if v is not None
        }
        propensity = analyze_retrieval_propensity(
            perturbation_analysis,
            weights=PropensityWeights.from_dict(weight_overrides or None),
            hysteresis_summary=routing_hysteresis_payload,
        )
        export_propensity_json(propensity_path, propensity)
        ps = propensity.summary
        retrieval_propensity_payload = {
            "avg_propensity_score": ps.get("avg_propensity_score"),
            "variance_by_category": ps.get("variance_by_category"),
            "binary_vs_continuous_flip_reduction_pct": ps.get(
                "binary_vs_continuous_flip_reduction_pct"
            ),
            "recall_fusion_flip_reduction_pct": ps.get("recall_fusion_flip_reduction_pct"),
            "hybrid_none_flip_reduction_pct": ps.get("hybrid_none_flip_reduction_pct"),
            "retrieval_loss_proxy": ps.get("retrieval_loss_proxy"),
            "instability_reduction_proxy": ps.get("instability_reduction_proxy"),
            "best_weight_set": ps.get("best_weight_set"),
            "best_thresholds": ps.get("best_thresholds"),
            "hysteresis_comparison": ps.get("hysteresis_comparison"),
            "interpretation": ps.get("interpretation"),
            "failed_categories": ps.get("failed_categories"),
        }
        logging.info(
            "Retrieval propensity: fusion flip reduction %.1f%%, "
            "instability reduction %.1f%%, retrieval loss proxy %.1f%%",
            100.0 * float(ps.get("recall_fusion_flip_reduction_pct", 0.0)),
            100.0 * float(ps.get("instability_reduction_proxy", 0.0)),
            100.0 * float(ps.get("retrieval_loss_proxy", 0.0)),
        )

    if (
        (args.continuous_pilot_routing or args.continuous_arch_validation)
        and perturbation_analysis is not None
    ):
        from eval.routing_continuous_pilot import (
            PropensityWeights as PilotWeights,
            analyze_continuous_pilot_routing,
            export_pilot_json,
        )

        weight_overrides = {
            k: v
            for k, v in {
                "w1": args.propensity_w1,
                "w2": args.propensity_w2,
                "w3": args.propensity_w3,
            }.items()
            if v is not None
        }
        pilot_analysis_obj = analyze_continuous_pilot_routing(
            perturbation_analysis,
            weights=PilotWeights.from_dict(weight_overrides or None),
            hysteresis_summary=routing_hysteresis_payload,
            canonicalization_summary=routing_canonicalization_payload,
            propensity_summary=retrieval_propensity_payload,
        )
        if args.continuous_pilot_routing:
            export_pilot_json(pilot_path, pilot_analysis_obj)
        ps = pilot_analysis_obj.summary
        continuous_pilot_payload = {
            "avg_propensity_score": ps.get("avg_propensity_score"),
            "instability_reduction_pct": ps.get("instability_reduction_pct"),
            "retrieval_loss_proxy": ps.get("retrieval_loss_proxy"),
            "flip_reduction_vs_canonical": ps.get("flip_reduction_vs_canonical"),
            "hybrid_none_flip_reduction_pct": ps.get("hybrid_none_flip_reduction_pct"),
            "best_thresholds": ps.get("best_thresholds"),
            "best_weight_set": ps.get("best_weight_set"),
            "pilot_resolves_all_unstable": ps.get("pilot_resolves_all_unstable"),
            "retrieval_continuity_score": ps.get("retrieval_continuity_score"),
            "per_category_retrieval_coverage": ps.get("per_category_retrieval_coverage"),
            "hysteresis_comparison": ps.get("hysteresis_comparison"),
            "canonicalization_comparison": ps.get("canonicalization_comparison"),
            "interpretation": ps.get("interpretation"),
        }
        logging.info(
            "Continuous pilot: instability reduction %.1f%%, retrieval loss %.1f%%, "
            "resolves all unstable=%s",
            100.0 * float(ps.get("instability_reduction_pct", 0.0)),
            100.0 * float(ps.get("retrieval_loss_proxy", 0.0)),
            ps.get("pilot_resolves_all_unstable"),
        )

    if args.continuous_arch_validation and perturbation_analysis is not None:
        from eval.routing_arch_validation import (
            analyze_continuous_arch_validation,
            export_arch_validation_json,
        )
        from eval.routing_continuous_pilot import PropensityWeights as PilotWeights

        weight_overrides = {
            k: v
            for k, v in {
                "w1": args.propensity_w1,
                "w2": args.propensity_w2,
                "w3": args.propensity_w3,
            }.items()
            if v is not None
        }
        arch_validation = analyze_continuous_arch_validation(
            perturbation_analysis,
            pilot_analysis=pilot_analysis_obj,
            weights=PilotWeights.from_dict(weight_overrides or None),
            hysteresis_summary=routing_hysteresis_payload,
            canonicalization_summary=routing_canonicalization_payload,
            propensity_summary=retrieval_propensity_payload,
        )
        export_arch_validation_json(arch_validation_path, arch_validation)
        av = arch_validation.summary
        continuous_arch_validation_payload = {
            "avg_propensity_score": av.get("avg_propensity_score"),
            "instability_reduction_pct": av.get("instability_reduction_pct"),
            "retrieval_loss_proxy": av.get("retrieval_loss_proxy"),
            "flip_reduction_vs_canonical": av.get("flip_reduction_vs_canonical"),
            "best_thresholds": av.get("best_thresholds"),
            "best_weight_set": av.get("best_weight_set"),
            "validation_passed": av.get("validation_passed"),
            "by_category": av.get("by_category"),
            "flip_type_summary": av.get("flip_type_summary"),
            "comparison_matrix": av.get("comparison_matrix"),
            "top_unstable_clusters": arch_validation.top_unstable_clusters,
            "threshold_sweep_top": arch_validation.threshold_sweep[:6],
            "interpretation": av.get("interpretation"),
        }
        if not args.continuous_pilot_routing:
            from eval.routing_continuous_pilot import export_pilot_json

            export_pilot_json(pilot_path, arch_validation.pilot)
        logging.info(
            "Arch validation: passed=%s, instability reduction %.1f%%, "
            "top unstable clusters=%d",
            av.get("validation_passed"),
            100.0 * float(av.get("instability_reduction_pct", 0.0)),
            len(arch_validation.top_unstable_clusters),
        )

    if args.shadow_retrieval_policy_analysis and perturbation_analysis is not None:
        from eval.shadow_retrieval_policy_eval import (
            analyze_shadow_retrieval_policy,
            export_shadow_policy_eval_json,
        )

        pilot_summary = (
            continuous_arch_validation_payload
            or continuous_pilot_payload
            or retrieval_propensity_payload
        )
        shadow_eval = analyze_shadow_retrieval_policy(
            perturbation_analysis,
            pilot_summary=pilot_summary,
        )
        export_shadow_policy_eval_json(shadow_policy_path, shadow_eval)
        ss = shadow_eval.summary
        shadow_retrieval_policy_payload = {
            "avg_propensity_score": ss.get("avg_propensity_score"),
            "divergence_rate": ss.get("divergence_rate"),
            "recall_fusion_eliminated_rate": ss.get("recall_fusion_eliminated_rate"),
            "hybrid_stability_gain": ss.get("hybrid_stability_gain"),
            "retrieval_coverage_delta": ss.get("retrieval_coverage_delta"),
            "instability_reduction_pct": ss.get("instability_reduction_pct"),
            "best_thresholds": ss.get("best_thresholds"),
            "by_category": ss.get("by_category"),
            "regression_suppression_count": ss.get("regression_suppression_count"),
            "stability_improvement_count": ss.get("stability_improvement_count"),
            "interpretation": ss.get("interpretation"),
            "regression_cases": shadow_eval.regression_cases[:10],
            "improvement_cases": shadow_eval.improvement_cases[:10],
        }
        logging.info(
            "Shadow retrieval policy: divergence %.1f%%, fusion eliminated %.1f%%",
            100.0 * float(ss.get("divergence_rate", 0.0)),
            100.0 * float(ss.get("recall_fusion_eliminated_rate", 0.0)),
        )

    csv_path = out_dir / "results.csv"
    json_path = out_dir / "run.json"
    write_csv(csv_path, results)
    write_run_json(
        json_path,
        corpus_path=corpus_path,
        corpus_meta=corpus_meta,
        config=config,
        results=results,
        summary=summary,
        run_id=run_id,
        notes=f"corpus_fp={corpus_fingerprint(corpus_path)}",
        routing_stability=routing_stability_payload,
        route_perturbation=route_perturbation_payload,
        routing_hysteresis=routing_hysteresis_payload,
        routing_canonicalization=routing_canonicalization_payload,
        retrieval_propensity=retrieval_propensity_payload,
        continuous_pilot_routing=continuous_pilot_payload,
        continuous_arch_validation=continuous_arch_validation_payload,
        shadow_retrieval_policy=shadow_retrieval_policy_payload,
    )

    comparison = None
    if args.baseline and args.baseline.is_file():
        baseline = load_run_json(args.baseline.resolve())
        current = load_run_json(json_path)
        comparison = compare_runs(
            baseline,
            current,
            min_delta=args.min_delta,
        )

    report_path = out_dir / "report.md"
    if args.report:
        from core.router_eval_report import format_evaluation_report

        report_path.write_text(
            format_evaluation_report(
                summary,
                routing_stability=routing_stability_payload,
                route_perturbation=route_perturbation_payload,
                routing_hysteresis=routing_hysteresis_payload,
                routing_canonicalization=routing_canonicalization_payload,
                retrieval_propensity=retrieval_propensity_payload,
                continuous_pilot_routing=continuous_pilot_payload,
                continuous_arch_validation=continuous_arch_validation_payload,
                shadow_retrieval_policy=shadow_retrieval_policy_payload,
            ),
            encoding="utf-8",
        )

    if args.json_only:
        import json
        from dataclasses import asdict

        payload = {"summary": asdict(summary), "comparison": comparison}
        print(json.dumps(payload, indent=2))
    else:
        if args.report:
            from core.router_eval_report import format_evaluation_report

            print(
                format_evaluation_report(
                    summary,
                    routing_stability=routing_stability_payload,
                    route_perturbation=route_perturbation_payload,
                    routing_hysteresis=routing_hysteresis_payload,
                    routing_canonicalization=routing_canonicalization_payload,
                    retrieval_propensity=retrieval_propensity_payload,
                    continuous_pilot_routing=continuous_pilot_payload,
                    continuous_arch_validation=continuous_arch_validation_payload,
                    shadow_retrieval_policy=shadow_retrieval_policy_payload,
                )
            )
        else:
            print(format_summary_text(summary))
        print()
        print(f"Corpus: {corpus_path} ({len(cases)} cases, fp={corpus_fingerprint(corpus_path)})")
        print(f"CSV: {csv_path}")
        print(f"Run JSON: {json_path}")
        if args.report:
            print(f"Report: {report_path}")
        if args.routing_stability_analysis:
            print(f"Stability clusters: {stability_clusters_path}")
        if args.route_perturbation_analysis:
            print(f"Perturbation detail: {perturbation_detail_path}")
        if args.simulate_hysteresis:
            print(f"Hysteresis comparison: {hysteresis_comparison_path}")
        if args.canonicalization_analysis:
            print(f"Canonicalization: {canonicalization_path}")
        if args.retrieval_propensity_analysis:
            print(f"Retrieval propensity: {propensity_path}")
        if args.continuous_pilot_routing or args.continuous_arch_validation:
            print(f"Continuous pilot: {pilot_path}")
        if args.continuous_arch_validation:
            print(f"Arch validation: {arch_validation_path}")
        if args.shadow_retrieval_policy_analysis:
            print(f"Shadow retrieval policy: {shadow_policy_path}")
        if comparison:
            print()
            print("Regression vs baseline:")
            print(
                f"  {comparison['metric']}: {comparison['baseline']:.1%} -> "
                f"{comparison['current']:.1%} (delta {comparison['delta']:+.1%})"
            )
            if comparison.get("new_failures"):
                print(f"  New failures: {len(comparison['new_failures'])}")
            if comparison.get("fixed_cases"):
                print(f"  Fixed cases: {len(comparison['fixed_cases'])}")
            if comparison.get("regressed"):
                print("  REGRESSION DETECTED")

    if args.fail_on_regression and comparison and comparison.get("regressed"):
        return 1
    if summary.errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
