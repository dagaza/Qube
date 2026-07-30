#!/usr/bin/env python3
"""
Offline Library RAG / chunking baseline harness (Phase 0 lite).

Examples:
  venv/bin/python tools/evaluate_library_chunking.py --seed --force-seed
  venv/bin/python tools/evaluate_library_chunking.py --run-id baseline_v1
  python3 tools/evaluate_library_chunking.py --no-embeddings  # metric unit smoke only
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
    return _ROOT / "eval" / "library_corpus" / "v1_baseline.json"


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
        logging.warning("Embedding model unavailable (%s)", exc)
        return None


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | [%(name)s] %(message)s",
    )

    from core.library_chunking_evaluation import (
        LibraryEvalConfig,
        build_summary,
        compare_runs,
        evaluate_case,
        format_summary_text,
        load_corpus,
        load_run_json,
        write_csv,
        write_run_json,
        collect_index_stats,
    )
    from core.library_eval_seed import (
        default_eval_lancedb_dir,
        seed_library_eval_corpus,
    )

    parser = argparse.ArgumentParser(description="Evaluate Library RAG retrieval baseline")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=_default_corpus(),
        help="Library eval corpus JSON",
    )
    parser.add_argument(
        "--lancedb-dir",
        type=Path,
        default=default_eval_lancedb_dir(),
        help="LanceDB directory (default: eval/.lancedb)",
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Seed fixture library with production ingest pipeline before eval",
    )
    parser.add_argument(
        "--force-seed",
        action="store_true",
        help="Re-index all fixture documents (eval scope only)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k for rag_search (default: 5)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Run id under eval/runs/<run_id>/ (default: timestamp)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (instead of eval/runs/<run_id>)",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Prior run.json for regression comparison",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero when metrics regress vs --baseline",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip retrieval eval (seed/index stats only)",
    )
    parser.add_argument(
        "--quiet-store",
        action="store_true",
        help="Suppress DocumentStore diagnostic prints",
    )
    args = parser.parse_args()

    corpus_path = args.corpus.resolve()
    corpus = load_corpus(corpus_path)
    cases = corpus["cases"]

    db_dir = args.lancedb_dir.resolve()
    db_dir.mkdir(parents=True, exist_ok=True)

    seed_summary = None
    store = None

    if args.seed or args.force_seed:
        from rag.embedder import EmbeddingModel
        from rag.store import DocumentStore

        embedder = EmbeddingModel()
        store = DocumentStore(db_dir, quiet=args.quiet_store)
        if args.force_seed and getattr(store, "dim_mismatch", False):
            logging.warning(
                "Recreating eval LanceDB table for embedding dim=%s (was mismatched).",
                store.vector_dim,
            )
            store.recreate_for_dim(store.vector_dim)
        seed_summary = seed_library_eval_corpus(
            store,
            embedder,
            force=args.force_seed,
        )
        print(seed_summary)

    if store is None:
        try:
            from rag.store import DocumentStore

            store = DocumentStore(db_dir, quiet=args.quiet_store)
        except Exception as exc:
            logging.error("DocumentStore unavailable: %s", exc)
            return 2

    index_stats = collect_index_stats(store)
    if index_stats.get("library_rows", 0) == 0:
        logging.warning(
            "No eval library rows in %s — run with --seed --force-seed first.",
            db_dir,
        )
    elif float(index_stats.get("meta_json_coverage") or 0.0) < 0.5:
        logging.warning(
            "Low meta_json coverage (%.0f%%). Re-seed with --seed --force-seed "
            "to index via the production ingest pipeline.",
            float(index_stats.get("meta_json_coverage") or 0.0) * 100.0,
        )
    config = LibraryEvalConfig(top_k=max(1, args.top_k))
    results = []

    embed_fn = _build_embed_fn(use_embeddings=not args.no_embeddings)
    if embed_fn is None:
        if args.no_embeddings:
            logging.info("Skipping retrieval cases (--no-embeddings)")
        else:
            logging.error("Embedding model required for retrieval eval.")
            return 2
    else:
        for case in cases:
            result = evaluate_case(case, embed_fn=embed_fn, store=store, config=config)
            results.append(result)
            status = "PASS" if result.success else "FAIL"
            logging.info(
                "[%s] %s hits=%d recall=%.2f rr=%.2f reason=%s",
                status,
                result.case_id,
                result.hit_count,
                result.recall_at_k,
                result.reciprocal_rank,
                result.failure_reason,
            )

    summary = build_summary(
        results,
        corpus_path=corpus_path,
        seed_summary=seed_summary,
        index_stats=index_stats,
        config=config,
    )

    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("library_%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir.resolve() if args.output_dir else (_default_runs_dir() / run_id)
    write_csv(output_dir / "results.csv", results)
    write_run_json(output_dir / "run.json", summary, results)

    print(format_summary_text(summary))
    print(f"Artifacts: {output_dir}")

    if args.baseline is not None:
        baseline = load_run_json(args.baseline.resolve())
        regressions = compare_runs(summary, baseline)
        if regressions:
            print("Regressions vs baseline:")
            for line in regressions:
                print(f"  - {line}")
            if args.fail_on_regression:
                return 1
        else:
            print("No regressions vs baseline.")

    if results and summary.get("success_count", 0) < len(results):
        return 1 if args.fail_on_regression else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
