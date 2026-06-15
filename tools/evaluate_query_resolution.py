#!/usr/bin/env python3
"""
Offline query-resolution evaluation harness.

Measures discourse query resolution (inference/web/routing strings) and
optional offline web fixture retrieval quality.

Examples:
  python3 tools/evaluate_query_resolution.py
  python3 tools/evaluate_query_resolution.py --no-embeddings
  python3 tools/evaluate_query_resolution.py --output-dir eval/runs/qr_smoke
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
    return _ROOT / "eval" / "router_corpus" / "query_resolution_v1.json"


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
        logging.warning(
            "Embedding model unavailable (%s); web fixture gate uses lexical overlap only",
            exc,
        )
        return None


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | [%(name)s] %(message)s",
    )

    from core.query_resolution_evaluation import (
        DEFAULT_WEB_FIXTURES_DIR,
        build_query_resolution_summary,
        evaluate_query_resolution_case,
        format_query_resolution_report,
        load_query_resolution_corpus,
        write_query_resolution_run_json,
    )

    parser = argparse.ArgumentParser(
        description="Evaluate discourse query resolution against a labeled corpus"
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=_default_corpus(),
        help="Path to query resolution corpus JSON",
    )
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=DEFAULT_WEB_FIXTURES_DIR,
        help="Directory of offline DuckDuckGo HTML fixtures",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for JSON run artifact",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Run identifier (default: UTC timestamp)",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip embedder; web fixture gate uses token overlap only",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit 1 when any case fails resolution expectations",
    )
    args = parser.parse_args()

    corpus_path = args.corpus.resolve()
    if not corpus_path.is_file():
        logging.error("Corpus not found: %s", corpus_path)
        return 2

    meta, cases = load_query_resolution_corpus(corpus_path)
    embed_fn = _build_embed_fn(use_embeddings=not args.no_embeddings)

    results = [
        evaluate_query_resolution_case(
            case,
            embed_fn=embed_fn,
            fixtures_dir=args.fixtures_dir.resolve(),
        )
        for case in cases
    ]
    summary = build_query_resolution_summary(results)

    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    out_dir = args.output_dir or (_default_runs_dir() / f"query_resolution_{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_path = out_dir / "run.json"

    write_query_resolution_run_json(
        run_path,
        meta={
            "run_id": run_id,
            "corpus": str(corpus_path),
            "corpus_description": meta.get("description", ""),
            "fixtures_dir": str(args.fixtures_dir.resolve()),
            "embeddings": embed_fn is not None,
            "case_count": len(cases),
        },
        summary=summary,
        results=results,
    )

    report = format_query_resolution_report(summary)
    print(report)
    print(f"\nWrote {run_path}")

    if args.fail_on_regression and summary.failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
