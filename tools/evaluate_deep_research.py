#!/usr/bin/env python3
"""Evaluate deep-research merge pipeline against a JSON corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.knowledge.deep_research import run_deep_research  # noqa: E402
from core.knowledge.deep_research_relevance import score_merged_bundle_relevance  # noqa: E402
from core.knowledge.types import SERVICE_SCIENTIFIC_EVIDENCE  # noqa: E402

_COVERAGE_RANK = {
    "none": 0,
    "poor": 1,
    "adequate": 2,
    "excellent": 3,
}


def _load_corpus(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("queries") or [])


def _entry_has_relevance_criteria(entry: dict) -> bool:
    return bool(entry.get("expect_any_tokens") or entry.get("reject_title_patterns"))


def _score_relevance(entry: dict, sources) -> dict:
    if not sources or not _entry_has_relevance_criteria(entry):
        return {"relevance_ok": None}
    return score_merged_bundle_relevance(
        sources,
        expect_any_tokens=list(entry.get("expect_any_tokens") or []),
        reject_title_patterns=list(entry.get("reject_title_patterns") or []),
        top_n=int(entry.get("relevance_top_n") or 3),
        min_relevant_in_top=int(entry.get("min_relevant_in_top") or 2),
    )


def _evaluate_query(entry: dict, *, live: bool, decompose_mode: str | None) -> dict:
    query = str(entry.get("query") or "").strip()
    if not query:
        return {"id": entry.get("id"), "status": "skipped", "reason": "empty_query"}

    if not live:
        payload = {"id": entry.get("id"), "query": query, "status": "dry_run"}
        if _entry_has_relevance_criteria(entry):
            payload["relevance_criteria"] = True
        if decompose_mode:
            payload["decompose_mode"] = decompose_mode
        return payload

    result = run_deep_research(
        query,
        knowledge_service=SERVICE_SCIENTIFIC_EVIDENCE,
        decompose_mode=decompose_mode,
    )
    bundle = result.merged_bundle
    diagnostics = dict(result.diagnostics or {})
    if bundle is None or not bundle.sources:
        return {
            "id": entry.get("id"),
            "query": query,
            "status": "no_results",
            "latency_ms": round(result.latency_ms, 1),
            "sub_queries": list(result.sub_queries),
            "decompose_mode": diagnostics.get("decompose_mode"),
            "relevance_ok": False if _entry_has_relevance_criteria(entry) else None,
            "diagnostics": diagnostics,
        }

    adapters = {s.adapter for s in bundle.sources}
    expect = set(entry.get("expect_adapters") or [])
    adapter_ok = bool(expect.intersection(adapters)) if expect else True
    min_sources = max(1, int(entry.get("min_merged_sources") or 1))
    sources_ok = len(bundle.sources) >= min_sources
    min_cov = str(entry.get("min_coverage_rank") or "adequate").lower()
    cov_ok = _COVERAGE_RANK.get(bundle.coverage, 0) >= _COVERAGE_RANK.get(min_cov, 2)

    relevance = _score_relevance(entry, bundle.sources)
    ok = adapter_ok and sources_ok and cov_ok

    payload = {
        "id": entry.get("id"),
        "query": query,
        "status": "ok" if ok else "partial",
        "latency_ms": round(result.latency_ms, 1),
        "sub_queries": list(result.sub_queries),
        "decompose_mode": diagnostics.get("decompose_mode"),
        "adapters": sorted(adapters),
        "merged_sources": len(bundle.sources),
        "coverage": bundle.coverage,
        "confidence": round(bundle.confidence, 3),
        "strategy": bundle.retrieval_strategy,
        "titles": [s.title for s in bundle.sources[:5]],
        "diagnostics": diagnostics,
        "relevance_ok": relevance.get("relevance_ok"),
        "relevant_in_top": relevance.get("relevant_in_top"),
        "relevance_top_n": relevance.get("relevance_top_n"),
        "examined_titles": relevance.get("examined_titles"),
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate deep-research corpus")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "eval" / "retrieval_corpus" / "v1_deep_research.json",
    )
    parser.add_argument("--live", action="store_true", help="Run live retrieval (network)")
    parser.add_argument(
        "--decompose",
        choices=["heuristic", "llm", "hybrid"],
        default="heuristic",
        help="Sub-query decomposition mode for live eval (default: heuristic)",
    )
    parser.add_argument(
        "--require-relevance",
        action="store_true",
        help="Exit non-zero unless relevance_ok meets --min-relevance-ok",
    )
    parser.add_argument(
        "--min-relevance-ok",
        type=int,
        default=2,
        help="Minimum relevance_ok count when --require-relevance (default: 2 of 3)",
    )
    args = parser.parse_args()

    entries = _load_corpus(args.corpus)
    decompose_mode = args.decompose if args.live else None
    results = [
        _evaluate_query(entry, live=args.live, decompose_mode=decompose_mode)
        for entry in entries
    ]
    ok = sum(1 for r in results if r.get("status") == "ok")
    partial = sum(1 for r in results if r.get("status") == "partial")
    relevance_scored = [r for r in results if r.get("relevance_ok") is not None]
    relevance_ok = sum(1 for r in relevance_scored if r.get("relevance_ok") is True)
    summary = {
        "decompose_mode": decompose_mode,
        "results": results,
        "ok": ok,
        "partial": partial,
        "total": len(results),
        "relevance_ok": relevance_ok,
        "relevance_total": len(relevance_scored),
    }
    print(json.dumps(summary, indent=2))

    if not args.live:
        return 0
    if ok < len(results):
        return 1
    if args.require_relevance and relevance_ok < args.min_relevance_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
