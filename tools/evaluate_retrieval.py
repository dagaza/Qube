#!/usr/bin/env python3
"""Evaluate external knowledge retrieval against a JSON corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.knowledge.types import SERVICE_SCIENTIFIC_EVIDENCE  # noqa: E402
from core.knowledge.web_retrieval import run_v2_web_retrieval  # noqa: E402


def _load_corpus(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("queries") or [])


def _evaluate_query(entry: dict, *, live: bool) -> dict:
    query = str(entry.get("query") or "").strip()
    if not query:
        return {"id": entry.get("id"), "status": "skipped", "reason": "empty_query"}

    if not live:
        return {
            "id": entry.get("id"),
            "query": query,
            "status": "dry_run",
        }

    outcome = run_v2_web_retrieval(
        query=query,
        semantic_query=query,
        knowledge_service=SERVICE_SCIENTIFIC_EVIDENCE,
    )
    bundle = outcome.bundle
    if bundle is None or not bundle.sources:
        return {
            "id": entry.get("id"),
            "query": query,
            "status": "no_results",
            "latency_ms": outcome.latency_ms,
        }

    adapters = {s.adapter for s in bundle.sources}
    abstract_hits = sum(1 for s in bundle.sources if s.fetch_status == "abstract")
    expect = set(entry.get("expect_adapters") or [])
    adapter_hit = bool(expect.intersection(adapters)) if expect else True
    expect_abstract = bool(entry.get("expect_abstract", True))
    abstract_ok = (not expect_abstract) or abstract_hits >= 1

    return {
        "id": entry.get("id"),
        "query": query,
        "status": "ok" if adapter_hit and abstract_ok else "partial",
        "latency_ms": round(outcome.latency_ms, 1),
        "adapters": sorted(adapters),
        "abstract_hits": abstract_hits,
        "coverage": bundle.coverage,
        "confidence": round(bundle.confidence, 3),
        "stop_reason": bundle.stop_reason,
        "titles": [s.title for s in bundle.sources],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate knowledge retrieval corpus")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "eval" / "retrieval_corpus" / "v1_scientific.json",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live adapter calls (requires network)",
    )
    args = parser.parse_args()

    entries = _load_corpus(args.corpus)
    results = [_evaluate_query(e, live=args.live) for e in entries]
    ok = sum(1 for r in results if r.get("status") in {"ok", "dry_run"})
    print(json.dumps({"corpus": str(args.corpus), "results": results}, indent=2))
    print(f"\nSummary: {ok}/{len(results)} entries ok or dry-run", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
