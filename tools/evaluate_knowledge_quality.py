#!/usr/bin/env python3
"""Phase 0 offline knowledge-quality eval (prompt contract + fixture replay)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.knowledge.quality_eval import (  # noqa: E402
    evaluate_ant_fixture,
    load_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Phase 0 knowledge-quality fixtures (offline)."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "eval" / "retrieval_corpus" / "v1_knowledge_quality.json",
        help="Corpus JSON listing fixtures / cases",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summary",
    )
    args = parser.parse_args()

    corpus = load_json(args.corpus)
    results: list[dict] = []
    for entry in corpus.get("queries") or []:
        fixture_rel = entry.get("fixture")
        if not fixture_rel:
            results.append(
                {
                    "id": entry.get("id"),
                    "skipped": True,
                    "reason": "no_offline_fixture",
                }
            )
            continue
        fixture = load_json(fixture_rel)
        report = evaluate_ant_fixture(fixture)
        report["id"] = entry.get("id")
        report["skipped"] = False
        results.append(report)

    summary = {
        "corpus": str(args.corpus),
        "total": len(results),
        "evaluated": sum(1 for r in results if not r.get("skipped")),
        "results": results,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"Knowledge quality eval — {summary['evaluated']}/{summary['total']} fixtures")
        for r in results:
            if r.get("skipped"):
                print(f"  SKIP {r.get('id')}: {r.get('reason')}")
                continue
            flags = [
                ("token_strip", r.get("composer_token_stripped")),
                ("abstention", r.get("prompt_allows_abstention")),
                ("sci_persona", r.get("scientific_persona")),
                ("date", r.get("date_injected")),
                ("metadata", r.get("evidence_pack_has_metadata")),
                ("freshness", r.get("freshness_populated")),
            ]
            flag_s = " ".join(
                f"{name}={'ok' if ok else 'FAIL'}" for name, ok in flags
            )
            print(
                f"  {r.get('id')}: coverage={r.get('coverage')} "
                f"conf={r.get('confidence')} kept={r.get('kept_count')} "
                f"reject_kept={r.get('kept_reject_title_count')} | {flag_s}"
            )
            if r.get("phase0_known_gap_token_ranker_keeps_irrelevant"):
                print(
                    "    note: token ranker still keeps reject-title hits "
                    "(Phase 2 reranker target)"
                )

    # Soft gate: Phase 1 contract checks must pass on evaluated fixtures.
    hard_failures = [
        r
        for r in results
        if not r.get("skipped")
        and (
            not r.get("composer_token_stripped")
            or not r.get("prompt_allows_abstention")
            or not r.get("scientific_persona")
            or not r.get("date_injected")
            or not r.get("evidence_pack_has_metadata")
            or not r.get("freshness_populated")
        )
    ]
    return 1 if hard_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
