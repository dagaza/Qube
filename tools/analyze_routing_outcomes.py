#!/usr/bin/env python3
"""
Summarize joined retrieval-outcome telemetry from routing_debug JSONL.

Requires QUBE_ROUTING_DEBUG_LOG=1 during chat sessions so turns persist
``retrieval_outcome`` blocks (schema_version >= 2).

Examples:
  python tools/analyze_routing_outcomes.py
  python tools/analyze_routing_outcomes.py --path ~/.qube/logs/routing_debug.log
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.routing_debug_sink import default_routing_debug_log_path  # noqa: E402

_RETRIEVAL_ROUTES = frozenset({"memory", "rag", "hybrid", "web", "internet"})


def _load_records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            records.append(obj)
    return records


def _outcome(rec: dict) -> dict | None:
    ro = rec.get("retrieval_outcome")
    return ro if isinstance(ro, dict) else None


def summarize(records: list[dict]) -> dict:
    total = 0
    with_outcome = 0
    downgrade = 0
    retrieval_attempted = 0
    retrieval_empty = 0
    rewrite_applied = 0
    rewrite_helped = 0
    target_mismatch = 0
    route_dist: Counter[str] = Counter()
    final_dist: Counter[str] = Counter()

    for rec in records:
        total += 1
        o = _outcome(rec)
        if not o:
            continue
        with_outcome += 1
        pre = str(o.get("execution_route_pre_downgrade") or "").lower()
        final = str(o.get("execution_route_final") or "").lower()
        route_dist[pre] += 1
        final_dist[final] += 1

        if pre in _RETRIEVAL_ROUTES:
            retrieval_attempted += 1
            hits = int(o.get("memory_hits") or 0) + int(o.get("rag_hits") or 0)
            hits += int(o.get("web_hits") or 0)
            if hits == 0:
                retrieval_empty += 1

        if o.get("downgrade_fired"):
            downgrade += 1

        if o.get("sidecar_rewrite_applied"):
            rewrite_applied += 1
            extra = int(o.get("hybrid_extra_memory") or 0) + int(
                o.get("hybrid_extra_rag") or 0
            )
            if extra > 0:
                rewrite_helped += 1

        rec_target = str(o.get("sidecar_recommended_target") or "").lower()
        top = str(o.get("top_intent") or "").lower()
        if rec_target and rec_target not in ("", "none") and top and rec_target != top:
            target_mismatch += 1

    def _pct(n: int, d: int) -> float:
        return (100.0 * n / d) if d else 0.0

    return {
        "total_records": total,
        "with_retrieval_outcome": with_outcome,
        "retrieval_attempted": retrieval_attempted,
        "retrieval_empty_pct": _pct(retrieval_empty, retrieval_attempted),
        "downgrade_fired": downgrade,
        "downgrade_pct_of_retrieval": _pct(downgrade, retrieval_attempted),
        "sidecar_rewrite_applied": rewrite_applied,
        "sidecar_rewrite_helped_pct": _pct(rewrite_helped, rewrite_applied),
        "sidecar_target_vs_top_intent_mismatch": target_mismatch,
        "route_pre_downgrade_distribution": dict(route_dist),
        "route_final_distribution": dict(final_dist),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze routing retrieval outcomes.")
    ap.add_argument(
        "--path",
        type=Path,
        default=None,
        help="routing_debug.log path (default: ~/.qube/logs/routing_debug.log)",
    )
    ap.add_argument("--json", action="store_true", help="Emit JSON summary only")
    args = ap.parse_args()

    path = args.path or default_routing_debug_log_path()
    records = _load_records(path)
    summary = summarize(records)

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print(f"Log: {path}")
    print(f"Records: {summary['total_records']} ({summary['with_retrieval_outcome']} with retrieval_outcome)")
    if summary["with_retrieval_outcome"] == 0:
        print(
            "No retrieval_outcome blocks found. Enable QUBE_ROUTING_DEBUG_LOG=1 "
            "and run chat turns after upgrading to schema v2."
        )
        return 0

    print(f"Retrieval attempted: {summary['retrieval_attempted']}")
    print(
        f"  Empty hits (route+gates): {summary['retrieval_empty_pct']:.1f}% "
        f"of retrieval attempts"
    )
    print(
        f"  Downgrade to CHAT: {summary['downgrade_fired']} turns "
        f"({summary['downgrade_pct_of_retrieval']:.1f}% of retrieval attempts)"
    )
    print(f"Sidecar rewrite applied: {summary['sidecar_rewrite_applied']}")
    if summary["sidecar_rewrite_applied"]:
        print(
            f"  Extra hybrid hits from rewrite: "
            f"{summary['sidecar_rewrite_helped_pct']:.1f}% of rewrite turns"
        )
    if summary["sidecar_target_vs_top_intent_mismatch"]:
        print(
            f"Sidecar recommended_target != router top_intent: "
            f"{summary['sidecar_target_vs_top_intent_mismatch']} turns"
        )
    print("Pre-downgrade routes:", summary["route_pre_downgrade_distribution"])
    print("Final routes:", summary["route_final_distribution"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
