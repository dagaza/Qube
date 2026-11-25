#!/usr/bin/env python3
"""Offline knowledge-quality eval over replay fixtures (Stage S1).

Ranks every fixture in the corpus with a production ranker and reports graded
ranking / abstention metrics overall and per category, plus the Phase 1
prompt-contract flags for scientific fixtures.

Exit status is 1 when any of these fail:
  * a Phase 1 contract flag on an evaluated scientific fixture (unchanged gate)
  * a fixture fails schema validation
  * an optional ``--min-*`` / ``--max-*`` aggregate gate
  * a per-category regression vs ``--compare`` baseline (§12.5 rule 7)

Examples::

    .venv/bin/python tools/evaluate_knowledge_quality.py
    .venv/bin/python tools/evaluate_knowledge_quality.py --ranker embed --query-mode standalone
    .venv/bin/python tools/evaluate_knowledge_quality.py --json > baseline.json
    .venv/bin/python tools/evaluate_knowledge_quality.py --compare baseline.json
    # S13 A/B arms: one heuristic off, compared against the default run
    .venv/bin/python tools/evaluate_knowledge_quality.py --ranker ce --ablate trial_grounding
    .venv/bin/python tools/evaluate_knowledge_quality.py --ranker ce --route-arm no_discipline_routing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.knowledge.quality_eval import (  # noqa: E402
    CONTRACT_FLAGS,
    POOL_MODES,
    QUERY_MODES,
    RANKER_NAMES,
    RANKING_ABLATIONS,
    RankerUnavailable,
    aggregate_reports,
    compare_to_baseline,
    evaluate_fixture,
    evaluate_prompt_contract,
    get_fixture_ranker,
    load_json,
    validate_fixture,
)
from core.knowledge.scientific_adapters import ROUTE_ARMS  # noqa: E402

_GATES = (
    # (cli flag, metric, direction)
    ("min_precision_at_k", "precision_at_k", "min"),
    ("min_ndcg_at_5", "ndcg_at_5", "min"),
    ("min_relevant_recall", "relevant_recall", "min"),
    ("min_abstain_correct", "abstain_correct", "min"),
    ("max_irrelevant_kept_rate", "irrelevant_kept_rate", "max"),
)


def _fmt(value) -> str:
    if value is None:
        return "  —  "
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:5.2f}"
    return str(value)


def _print_table(title: str, groups: dict[str, dict]) -> None:
    cols = (
        "fixtures",
        "precision_at_k",
        "ndcg_at_5",
        "irrelevant_kept_rate",
        "relevant_recall",
        "abstain_correct",
        "abstain_precision",
        "abstain_recall",
        "subject_in_query",
    )
    short = ("n", "P@k", "nDCG@5", "irrKept", "relRec", "abstOK", "abstP", "abstR", "subjQ")
    print(f"\n{title}")
    print(f"  {'group':28} " + " ".join(f"{s:>7}" for s in short))
    for name, row in groups.items():
        print(f"  {name[:28]:28} " + " ".join(f"{_fmt(row.get(c)):>7}" for c in cols))


def _is_scientific(entry: dict, fixture: dict) -> bool:
    path = str(fixture.get("path") or "").lower()
    tool = str(entry.get("tool") or "").lower()
    return path == "scientific_evidence" or tool in ("evidence", "science")


def main() -> int:
    # Windows consoles / CI runners default stdout to cp1252; never let an
    # unencodable character in the report crash the run (and mask the gates).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(
        description="Evaluate knowledge-quality replay fixtures (offline)."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "eval" / "retrieval_corpus" / "v1_knowledge_quality.json",
        help="Corpus JSON listing fixtures / cases",
    )
    parser.add_argument("--ranker", default="token", choices=RANKER_NAMES)
    parser.add_argument(
        "--query-mode",
        default="production",
        choices=QUERY_MODES,
        help="production = query today's pipeline ranks with; "
        "standalone = gold standalone question (S5/S6 target)",
    )
    parser.add_argument(
        "--pool-mode",
        default="production",
        choices=POOL_MODES,
        help="production = rows within the legacy per-adapter depth; widened = the "
        "depth S8 requests when the relevance gate runs; full = whole capture",
    )
    parser.add_argument("--k", type=int, default=3, help="k for precision@k")
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Keep synthetic: true candidates in the pool (excluded by default)",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="Baseline produced by --json; fail on per-category regression",
    )
    for flag, _, _ in _GATES:
        parser.add_argument(f"--{flag.replace('_', '-')}", dest=flag, type=float, default=None)
    parser.add_argument(
        "--recency-shift-days",
        type=int,
        default=0,
        help="Judge source freshness as if each capture were this many days older (S7)",
    )
    parser.add_argument(
        "--ablate",
        action="append",
        default=[],
        choices=RANKING_ABLATIONS,
        help="Switch one scientific ranking heuristic off (S13 A/B arm; repeatable)",
    )
    parser.add_argument(
        "--route-arm",
        default=None,
        choices=ROUTE_ARMS,
        help="Rank the pool of one S13 routing arm; fixtures without captured "
        "routes keep their first-captured pool",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON summary")
    args = parser.parse_args()

    try:
        rank_fn = get_fixture_ranker(args.ranker)
    except RankerUnavailable as exc:
        print(f"Ranker {args.ranker!r} unavailable: {exc}", file=sys.stderr)
        return 2

    corpus = load_json(args.corpus)
    results: list[dict] = []
    for entry in corpus.get("queries") or []:
        fixture_rel = entry.get("fixture")
        if not fixture_rel:
            results.append(
                {"id": entry.get("id"), "skipped": True, "reason": "no_offline_fixture"}
            )
            continue
        if not (ROOT / fixture_rel).is_file():
            results.append(
                {"id": entry.get("id"), "skipped": True, "reason": "fixture_file_missing",
                 "fixture": fixture_rel}
            )
            continue
        fixture = load_json(fixture_rel)
        schema_errors = validate_fixture(fixture)
        report = evaluate_fixture(
            fixture,
            ranker=rank_fn,
            query_mode=args.query_mode,
            include_synthetic=args.include_synthetic,
            pool_mode=args.pool_mode,
            k=args.k,
            recency_shift_days=args.recency_shift_days,
            ablate=args.ablate,
            route_arm=args.route_arm,
        )
        report["ranker"] = args.ranker
        if args.ablate:
            report["ablate"] = sorted(args.ablate)
        if args.route_arm:
            report["route_arm"] = args.route_arm
        report["id"] = entry.get("id")
        report["skipped"] = False
        report["schema_errors"] = schema_errors
        if _is_scientific(entry, fixture):
            report["contract"] = evaluate_prompt_contract(
                fixture, include_synthetic=args.include_synthetic
            )
        results.append(report)

    evaluated = [r for r in results if not r.get("skipped")]
    overall = aggregate_reports(evaluated)
    by_category = aggregate_reports(evaluated, group_by="category")
    by_path = aggregate_reports(evaluated, group_by="path")

    failures: list[str] = [
        f"{r['id']}: fixture file missing ({r.get('fixture')})"
        for r in results
        if r.get("reason") == "fixture_file_missing"
    ]
    for r in evaluated:
        for err in r.get("schema_errors") or []:
            failures.append(f"{r['id']}: schema: {err}")
        contract = r.get("contract")
        if contract:
            for flag in CONTRACT_FLAGS:
                if not contract.get(flag):
                    failures.append(f"{r['id']}: contract flag {flag} failed")

    all_row = overall.get("all", {})
    for flag, metric, direction in _GATES:
        threshold = getattr(args, flag)
        if threshold is None:
            continue
        value = all_row.get(metric)
        if value is None:
            failures.append(f"gate {flag}: metric {metric} undefined on this corpus")
        elif direction == "min" and value < threshold:
            failures.append(f"gate {flag}: {metric}={value:.3f} < {threshold}")
        elif direction == "max" and value > threshold:
            failures.append(f"gate {flag}: {metric}={value:.3f} > {threshold}")

    regressions: list[dict] = []
    if args.compare is not None:
        baseline = json.loads(args.compare.read_text(encoding="utf-8"))
        base_aggs = baseline.get("aggregates") or {}
        for scope, current in (("all", overall), ("category", by_category), ("path", by_path)):
            for reg in compare_to_baseline(current, base_aggs.get(scope) or {}):
                reg["scope"] = scope
                regressions.append(reg)
        for reg in regressions:
            failures.append(
                f"regression [{reg['scope']}:{reg['group']}] {reg['metric']}: "
                f"{reg['baseline']:.3f} -> {reg['current']:.3f} (tol {reg['tolerance']})"
            )

    summary = {
        "corpus": str(args.corpus),
        "ranker": args.ranker,
        "query_mode": args.query_mode,
        "include_synthetic": args.include_synthetic,
        "pool_mode": args.pool_mode,
        "recency_shift_days": args.recency_shift_days,
        "total": len(results),
        "evaluated": len(evaluated),
        "aggregates": {"all": overall, "category": by_category, "path": by_path},
        "regressions": regressions,
        "failures": failures,
        "results": results,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"Knowledge quality eval — {len(evaluated)}/{len(results)} fixtures "
            f"(ranker={args.ranker}, query={args.query_mode}, pool={args.pool_mode})"
        )
        for r in results:
            if r.get("skipped"):
                print(f"  SKIP {r.get('id')}: {r.get('reason')}")
                continue
            print(
                f"  {r['id']} [{r['path']}/{r['category']}] expect={r.get('expect') or '—'} "
                f"pool={r['pool_size']} kept={r['kept_count']} "
                f"P@{r['k']}={_fmt(r['precision_at_k']).strip()} "
                f"irrKept={_fmt(r['irrelevant_kept_rate']).strip()} "
                f"abstained={_fmt(r['abstained'])} abstainOK={_fmt(r['abstain_correct'])}"
            )
            if r.get("synthetic_excluded"):
                print(f"    note: {r['synthetic_excluded']} synthetic candidate(s) excluded")
            contract = r.get("contract")
            if contract:
                flags = " ".join(
                    f"{name}={'ok' if contract.get(name) else 'FAIL'}" for name in CONTRACT_FLAGS
                )
                print(
                    f"    contract: {flags} | coverage={contract.get('coverage')} "
                    f"conf={contract.get('confidence')}"
                )
        _print_table("Overall", overall)
        _print_table("By path", by_path)
        _print_table("By category", by_category)
        rec = all_row
        if rec.get("recency_turns"):
            print(
                f"\nRecency (S7, shift {args.recency_shift_days} d): "
                f"{rec['recency_turns']} turn(s) need current information; "
                f"{rec['stale_only_turns']} kept only stale/undated sources with >=1 stale; "
                f"stale_marked={_fmt(rec.get('stale_marked')).strip()}; "
                f"stale presented as sufficient: {rec['stale_presented_sufficient']}"
            )
        if failures:
            print("\nFAILURES:")
            for f in failures:
                print(f"  - {f}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
