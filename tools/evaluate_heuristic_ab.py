#!/usr/bin/env python3
"""A/B of the scientific heuristics on the knowledge-quality corpus (plan S13).

For every scientific fixture and query mode, the production replay is compared
with one arm per heuristic switched off:

* ``trial_grounding``  — no trial-acronym boost (ranking side);
* ``discipline_routing`` — every query routed as ``general_science`` (pool side).

Routing arms rank a different pool, so the comparison uses absolute kept
counts (relevant / partial / irrelevant) and the sufficiency state, not
pool-relative recall (S8 caveat). A fixture without captured ``routes`` has no
routing arms. ``gain`` is ``2·relevant + partial − irrelevant`` of the kept set;
a positive ``delta`` means the heuristic helps that fixture.

S21 adds nDCG@5 and relevant recall on the *shared* pool: both arms' kept rows
scored against the union of the two arms' pools (``quality_eval.shared_pool``).
For a ranking ablation both arms rank the same pool, so it equals the own pool.

Usage::

    .venv/bin/python tools/evaluate_heuristic_ab.py                # ce + token
    .venv/bin/python tools/evaluate_heuristic_ab.py --ranker ce --json > ab.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.knowledge import quality_eval as qe  # noqa: E402
from core.knowledge.scientific_adapters import (  # noqa: E402
    ROUTE_NO_DISCIPLINE,
    ROUTE_PRODUCTION,
)

ARMS: dict[str, dict[str, Any]] = {
    "trial_grounding": {"ablate": (qe.ABLATE_TRIAL_GROUNDING,)},
    "discipline_routing": {"route_arm": ROUTE_NO_DISCIPLINE},
}


# ``evaluate_fixture`` reports kept titles cut to this many characters.
KEPT_TITLE_CHARS = qe.KEPT_TITLE_CHARS


def _title_labels(fixture: dict[str, Any]) -> dict[str, Any]:
    """Map each candidate's reported (truncated) title to its label.

    Matching on the full title silently dropped every kept row whose title is
    longer than the report's cut. A key shared by two different labels would
    make the count ambiguous, so it is refused.
    """
    labels: dict[str, Any] = {}
    for c in fixture.get("candidates") or []:
        key = str(c.get("title") or "")[:KEPT_TITLE_CHARS]
        if key in labels and labels[key] != c.get("label"):
            raise ValueError(f"{fixture.get('id')}: title {key!r} has conflicting labels")
        labels[key] = c.get("label")
    return labels


def _outcome(report: dict[str, Any], fixture: dict[str, Any]) -> dict[str, Any]:
    labels = _title_labels(fixture)
    kept = list(report.get("kept_titles") or [])
    got = [labels.get(t) for t in kept]
    if None in got:
        missing = [t for t, g in zip(kept, got) if g is None]
        raise ValueError(f"{fixture.get('id')}: kept rows without a label: {missing}")
    rel, part, irr = got.count("relevant"), got.count("partial"), got.count("irrelevant")
    return {
        "relevant": rel,
        "partial": part,
        "irrelevant": irr,
        "gain": 2 * rel + part - irr,
        "state": report.get("state"),
        "expect": report.get("expect"),
        "abstain_correct": report.get("abstain_correct"),
        "pool_size": report.get("pool_size"),
        "kept": kept,
    }


def _shared_scores(fixture: dict[str, Any], report: dict[str, Any], pool: list[dict[str, Any]]) -> dict[str, Any]:
    scored = qe.rescore_on_pool(fixture, report, report["kept_labels"], pool)
    return {"ndcg_at_5": scored["ndcg_at_5"], "relevant_recall": scored["relevant_recall"]}


def run(rankers: list[str], modes: list[str]) -> list[dict[str, Any]]:
    corpus = qe.load_json("eval/retrieval_corpus/v1_knowledge_quality.json")
    rows: list[dict[str, Any]] = []
    for ranker in rankers:
        fn = qe.get_fixture_ranker(ranker)
        pool_mode = "widened" if ranker == "ce" else "production"
        for mode in modes:
            for entry in corpus.get("queries") or []:
                if entry.get("path") != "scientific_evidence" or not entry.get("fixture"):
                    continue
                fixture = qe.load_json(entry["fixture"])
                routed = qe.fixture_has_routes(fixture, mode)
                base_arm = ROUTE_PRODUCTION if routed else None
                common = {"ranker": fn, "query_mode": mode, "pool_mode": pool_mode}
                base_report = qe.evaluate_fixture(fixture, route_arm=base_arm, **common)
                base = _outcome(base_report, fixture)
                for arm, kwargs in ARMS.items():
                    if "route_arm" in kwargs and not routed:
                        continue
                    kw = {"route_arm": base_arm, **kwargs}
                    off_report = qe.evaluate_fixture(fixture, **common, **kw)
                    off = _outcome(off_report, fixture)
                    pool = qe.shared_pool(
                        fixture,
                        [
                            {"query_mode": mode, "pool_mode": pool_mode, "route_arm": base_arm},
                            {"query_mode": mode, "pool_mode": pool_mode, "route_arm": kw["route_arm"]},
                        ],
                    )
                    rows.append(
                        {
                            "ranker": ranker,
                            "query_mode": mode,
                            "id": entry["id"],
                            "category": entry.get("category"),
                            "heuristic": arm,
                            "production": base,
                            "off": off,
                            "changed": base["kept"] != off["kept"] or base["state"] != off["state"],
                            "delta": base["gain"] - off["gain"],
                            "shared": {
                                "pool_size": len(pool),
                                "production": _shared_scores(fixture, base_report, pool),
                                "off": _shared_scores(fixture, off_report, pool),
                            },
                        }
                    )
    return rows


def _mean(values) -> float | None:
    nums = [float(v) for v in values if v is not None]
    return sum(nums) / len(nums) if nums else None


def _fmt(value: float | None) -> str:
    return "  —  " if value is None else f"{value:.3f}"


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = defaultdict(dict)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["heuristic"], r["ranker"], r["query_mode"])].append(r)
    for (heuristic, ranker, mode), items in sorted(groups.items()):
        changed = [r for r in items if r["changed"]]
        out[heuristic][f"{ranker}/{mode}"] = {
            "fixtures": len(items),
            "changed": len(changed),
            "heuristic_helps": sum(1 for r in changed if r["delta"] > 0),
            "heuristic_hurts": sum(1 for r in changed if r["delta"] < 0),
            "net_gain": sum(r["delta"] for r in items),
            "abstain_correct_on": sum(1 for r in items if r["production"]["abstain_correct"]),
            "abstain_correct_off": sum(1 for r in items if r["off"]["abstain_correct"]),
            # S21: both arms scored on the union of their pools.
            "shared_ndcg_on": _mean(r["shared"]["production"]["ndcg_at_5"] for r in items),
            "shared_ndcg_off": _mean(r["shared"]["off"]["ndcg_at_5"] for r in items),
            "shared_recall_on": _mean(r["shared"]["production"]["relevant_recall"] for r in items),
            "shared_recall_off": _mean(r["shared"]["off"]["relevant_recall"] for r in items),
        }
    return dict(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ranker", action="append", choices=qe.RANKER_NAMES, default=[])
    parser.add_argument("--query-mode", action="append", choices=("production", "standalone"), default=[])
    parser.add_argument("--json", action="store_true", help="Emit every row plus the summary")
    args = parser.parse_args()
    try:
        rows = run(args.ranker or ["ce", "token"], args.query_mode or ["production", "standalone"])
    except qe.RankerUnavailable as exc:
        print(f"ranker unavailable: {exc}", file=sys.stderr)
        return 2
    summary = summarise(rows)
    if args.json:
        json.dump({"summary": summary, "rows": rows}, sys.stdout, indent=1)
        print()
        return 0
    for heuristic, by_run in summary.items():
        print(f"== {heuristic}")
        for run_id, s in by_run.items():
            print(
                f"  {run_id:22s} changed {s['changed']:2d}/{s['fixtures']:2d}  helps {s['heuristic_helps']:2d}"
                f"  hurts {s['heuristic_hurts']:2d}  net {s['net_gain']:+d}"
                f"  abstain-correct on/off {s['abstain_correct_on']}/{s['abstain_correct_off']}"
            )
            print(
                f"  {'':22s} shared pool: nDCG@5 on/off {_fmt(s['shared_ndcg_on'])}/{_fmt(s['shared_ndcg_off'])}"
                f"  recall on/off {_fmt(s['shared_recall_on'])}/{_fmt(s['shared_recall_off'])}"
            )
        for r in rows:
            if r["heuristic"] == heuristic and r["changed"]:
                p, o = r["production"], r["off"]
                sp, so = r["shared"]["production"], r["shared"]["off"]
                print(
                    f"    {r['ranker']:5s} {r['query_mode'][:4]} {r['id']:38s} on {p['relevant']}/{p['partial']}/{p['irrelevant']}"
                    f" {p['state']:12s} off {o['relevant']}/{o['partial']}/{o['irrelevant']} {o['state']:12s} delta {r['delta']:+d}"
                    f"  shared nDCG {_fmt(sp['ndcg_at_5'])}/{_fmt(so['ndcg_at_5'])}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
