#!/usr/bin/env python3
"""Evaluate library and memory retrieval offline from a capture (plan S33).

Replays every case's production call (``rag_search`` / ``memory_search``)
against the captured rows and reports the knowledge-quality metrics per run
and category, plus what the relevance floors keep and drop.

  .venv/bin/python tools/evaluate_library_memory.py \
      --capture eval/fixtures/library_memory/captured_balanced.json \
      [--json] [--out <report.json>] [--compare <baseline.json>]

With ``--rerank-scores eval/fixtures/library_memory/rerank_scores.json`` it
reports the recorded cross-encoder scores of each run's candidates instead,
per label, next to the embedding score (plan S34).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _fmt(v) -> str:
    if v is None:
        return "  —  "
    if isinstance(v, float):
        return f"{v:5.2f}"
    return f"{v:>5}"


def format_text(report: dict) -> str:
    lines = [
        f"embedding: {report.get('embedding_mode')} ({report.get('model')})  floors: {report.get('floors')}",
    ]
    for run, s in report["summary"].items():
        a = s["aggregate"]
        f = s["floor"]
        lines.append("")
        lines.append(
            f"[{run}] cases={a['fixtures']} P@k={_fmt(a['precision_at_k'])} nDCG@5={_fmt(a['ndcg_at_5'])} "
            f"irrelevant_kept={_fmt(a['irrelevant_kept_rate'])} recall={_fmt(a['relevant_recall'])} "
            f"abstain_correct={_fmt(a['abstain_correct'])}"
        )
        lines.append(
            f"  floor {f['floor']}: vector rows {f['vector_rows']}; below {f['below_floor']}; above {f['above_floor']}; "
            f"cases with every row below {f['cases_with_every_row_below']}"
        )
        sem = s["scores"]["semantic"]
        for lab in ("relevant", "partial", "irrelevant"):
            p = sem.get(lab)
            if p:
                lines.append(
                    f"  semantic {lab:10s} n={p['n']:4d} min={p['min']:.3f} p10={p['p10']:.3f} "
                    f"p50={p['p50']:.3f} p90={p['p90']:.3f} max={p['max']:.3f}"
                )
        lines.append(
            f"  AUC relevant>irrelevant {s['scores']['auc_relevant_vs_irrelevant']}  "
            f"useful>irrelevant {s['scores']['auc_useful_vs_irrelevant']}"
        )
        if s.get("memory_rows_kept") is not None:
            lines.append(f"  memory rows kept as library chunks: {s['memory_rows_kept']}")
        lines.append(f"  useful rows the channels never returned: {s['useful_missed_by_channels']}")
        for cat, c in s["by_category"].items():
            lines.append(
                f"    {cat:18s} n={c['fixtures']:3d} P@k={_fmt(c['precision_at_k'])} nDCG@5={_fmt(c['ndcg_at_5'])} "
                f"irr={_fmt(c['irrelevant_kept_rate'])} rec={_fmt(c['relevant_recall'])} abst={_fmt(c['abstain_correct'])}"
            )
    return "\n".join(lines)


def format_rerank_text(report: dict) -> str:
    lines = [f"embedding: {report.get('embedding_mode')}  cross-encoder: {report.get('model')}"]
    for run, s in report["runs"].items():
        lines.append("")
        lines.append(
            f"[{run}] pairs={s['pairs']} missing={s['missing']} AUC relevant>irrelevant: "
            f"cross-encoder {s['auc_ce_relevant_vs_irrelevant']} embedding {s['auc_semantic_relevant_vs_irrelevant']}; "
            f"useful>irrelevant: cross-encoder {s['auc_ce_useful_vs_irrelevant']}"
        )
        for lab in ("relevant", "partial", "irrelevant"):
            p = s["ce"].get(lab)
            if p:
                lines.append(
                    f"  ce {lab:10s} n={p['n']:4d} min={p['min']:7.2f} p10={p['p10']:7.2f} "
                    f"p50={p['p50']:7.2f} p90={p['p90']:7.2f} max={p['max']:7.2f}"
                )
        lines.append(f"  rows at or above: {s['at_or_above']}")
        for key in ("best_ce_abstain_cases", "best_useful_ce_answerable_cases"):
            p = s.get(key)
            if p:
                lines.append(f"  {key}: n={p['n']} min={p['min']} p10={p['p10']} p50={p['p50']} p90={p['p90']} max={p['max']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from core.knowledge import library_memory_eval as lme

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--capture", required=True)
    parser.add_argument("--corpus", default=str(lme.DEFAULT_CORPUS))
    parser.add_argument("--stage", default="")
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    parser.add_argument("--out", default=None, help="write the report to this file")
    parser.add_argument("--compare", default=None, help="baseline report: list rule-7 regressions")
    parser.add_argument(
        "--rerank-scores", default=None, help="recorded cross-encoder scores: report their distributions (S34)"
    )
    args = parser.parse_args(argv)

    corpus = lme.load_corpus(args.corpus)
    capture = lme.load_json(args.capture)
    fp = lme.corpus_fingerprint(args.corpus)
    if capture.get("corpus_fingerprint") != fp:
        print(f"capture fingerprint {capture.get('corpus_fingerprint')} != corpus {fp}; re-capture", file=sys.stderr)
        return 2
    if args.rerank_scores:
        report = lme.rerank_report(capture, corpus, lme.load_json(args.rerank_scores))
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=1, ensure_ascii=False) if args.json else format_rerank_text(report))
        return 0
    report = lme.build_report(capture, corpus, stage=args.stage)

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False) if args.json else format_text(report))

    if args.compare:
        regressions = lme.compare_reports(report, lme.load_json(args.compare))
        print(f"\nregressions vs {args.compare}: {len(regressions)}")
        for r in regressions:
            print(f"  {r['run']} {r['group']} {r['metric']}: {r['baseline']} -> {r['current']} (tolerance {r['tolerance']})")
        return 1 if regressions else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
