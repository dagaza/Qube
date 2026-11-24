#!/usr/bin/env python3
"""Offline eval of post-generation citation grounding scores (plan S11).

Scores every labelled (claim, source) pair in the corpus with the S3
cross-encoder in each claim-unit / source-mode combination and reports how well
the score separates supported from unsupported citations (AUC, Spearman), what
a "best source below threshold" flag would catch, and the per-turn latency of
the production path on the stored answers.

Needs the reranker model on disk (``HF_HUB_OFFLINE=1`` is fine). Exit status 1
when the reranker is missing or, with ``--compare``, when the default mode's
AUC falls by more than 0.02 from the baseline.

Examples::

    HF_HUB_OFFLINE=1 .venv/bin/python tools/evaluate_citation_grounding.py
    HF_HUB_OFFLINE=1 .venv/bin/python tools/evaluate_citation_grounding.py --json > baseline.json
    HF_HUB_OFFLINE=1 .venv/bin/python tools/evaluate_citation_grounding.py --compare eval/citation_grounding/baselines/s11.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import citation_grounding as cg  # noqa: E402

DEFAULT_CORPUS = ROOT / "eval" / "citation_grounding" / "v1_labelled_pairs.json"
LABEL_VALUE = {"unsupported": 0, "partial": 1, "supported": 2}
THRESHOLDS = (-2.0, 0.0, 2.0)
AUC_TOLERANCE = 0.02
MODES = [(u, m) for u in (cg.UNIT_SENTENCE, cg.UNIT_SPAN) for m in (cg.MODE_HEAD, cg.MODE_WINDOW)]


def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


def _r(v):
    return round(v, 3) if isinstance(v, float) else v


def pair_scores(pairs: list[dict], unit: str, mode: str) -> list[float]:
    from core.knowledge.ranking import rerank

    out = []
    for p in pairs:
        question = p["span"] if unit == cg.UNIT_SPAN and p.get("span") else p["claim"]
        docs = cg.source_windows({"filename": p.get("source_title"), "content": p.get("source_text")}, mode=mode)
        result = rerank.rerank_texts(question, docs) if docs else None
        if result is None:
            raise SystemExit(f"reranker unavailable: {rerank.unavailable_reason(question)}")
        out.append(max(result.scores))
    return out


def mode_metrics(pairs: list[dict], scores: list[float]) -> dict:
    labels = [LABEL_VALUE[p["label"]] for p in pairs]
    idx = range(len(pairs))
    cited = [i for i in idx if p_kind(pairs[i]) == "cited"]
    row = {
        "auc_supported_vs_unsupported": auc([scores[i] for i in idx if labels[i] == 2], [scores[i] for i in idx if labels[i] == 0]),
        "auc_any_support_vs_unsupported": auc([scores[i] for i in idx if labels[i] >= 1], [scores[i] for i in idx if labels[i] == 0]),
        "auc_cited_only": auc([scores[i] for i in cited if labels[i] == 2], [scores[i] for i in cited if labels[i] == 0]),
        "spearman": spearman(scores, labels),
        "spearman_cited_only": spearman([scores[i] for i in cited], [labels[i] for i in cited]),
    }
    unsupported_cited = sum(1 for i in cited if labels[i] == 0)
    for t in THRESHOLDS:
        flagged = [i for i in cited if scores[i] < t]
        row[f"cited_below_{t:g}"] = {
            "flagged": len(flagged),
            "unsupported": sum(labels[i] == 0 for i in flagged),
            "partial": sum(labels[i] == 1 for i in flagged),
            "supported": sum(labels[i] == 2 for i in flagged),
            "recall_unsupported": _r(sum(labels[i] == 0 for i in flagged) / unsupported_cited) if unsupported_cited else None,
        }
    return {k: _r(v) for k, v in row.items()}


def p_kind(p: dict) -> str:
    return str(p.get("kind") or "cited")


def turn_latency(answers: list[dict]) -> dict:
    lat: list[float] = []
    pairs: list[int] = []
    texts: list[int] = []
    for a in answers:
        report = cg.score_citation_grounding(a["answer"], a["sources"])
        if report.status != "scored":
            continue
        lat.append(report.latency_ms)
        pairs.append(len(report.scored_pairs))
        by_id = {str(s["id"]): s for s in a["sources"]}
        texts.append(sum(len(cg.source_windows(by_id[p.source_id])) for p in report.scored_pairs if p.source_id in by_id))
    if not lat:
        return {"turns": 0}
    lat.sort()
    return {
        "turns": len(lat),
        "p50_ms": round(statistics.median(lat), 1),
        "p95_ms": round(lat[int(0.95 * (len(lat) - 1))], 1),
        "max_ms": round(lat[-1], 1),
        "pairs_p50": statistics.median(pairs),
        "pairs_max": max(pairs),
        "texts_max": max(texts),
    }


def evaluate(corpus: dict, *, latency: bool = True) -> dict:
    from core.knowledge.ranking import rerank

    pairs = corpus["pairs"]
    with patch("core.app_settings.get_knowledge_reranker_enabled", return_value=True):
        if rerank.unavailable_reason("warm up") is not None:
            raise SystemExit(f"reranker unavailable: {rerank.unavailable_reason('warm up')}")
        rerank.rerank_texts("warm up", ["warm up"])
        modes = {f"{u}/{m}": mode_metrics(pairs, pair_scores(pairs, u, m)) for u, m in MODES}
        out = {
            "corpus": corpus.get("id"),
            "pairs": len(pairs),
            "counts": corpus.get("counts"),
            "default_mode": f"{cg.DEFAULT_CLAIM_UNIT}/{cg.DEFAULT_MODE}",
            "modes": modes,
        }
        if latency:
            out["latency_default"] = turn_latency(corpus.get("answers") or [])
    return out


def compare(current: dict, baseline: dict) -> list[str]:
    key = current["default_mode"]
    cur = current["modes"][key]["auc_supported_vs_unsupported"]
    base = (baseline.get("modes") or {}).get(key, {}).get("auc_supported_vs_unsupported")
    if base is None or cur is None:
        return []
    if cur < base - AUC_TOLERANCE:
        return [f"{key} AUC {cur} < baseline {base} - {AUC_TOLERANCE}"]
    return []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-latency", action="store_true")
    ap.add_argument("--compare")
    args = ap.parse_args(argv)
    corpus = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    result = evaluate(corpus, latency=not args.no_latency)
    failures = compare(result, json.loads(Path(args.compare).read_text(encoding="utf-8"))) if args.compare else []
    if args.json:
        print(json.dumps(result, indent=1))
    else:
        print(f"{result['corpus']}: {result['pairs']} pairs {result['counts']}")
        for name, m in result["modes"].items():
            mark = "*" if name == result["default_mode"] else " "
            print(
                f"{mark} {name:16s} AUC(sup vs unsup) {m['auc_supported_vs_unsupported']}  "
                f"AUC(any vs unsup) {m['auc_any_support_vs_unsupported']}  "
                f"spearman {m['spearman']}  cited<0 {m['cited_below_0']}"
            )
        if "latency_default" in result:
            print(f"latency (default mode, per turn): {result['latency_default']}")
    for f in failures:
        print(f"REGRESSION: {f}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
