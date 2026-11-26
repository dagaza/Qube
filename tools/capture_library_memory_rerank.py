#!/usr/bin/env python3
"""Record cross-encoder scores for the library and memory corpus (plan S34).

Scores every candidate of every run of the committed captures (the pool the
S34 shadow pass sees: every vector and FTS row, before the floors) with the
S3 cross-encoder, once per (question, passage), and writes the scores for
``tools/evaluate_library_memory.py --rerank-scores`` and for offline replays.

Run it under a scratch HOME with the reranker copied in (plan §7):

  HOME=<scratch> HF_HUB_OFFLINE=1 .venv/bin/python tools/capture_library_memory_rerank.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MODES = ("balanced", "fast", "power")


def main(argv: list[str] | None = None) -> int:
    from core.knowledge import library_memory_eval as lme
    from core.knowledge.ranking import rerank

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", default=str(lme.DEFAULT_CORPUS))
    parser.add_argument("--captures-dir", default=str(_ROOT / "eval" / "fixtures" / "library_memory"))
    parser.add_argument("--out", default=str(lme.DEFAULT_RERANK_SCORES))
    args = parser.parse_args(argv)

    reason = rerank.unavailable_reason()
    if reason is not None:
        print(f"reranker unavailable: {reason}", file=sys.stderr)
        return 2
    corpus = lme.load_corpus(args.corpus)
    captures = [lme.load_json(Path(args.captures_dir) / f"captured_{m}.json") for m in MODES]

    def score(question: str, passages: list[str]):
        result = rerank.rerank_texts(question, passages)
        return None if result is None else result.scores

    recorded = lme.record_rerank_scores(
        captures, corpus, score, model=rerank.RERANKER_MODEL_ID, text_chars=rerank.DEFAULT_TEXT_CHARS
    )
    Path(args.out).write_text(json.dumps(recorded, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    pairs = sum(len(v) for v in recorded["scores"].values())
    print(f"recorded {pairs} scores for {len(recorded['scores'])} questions -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
