#!/usr/bin/env python3
"""Capture the library and memory corpus with a real embedder (plan S33).

Builds each corpus store in a temporary LanceDB through the production write
paths, runs every case's production retrieval call against a recording table,
checks that an offline replay keeps exactly what the live call kept, and
writes the capture (no vectors) for ``tools/evaluate_library_memory.py``.

Run it under a scratch HOME with the search models copied in (plan §7):

  HOME=<scratch> HF_HUB_OFFLINE=1 .venv/bin/python tools/capture_library_memory_fixture.py \
      --embedding-mode balanced
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _default_out(mode: str) -> Path:
    return _ROOT / "eval" / "fixtures" / "library_memory" / f"captured_{mode}.json"


def main(argv: list[str] | None = None) -> int:
    from core.embedding_modes import MODE_IDS, get_mode_spec
    from core.knowledge import library_memory_eval as lme

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--embedding-mode", choices=MODE_IDS, default="balanced")
    parser.add_argument("--corpus", default=str(lme.DEFAULT_CORPUS))
    parser.add_argument("--out", default=None, help="capture file (default: eval/fixtures/library_memory/captured_<mode>.json)")
    parser.add_argument("--work-dir", default=None, help="where the temporary LanceDB stores go")
    parser.add_argument(
        "--now",
        type=float,
        default=None,
        help=(
            "capture clock in epoch seconds (default: now). Memory timestamps are this minus "
            "each memory's age, and FTS indexes them, so pass an earlier capture's "
            "captured_now to reproduce it exactly"
        ),
    )
    args = parser.parse_args(argv)

    corpus = lme.load_corpus(args.corpus)
    problems = lme.validate_corpus(corpus)
    if problems:
        print("corpus problems:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2

    from rag.embedder import EmbeddingModel

    embedder = EmbeddingModel(mode_id=args.embedding_mode)
    model = get_mode_spec(args.embedding_mode).fastembed_model

    with tempfile.TemporaryDirectory(dir=args.work_dir) as work:
        capture = lme.capture_corpus(
            corpus,
            Path(work),
            embedder,
            corpus_path=args.corpus,
            now=args.now,
            embedding_mode=args.embedding_mode,
            model_name=model,
        )

    chunks = capture["stores"]["full"]["chunks"]
    problems = lme.validate_corpus(corpus, chunks)
    if problems:
        print("label problems:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2

    mismatches = []
    for case in corpus["cases"]:
        for run in lme.case_runs(case):
            live = capture["cases"][case["id"]]["runs"][run]["kept"]
            replayed = lme.replay_run(capture, case, run)
            if replayed != live:
                mismatches.append(f"{case['id']}/{run}: live {live} replay {replayed}")
    if mismatches:
        print("replay does not reproduce the live calls:\n  " + "\n  ".join(mismatches), file=sys.stderr)
        return 3

    out = Path(args.out) if args.out else _default_out(args.embedding_mode)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(capture, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    runs = sum(len(c["runs"]) for c in capture["cases"].values())
    print(f"captured {len(capture['cases'])} cases, {runs} runs ({args.embedding_mode}, {model}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
