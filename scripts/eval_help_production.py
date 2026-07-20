#!/usr/bin/env python3
"""Run production-path help retrieval eval via rag_search."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock

if "lancedb" not in sys.modules:
    sys.modules["lancedb"] = MagicMock()
if "pyarrow" not in sys.modules:
    sys.modules["pyarrow"] = MagicMock()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.help_production_eval import (  # noqa: E402
    PRODUCTION_RAG_POOL_TARGET,
    PRODUCTION_TOP1_TARGET,
    PRODUCTION_TOP3_TARGET,
    assert_production_targets,
    evaluate_production_help_retrieval,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locale", default="en")
    args = parser.parse_args()

    summary = evaluate_production_help_retrieval(locale=args.locale)
    positive = summary.total - summary.negative_total
    rag_pool_rate = (
        summary.rag_pool_hits / summary.rag_pool_total
        if summary.rag_pool_total
        else 1.0
    )
    print(
        f"Production help eval ({summary.total} cases): "
        f"lexical top-1 {summary.top1_hits}/{positive} ({summary.top1_rate:.1%}), "
        f"top-3 {summary.top3_hits}/{positive} ({summary.top3_rate:.1%}), "
        f"rag-pool {summary.rag_pool_hits}/{summary.rag_pool_total} "
        f"({rag_pool_rate:.1%}), "
        f"targets rag-pool>={PRODUCTION_RAG_POOL_TARGET:.0%} "
        f"top-1>={PRODUCTION_TOP1_TARGET:.0%} top-3>={PRODUCTION_TOP3_TARGET:.0%}"
    )
    try:
        assert_production_targets(summary)
    except AssertionError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("Production help retrieval targets met.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
