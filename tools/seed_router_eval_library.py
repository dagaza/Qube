#!/usr/bin/env python3
"""
Seed the router-eval fixture library and memories into an isolated LanceDB directory.

Examples:
  venv/bin/python tools/seed_router_eval_library.py
  venv/bin/python tools/seed_router_eval_library.py --lancedb-dir eval/.lancedb --force
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | [%(name)s] %(message)s",
    )

    from core.router_eval_seed import (
        default_eval_lancedb_dir,
        seed_router_eval_library,
    )
    from rag.embedder import EmbeddingModel
    from rag.store import DocumentStore

    parser = argparse.ArgumentParser(
        description="Seed router-eval fixture documents and memories into LanceDB"
    )
    parser.add_argument(
        "--lancedb-dir",
        type=Path,
        default=default_eval_lancedb_dir(),
        help="Target LanceDB directory (default: eval/.lancedb)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Purge prior eval rows and re-index all fixtures",
    )
    parser.add_argument(
        "--quiet-store",
        action="store_true",
        help="Suppress DocumentStore diagnostic prints",
    )
    args = parser.parse_args()

    db_dir = args.lancedb_dir.resolve()
    db_dir.mkdir(parents=True, exist_ok=True)

    embedder = EmbeddingModel()
    store = DocumentStore(db_dir, quiet=args.quiet_store)
    summary = seed_router_eval_library(store, embedder, force=args.force)

    print(json.dumps({"lancedb_dir": str(db_dir), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
