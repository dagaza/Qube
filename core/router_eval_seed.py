"""
Seed LanceDB with router-eval fixture library docs and synthetic memories.

Used by ``tools/seed_router_eval_library.py`` and ``tools/evaluate_router.py --seed-eval-library``.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from core.paths import install_root
from rag.chunker import chunk_text
from rag.embedder import MAX_EMBED_CHARS

logger = logging.getLogger("Qube.RouterEvalSeed")

SEED_SCHEMA = "qube.router_eval_seed.v1"
EVAL_LIBRARY_PREFIX = "eval_"
FIXTURES_DIR = install_root() / "eval" / "fixtures"
LIBRARY_DIR = FIXTURES_DIR / "library"
MEMORIES_FILE = FIXTURES_DIR / "memories.json"


def default_eval_lancedb_dir() -> Path:
    return install_root() / "eval" / ".lancedb"


def _fixture_fingerprint() -> str:
    parts: list[str] = []
    if MEMORIES_FILE.is_file():
        parts.append(MEMORIES_FILE.read_bytes())
    if LIBRARY_DIR.is_dir():
        for path in sorted(LIBRARY_DIR.glob("*.md")):
            parts.append(path.name.encode())
            parts.append(path.read_bytes())
    digest = hashlib.sha256(b"".join(parts)).hexdigest()
    return digest[:16]


def _manifest_path(db_dir: Path) -> Path:
    return db_dir / ".router_eval_seed.json"


def is_eval_library_seeded(db_dir: Path) -> bool:
    manifest = _manifest_path(db_dir)
    if not manifest.is_file():
        return False
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        data.get("schema") == SEED_SCHEMA
        and data.get("fingerprint") == _fixture_fingerprint()
    )


def load_memory_fixtures() -> list[dict[str, Any]]:
    if not MEMORIES_FILE.is_file():
        raise FileNotFoundError(f"memory fixtures not found: {MEMORIES_FILE}")
    data = json.loads(MEMORIES_FILE.read_text(encoding="utf-8"))
    rows = data.get("memories")
    if not isinstance(rows, list) or not rows:
        raise ValueError("memories.json must contain a non-empty 'memories' list")
    return rows


def list_library_fixture_paths() -> list[Path]:
    if not LIBRARY_DIR.is_dir():
        raise FileNotFoundError(f"library fixtures not found: {LIBRARY_DIR}")
    paths = sorted(LIBRARY_DIR.glob("*.md"))
    if not paths:
        raise ValueError(f"no .md fixtures under {LIBRARY_DIR}")
    return paths


def _memory_source(tier: str, key: str) -> str:
    return f"qube_memory::{tier}::{key}"


def seed_router_eval_library(
    store: Any,
    embedder: Any,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """
    Index eval fixture documents and memories into ``store``.

    Returns summary dict with chunk/memory counts.
    """
    db_dir = Path(getattr(store, "db_path", default_eval_lancedb_dir()))
    fingerprint = _fixture_fingerprint()

    if not force and is_eval_library_seeded(db_dir):
        logger.info("Eval library already seeded (fingerprint=%s); skipping", fingerprint)
        return {"skipped": True, "fingerprint": fingerprint}

    if force:
        _purge_eval_rows(store)

    library_paths = list_library_fixture_paths()
    memory_rows = load_memory_fixtures()

    doc_chunks = 0
    doc_files = 0
    for path in library_paths:
        source = (
            path.name
            if path.name.startswith(EVAL_LIBRARY_PREFIX)
            else f"{EVAL_LIBRARY_PREFIX}{path.name}"
        )
        if not force and store.source_exists(source):
            logger.debug("Skipping existing library source: %s", source)
            continue

        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        chunks = [c[:MAX_EMBED_CHARS] for c in chunk_text(text)]
        if not chunks:
            continue

        vectors = embedder.embed(chunks)
        records = []
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            records.append({
                "vector": vector.tolist(),
                "text": chunk,
                "source": source,
                "chunk_id": idx,
            })
        store.add_chunks(records)
        doc_chunks += len(records)
        doc_files += 1
        logger.info("Indexed %s (%d chunks)", source, len(records))

    mem_rows = 0
    mem_records = []
    for row in memory_rows:
        tier = str(row.get("tier") or "context").strip().lower()
        key = str(row.get("key") or row.get("id") or "fact").strip()
        content = str(row.get("content") or "").strip()
        if not content:
            continue

        category = str(row.get("category") or tier).strip().lower()
        source = _memory_source(tier, key)
        if not force and store.source_exists(source):
            logger.debug("Skipping existing memory source: %s", source)
            continue

        payload = {
            "content": content,
            "confidence": float(row.get("confidence", 0.9)),
            "category": category,
            "strength": int(row.get("strength", 1)),
            "decay": float(row.get("decay", 1.0)),
        }
        if row.get("provenance_quote"):
            payload["provenance_quote"] = str(row["provenance_quote"])

        vector = embedder.embed_query(content)
        mem_records.append({
            "vector": vector.tolist(),
            "text": json.dumps(payload),
            "source": source,
            "chunk_id": 0,
        })
        mem_rows += 1

    if mem_records:
        store.add_chunks(mem_records)
        logger.info("Indexed %d memory rows", mem_rows)

    db_dir.mkdir(parents=True, exist_ok=True)
    _manifest_path(db_dir).write_text(
        json.dumps(
            {
                "schema": SEED_SCHEMA,
                "fingerprint": fingerprint,
                "library_files": doc_files,
                "library_chunks": doc_chunks,
                "memory_rows": mem_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "skipped": False,
        "fingerprint": fingerprint,
        "library_files": doc_files,
        "library_chunks": doc_chunks,
        "memory_rows": mem_rows,
    }


def _is_safe_force_purge_dir(db_dir: Path) -> bool:
    """Only allow destructive re-seed inside the eval fixture database."""
    db_dir = db_dir.resolve()
    eval_root = (install_root() / "eval").resolve()
    return eval_root in db_dir.parents


def _purge_eval_rows(store: Any) -> None:
    """Remove prior eval fixture rows before a forced re-seed."""
    db_dir = Path(getattr(store, "db_path", default_eval_lancedb_dir()))
    if not _is_safe_force_purge_dir(db_dir):
        raise ValueError(
            f"refusing --force seed outside eval scope: {db_dir} "
            f"(use default {default_eval_lancedb_dir()})"
        )
    try:
        sources = store.get_all_indexed_sources()
    except Exception as exc:
        logger.warning("Could not list sources for purge: %s", exc)
        return

    for source in sources:
        src = str(source)
        if src.startswith(EVAL_LIBRARY_PREFIX) or src.startswith("qube_memory::"):
            try:
                store.delete_document(src)
            except Exception as exc:
                logger.warning("Failed to purge %s: %s", src, exc)

    manifest = _manifest_path(Path(getattr(store, "db_path", default_eval_lancedb_dir())))
    if manifest.is_file():
        manifest.unlink(missing_ok=True)
