"""
Seed LanceDB with library-eval fixtures using the production ingest pipeline.

Used by ``tools/evaluate_library_chunking.py`` (Phase 0 lite baseline harness).
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from core.chunking.chunk_metadata import chunk_record_to_meta_json
from core.chunking.embed_context import library_chunk_embed_text
from core.chunking.ingest_pipeline import chunk_document_for_ingest
from core.knowledge.document.builders.library_builder import build_document_from_path
from core.paths import install_root
from rag.embedder import MAX_EMBED_CHARS

logger = logging.getLogger("Qube.LibraryEvalSeed")

SEED_SCHEMA = "qube.library_eval_seed.v1"
SEED_PIPELINE = "library_ingest_v2"
EVAL_LIBRARY_PREFIX = "eval_"
FIXTURES_DIR = install_root() / "eval" / "fixtures"
LIBRARY_DIR = FIXTURES_DIR / "library"


def default_eval_lancedb_dir() -> Path:
    return install_root() / "eval" / ".lancedb"


def _manifest_path(db_dir: Path) -> Path:
    return db_dir / ".library_eval_seed.json"


def _fixture_fingerprint() -> str:
    parts: list[bytes] = [SEED_PIPELINE.encode()]
    if LIBRARY_DIR.is_dir():
        for path in sorted(LIBRARY_DIR.glob("*.md")):
            parts.append(path.name.encode())
            parts.append(path.read_bytes())
    digest = hashlib.sha256(b"".join(parts)).hexdigest()
    return digest[:16]


def list_library_fixture_paths() -> list[Path]:
    if not LIBRARY_DIR.is_dir():
        raise FileNotFoundError(f"library fixtures not found: {LIBRARY_DIR}")
    paths = sorted(LIBRARY_DIR.glob("*.md"))
    if not paths:
        raise ValueError(f"no .md fixtures under {LIBRARY_DIR}")
    return paths


def _source_name(path: Path) -> str:
    if path.name.startswith("decoy_"):
        return path.name
    if path.name.startswith(EVAL_LIBRARY_PREFIX):
        return path.name
    return f"{EVAL_LIBRARY_PREFIX}{path.name}"


def is_library_eval_seeded(db_dir: Path) -> bool:
    manifest = _manifest_path(db_dir)
    if not manifest.is_file():
        return False
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        data.get("schema") == SEED_SCHEMA
        and data.get("pipeline") == SEED_PIPELINE
        and data.get("fingerprint") == _fixture_fingerprint()
    )


def _is_safe_force_purge_dir(db_dir: Path) -> bool:
    db_dir = db_dir.resolve()
    eval_root = (install_root() / "eval").resolve()
    return eval_root in db_dir.parents


def _purge_library_fixture_rows(store: Any) -> None:
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
        if src.startswith(EVAL_LIBRARY_PREFIX) or src.startswith("decoy_"):
            try:
                store.delete_document(src)
            except Exception as exc:
                logger.warning("Failed to purge %s: %s", src, exc)

    manifest = _manifest_path(db_dir)
    if manifest.is_file():
        manifest.unlink(missing_ok=True)


def seed_library_eval_corpus(
    store: Any,
    embedder: Any,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """
    Index eval fixture documents through ``Document`` → ``ChunkRecord`` ingest.

    Returns summary with chunk counts, timing, and structural metadata coverage.
    """
    db_dir = Path(getattr(store, "db_path", default_eval_lancedb_dir()))
    fingerprint = _fixture_fingerprint()

    if not force and is_library_eval_seeded(db_dir):
        logger.info("Library eval corpus already seeded (fingerprint=%s); skipping", fingerprint)
        return {"skipped": True, "fingerprint": fingerprint}

    if force:
        _purge_library_fixture_rows(store)

    library_paths = list_library_fixture_paths()
    started = time.perf_counter()

    doc_chunks = 0
    doc_files = 0
    chunks_with_meta = 0
    per_doc_stats: list[dict[str, Any]] = []

    batch_size = 32
    for path in library_paths:
        source = _source_name(path)
        if not force and store.source_exists(source):
            logger.debug("Skipping existing library source: %s", source)
            continue

        document = build_document_from_path(path)
        chunk_records = chunk_document_for_ingest(document)
        if not chunk_records:
            logger.warning("No chunks produced for %s", source)
            continue

        embed_inputs = [
            library_chunk_embed_text(
                source,
                record.body,
                section_heading=record.heading,
                breadcrumb=record.breadcrumb,
            )[:MAX_EMBED_CHARS]
            for record in chunk_records
        ]

        records: list[dict[str, Any]] = []
        for b_start in range(0, len(chunk_records), batch_size):
            batch_records = chunk_records[b_start : b_start + batch_size]
            batch_embed = embed_inputs[b_start : b_start + batch_size]
            vectors = embedder.embed(batch_embed)
            for j, (record, vector) in enumerate(zip(batch_records, vectors)):
                meta_json = chunk_record_to_meta_json(record)
                if meta_json:
                    chunks_with_meta += 1
                records.append({
                    "vector": vector.tolist(),
                    "text": record.body[:MAX_EMBED_CHARS],
                    "source": source,
                    "chunk_id": b_start + j,
                    "meta_json": meta_json,
                })

        store.add_chunks(records)
        char_lengths = [len(r["text"]) for r in records]
        per_doc_stats.append({
            "source": source,
            "chunks": len(records),
            "avg_chars": round(sum(char_lengths) / len(char_lengths), 1),
            "avg_est_tokens": round(sum(char_lengths) / len(char_lengths) / 4.0, 1),
            "meta_coverage": round(
                sum(1 for r in records if r.get("meta_json")) / len(records),
                3,
            ),
        })
        doc_chunks += len(records)
        doc_files += 1
        logger.info("Indexed %s (%d chunks)", source, len(records))

    elapsed_s = round(time.perf_counter() - started, 3)
    summary = {
        "skipped": False,
        "fingerprint": fingerprint,
        "pipeline": SEED_PIPELINE,
        "library_files": doc_files,
        "library_chunks": doc_chunks,
        "chunks_with_meta_json": chunks_with_meta,
        "meta_json_coverage": round(chunks_with_meta / doc_chunks, 3) if doc_chunks else 0.0,
        "ingest_elapsed_s": elapsed_s,
        "per_document": per_doc_stats,
    }

    db_dir.mkdir(parents=True, exist_ok=True)
    _manifest_path(db_dir).write_text(
        json.dumps({"schema": SEED_SCHEMA, **summary}, indent=2),
        encoding="utf-8",
    )
    return summary
