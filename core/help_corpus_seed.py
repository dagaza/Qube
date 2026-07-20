"""Seed bundled help corpus documents into Library (Qube folder) + LanceDB."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from core import __version__ as app_version_module
from core.help_corpus_manifest import (
    HELP_DOC_SOURCE_PREFIX,
    bundled_help_locale_dir,
    bundled_help_manifest_path,
    help_doc_source,
    iter_manifest_documents,
    load_manifest,
    manifest_is_compatible_with_app,
)
from core.paths import user_data_root
from core.help_corpus_text import help_chunk_embed_text
from core.help_markdown_chunker import chunk_help_markdown

logger = logging.getLogger("Qube.HelpCorpusSeed")

STATE_SCHEMA = "qube.help_corpus_state.v1"
_MAX_EMBED_CHARS = 2400  # match rag.embed_utils.MAX_EMBED_CHARS


def user_help_corpus_state_path() -> Path:
    return user_data_root() / "help_corpus_state.json"


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_state() -> dict[str, Any]:
    path = user_help_corpus_state_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_state(state: dict[str, Any]) -> None:
    path = user_help_corpus_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _manifest_doc_paths(manifest: dict[str, Any]) -> dict[str, str]:
    """Map LanceDB source -> relative composed path."""
    out: dict[str, str] = {}
    for doc in iter_manifest_documents(manifest):
        rel = str(doc["path"]).replace("\\", "/")
        out[help_doc_source(rel)] = rel
    return out


def should_seed_help_corpus(
    manifest: dict[str, Any] | None = None,
    *,
    locale: str = "en",
    app_version: str | None = None,
) -> tuple[bool, str]:
    try:
        data = manifest or load_manifest(locale=locale)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        return False, str(exc)

    version = app_version or app_version_module.__version__
    ok, reason = manifest_is_compatible_with_app(data, version)
    if not ok:
        return False, reason

    corpus_version = str(data.get("corpus_version") or "")
    state = _load_state()
    if state.get("schema") != STATE_SCHEMA:
        return True, "no prior help corpus state"

    if state.get("corpus_version") != corpus_version:
        return True, f"corpus_version changed ({state.get('corpus_version')} -> {corpus_version})"

    declared = _manifest_doc_paths(data)
    stored_hashes = state.get("doc_hashes") or {}
    if not isinstance(stored_hashes, dict):
        return True, "invalid stored doc_hashes"

    root = bundled_help_locale_dir(locale)
    for source, rel in declared.items():
        path = root / rel
        if not path.is_file():
            return True, f"missing bundled doc: {rel}"
        digest = _file_hash(path)
        if stored_hashes.get(source) != digest:
            return True, f"content changed: {rel}"

    stale_sources = set(stored_hashes) - set(declared)
    if stale_sources:
        return True, f"removed docs: {', '.join(sorted(stale_sources))}"

    return False, "help corpus up to date"


def seed_help_corpus(
    store: Any,
    embedder: Any,
    db_manager: Any,
    *,
    locale: str = "en",
    force: bool = False,
) -> dict[str, Any]:
    """
    Index bundled help documents into LanceDB and register them in SQLite.

    Returns a summary dict with counts and skip reason when applicable.
    """
    manifest_path = bundled_help_manifest_path(locale)
    if not manifest_path.is_file():
        logger.info("No bundled help manifest at %s; skipping seed", manifest_path)
        return {"skipped": True, "reason": "manifest missing"}

    manifest = load_manifest(manifest_path)
    if not force:
        need_seed, reason = should_seed_help_corpus(manifest, locale=locale)
        if not need_seed:
            logger.info("Help corpus seed skipped: %s", reason)
            return {"skipped": True, "reason": reason}

    if embedder is None:
        logger.warning("Help corpus seed skipped: embedder not loaded")
        return {"skipped": True, "reason": "embedder unavailable"}

    ok, compat_reason = manifest_is_compatible_with_app(manifest, app_version_module.__version__)
    if not ok:
        logger.info("Help corpus seed skipped: %s", compat_reason)
        return {"skipped": True, "reason": compat_reason}

    root = bundled_help_locale_dir(locale)
    folder_id = db_manager.get_qube_library_folder_id()
    source_to_rel = _manifest_doc_paths(manifest)
    doc_by_source = {
        help_doc_source(str(doc["path"])): doc for doc in iter_manifest_documents(manifest)
    }
    state = _load_state()
    previous_hashes = state.get("doc_hashes") if isinstance(state.get("doc_hashes"), dict) else {}

    indexed = 0
    reindexed = 0
    removed = 0
    chunk_total = 0
    new_hashes: dict[str, str] = {}

    for source, rel in sorted(source_to_rel.items()):
        path = root / rel
        digest = _file_hash(path)
        new_hashes[source] = digest

        if not force and previous_hashes.get(source) == digest and store.source_exists(source):
            logger.debug("Help doc unchanged: %s", source)
            continue

        text = path.read_text(encoding="utf-8").strip()
        if not text:
            logger.warning("Skipping empty help doc: %s", rel)
            continue

        if store.source_exists(source):
            store.delete_document(source)
            reindexed += 1
        try:
            db_manager.delete_document_metadata(source)
        except Exception as exc:
            logger.debug("No prior SQLite metadata for %s: %s", source, exc)

        chunks = [c[:_MAX_EMBED_CHARS] for c in chunk_help_markdown(text)]
        if not chunks:
            logger.warning("No chunks produced for help doc: %s", rel)
            continue

        doc_meta = doc_by_source.get(source, {"path": rel, "title": rel})
        embed_inputs = [
            help_chunk_embed_text(doc_meta, chunk)[:_MAX_EMBED_CHARS] for chunk in chunks
        ]
        vectors = embedder.embed(embed_inputs)
        records = []
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            records.append(
                {
                    "vector": vector.tolist(),
                    "text": chunk,
                    "source": source,
                    "chunk_id": idx,
                }
            )
        store.add_chunks(records)

        file_size_kb = round(path.stat().st_size / 1024, 2)
        db_manager.add_document_metadata(
            source,
            file_size_kb,
            len(records),
            folder_id=folder_id,
        )

        indexed += 1
        chunk_total += len(records)
        logger.info("Indexed help doc %s (%d chunks)", source, len(records))

    stale_sources = set(previous_hashes) - set(source_to_rel)
    for source in sorted(stale_sources):
        if store.source_exists(source):
            store.delete_document(source)
        try:
            db_manager.delete_document_metadata(source)
        except Exception as exc:
            logger.debug("Failed deleting stale help metadata %s: %s", source, exc)
        removed += 1
        logger.info("Removed stale help doc: %s", source)

    state = {
        "schema": STATE_SCHEMA,
        "locale": locale,
        "corpus_version": str(manifest.get("corpus_version") or ""),
        "collection_id": str(manifest.get("collection_id") or ""),
        "doc_hashes": new_hashes,
        "seeded_app_version": app_version_module.__version__,
    }
    _save_state(state)

    return {
        "skipped": False,
        "indexed": indexed,
        "reindexed": reindexed,
        "removed": removed,
        "chunks": chunk_total,
        "corpus_version": state["corpus_version"],
    }


def seed_help_corpus_if_needed(store: Any, embedder: Any, db_manager: Any) -> dict[str, Any]:
    """Convenience wrapper for startup hook."""
    return seed_help_corpus(store, embedder, db_manager, force=False)


def list_help_corpus_sources(manifest: dict[str, Any] | None = None) -> list[str]:
    data = manifest or load_manifest()
    return sorted(_manifest_doc_paths(data))


def is_help_corpus_source(source: str) -> bool:
    return str(source or "").startswith(HELP_DOC_SOURCE_PREFIX)
