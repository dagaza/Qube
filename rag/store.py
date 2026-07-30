# --- rag/store.py ---
from __future__ import annotations

import json
import lancedb
import pyarrow as pa
import numpy as np
from pathlib import Path
import logging
import os

from core.embedding_modes import DEFAULT_MODE, get_mode_spec
from core.paths import default_lancedb_dir

logger = logging.getLogger("Qube.RAG.Store")

DB_PATH = default_lancedb_dir()
TABLE_NAME = "documents"
META_JSON_COLUMN = "meta_json"
# Back-compat alias for tests and legacy imports (Balanced / jina default dim).
VECTOR_DIM = get_mode_spec(DEFAULT_MODE).vector_dim


def _schema_for_dim(vector_dim: int) -> pa.Schema:
    return pa.schema([
        pa.field("vector", pa.list_(pa.float32(), int(vector_dim))),
        pa.field("text", pa.utf8()),
        pa.field("source", pa.utf8()),
        pa.field("chunk_id", pa.int32()),
        pa.field(META_JSON_COLUMN, pa.utf8()),
    ])


def _normalize_chunk_row(chunk: dict) -> dict:
    row = dict(chunk)
    meta = row.get(META_JSON_COLUMN)
    if meta is None:
        row[META_JSON_COLUMN] = ""
    elif isinstance(meta, dict):
        row[META_JSON_COLUMN] = json.dumps(meta, ensure_ascii=False, separators=(",", ":"))
    else:
        row[META_JSON_COLUMN] = str(meta)
    return row


class DocumentStore:
    def __init__(
        self,
        db_path: Path | os.PathLike[str] | None = None,
        *,
        expected_vector_dim: int | None = None,
        quiet: bool = False,
    ):
        self.db_path = Path(db_path) if db_path is not None else DB_PATH
        self.vector_dim = int(
            expected_vector_dim or get_mode_spec(DEFAULT_MODE).vector_dim
        )
        self.dim_mismatch = False
        self.quiet = quiet
        if not quiet:
            print(f"\n🔍 VECTOR STORE DIAGNOSTIC")
            print(f"📍 Database Absolute Path: {self.db_path}")

        self.db_path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_path))
        self._open_or_create_table()

        try:
            from core.memory_source_migration import migrate_legacy_memory_sources
            migrate_legacy_memory_sources(self)
        except Exception as e:
            logger.warning("Legacy memory source migration skipped: %s", e)

        self._ensure_meta_json_column()

    def _ensure_meta_json_column(self) -> None:
        """Add ``meta_json`` to existing tables when missing (Phase 3 migration)."""
        try:
            if META_JSON_COLUMN in self.table.schema.names:
                return
        except Exception as exc:
            logger.debug("Could not inspect LanceDB schema for meta_json: %s", exc)
            return

        try:
            if hasattr(self.table, "add_columns"):
                self.table.add_columns({META_JSON_COLUMN: "''"})
                logger.info("Added %s column to LanceDB table %s", META_JSON_COLUMN, TABLE_NAME)
                return
        except Exception as exc:
            logger.debug("LanceDB add_columns for meta_json skipped: %s", exc)

        logger.info(
            "%s will be populated as new rows are ingested (legacy rows remain blank).",
            META_JSON_COLUMN,
        )

    def _open_or_create_table(self) -> None:
        if TABLE_NAME in self.db.table_names():
            self.table = self.db.open_table(TABLE_NAME)
            existing_dim = self.table.schema.field("vector").type.list_size
            if not self.quiet:
                print(f"📏 Table '{TABLE_NAME}' found. Dimension: {existing_dim}")
            if existing_dim != self.vector_dim:
                logger.warning(
                    "Vector dimension mismatch: table=%s expected=%s",
                    existing_dim,
                    self.vector_dim,
                )
                self.dim_mismatch = True
        else:
            self.table = self.db.create_table(
                TABLE_NAME, schema=_schema_for_dim(self.vector_dim)
            )
            if not self.quiet:
                print(
                    f"✨ Created fresh '{TABLE_NAME}' table with "
                    f"{self.vector_dim} dimensions."
                )

    def recreate_for_dim(self, vector_dim: int) -> None:
        """Drop and recreate the documents table for a new embedding dimension."""
        self.vector_dim = int(vector_dim)
        self.dim_mismatch = False
        if TABLE_NAME in self.db.table_names():
            self.db.drop_table(TABLE_NAME)
        self.table = self.db.create_table(
            TABLE_NAME, schema=_schema_for_dim(self.vector_dim)
        )
        logger.info("Recreated LanceDB table %s with dim=%s", TABLE_NAME, self.vector_dim)

    def export_all_rows(self) -> list[dict]:
        """Export indexed rows before a wipe/reindex."""
        columns = ["text", "source", "chunk_id"]
        try:
            if META_JSON_COLUMN in self.table.schema.names:
                columns.append(META_JSON_COLUMN)
        except Exception:
            pass
        try:
            return (
                self.table.search()
                .select(columns)
                .limit(1_000_000)
                .to_list()
            )
        except Exception as exc:
            logger.warning("Failed to export LanceDB rows: %s", exc)
            return []

    def rebuild_fts_index(self) -> None:
        try:
            self.table.create_fts_index("text", replace=True)
            logger.info("FTS keyword index built successfully.")
        except Exception as exc:
            logger.warning(
                "Could not build FTS index (ensure 'tantivy' is installed): %s",
                exc,
            )

    def get_all_indexed_sources(self) -> list[str]:
        try:
            results = self.table.search().select(["source"]).limit(100000).to_list()
            unique_sources = {item["source"] for item in results if "source" in item}
            return list(unique_sources)
        except Exception as e:
            logger.warning("Failed to query LanceDB for unique sources: %s", e)
        return []

    def find_sources_matching_text(self, query: str) -> set[str]:
        q = (query or "").strip()
        if not q:
            return set()
        needle = q.lower()
        matched: set[str] = set()
        try:
            rows = (
                self.table.search(q, query_type="fts")
                .select(["source"])
                .limit(100_000)
                .to_list()
            )
            for row in rows:
                src = row.get("source")
                if src:
                    matched.add(str(src))
        except Exception as e:
            logger.debug("LanceDB FTS library search skipped: %s", e)
        if matched:
            return matched
        dummy = [0.0] * self.vector_dim
        try:
            for src in self.get_all_indexed_sources():
                esc = str(src).replace("'", "''")
                rows = (
                    self.table.search(dummy)
                    .limit(50_000)
                    .where(f"source = '{esc}'")
                    .select(["text"])
                    .to_list()
                )
                if any(needle in (r.get("text") or "").lower() for r in rows):
                    matched.add(str(src))
        except Exception as e:
            logger.warning("Library substring scan failed: %s", e)
        return matched

    def add_chunks(self, chunks: list[dict], *, rebuild_fts: bool = True):
        normalized = [_normalize_chunk_row(chunk) for chunk in chunks]
        for chunk in normalized:
            vec = chunk.get("vector")
            if vec is not None and len(vec) != self.vector_dim:
                raise ValueError(
                    f"Vector dim mismatch: got {len(vec)}, expected {self.vector_dim}"
                )
        self.table.add(normalized)
        if rebuild_fts:
            self.rebuild_fts_index()

    def search(self, query_vector: np.ndarray, query_text: str = None, top_k: int = 5) -> list[dict]:
        if query_text:
            try:
                return (
                    self.table.search(query_type="hybrid")
                    .vector(query_vector)
                    .text(query_text)
                    .limit(top_k)
                    .select(["text", "source", "chunk_id"])
                    .to_list()
                )
            except Exception as e:
                logger.warning(f"Hybrid search failed, falling back to pure vector: {e}")

        return (
            self.table.search(query_vector)
            .limit(top_k)
            .select(["text", "source", "chunk_id"])
            .to_list()
        )

    def source_exists(self, source: str) -> bool:
        try:
            result = (
                self.table.search([0.0] * self.vector_dim)
                .limit(1)
                .where(f"source = '{source}'")
                .to_list()
            )
            return len(result) > 0
        except Exception:
            return False

    def delete_document(self, source_name: str):
        from datetime import timedelta

        try:
            self.table.delete(f'source = "{source_name}"')
            logger.info(f"Logical delete complete for '{source_name}'.")
            try:
                if hasattr(self.table, "optimize"):
                    self.table.optimize(cleanup_older_than=timedelta(seconds=0))
            except Exception as cleanup_err:
                logger.warning(
                    "Rows deleted, but physical disk cleanup was bypassed: %s",
                    cleanup_err,
                )
        except Exception as e:
            logger.error(f"Failed to delete vectors for '{source_name}' from LanceDB: {e}")

    def rename_document(self, old_source: str, new_source: str) -> bool:
        try:
            self.table.update(where=f"source = '{old_source}'", values={"source": new_source})
            logger.info(f"Successfully renamed vectors from '{old_source}' to '{new_source}'.")
            return True
        except Exception as e:
            logger.warning(f"Native update failed or not supported, attempting manual rewrite: {e}")
            try:
                records = (
                    self.table.search([0.0] * self.vector_dim)
                    .limit(100000)
                    .where(f"source = '{old_source}'")
                    .to_list()
                )
                if not records:
                    logger.warning(f"No vectors found for '{old_source}' during rename.")
                    return False
                for r in records:
                    r["source"] = new_source
                    r.pop("_distance", None)
                self.table.add(records)
                self.table.delete(f"source = '{old_source}'")
                logger.info(f"Fallback rename complete for '{new_source}'.")
                return True
            except Exception as fallback_err:
                logger.error(
                    f"CRITICAL: Complete failure to rename document in Vector Store: {fallback_err}"
                )
                return False

    def _fetch_document_chunks(self, source: str) -> list[dict]:
        results = (
            self.table.search([0.0] * self.vector_dim)
            .limit(10000)
            .where(f"source = '{source}'")
            .to_list()
        )
        results.sort(key=lambda x: x["chunk_id"])
        return results

    def reconstruct_document(self, source: str) -> str:
        from core.chunking.library_preview import build_library_preview_plain

        try:
            results = self._fetch_document_chunks(source)
            return build_library_preview_plain(results)
        except Exception as e:
            logger.error(f"Failed to reconstruct {source}: {e}")
            return f"Error loading document: {str(e)}"

    def reconstruct_document_for_preview(
        self,
        source: str,
        *,
        breadcrumb_color: str,
        body_color: str,
        font_pt: float = 12.0,
    ) -> tuple[str, bool]:
        """Return stitched preview content and whether it is HTML (structured metadata)."""
        from core.chunking.library_preview import build_library_preview

        try:
            results = self._fetch_document_chunks(source)
            return build_library_preview(
                results,
                breadcrumb_color=breadcrumb_color,
                body_color=body_color,
                font_pt=font_pt,
            )
        except Exception as e:
            logger.error(f"Failed to reconstruct preview for {source}: {e}")
            return f"Error loading document: {str(e)}", False
