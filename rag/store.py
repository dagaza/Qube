# --- rag/store.py ---
from __future__ import annotations

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
# Back-compat alias for tests and legacy imports (Balanced / jina default dim).
VECTOR_DIM = get_mode_spec(DEFAULT_MODE).vector_dim


def _schema_for_dim(vector_dim: int) -> pa.Schema:
    return pa.schema([
        pa.field("vector", pa.list_(pa.float32(), int(vector_dim))),
        pa.field("text", pa.utf8()),
        pa.field("source", pa.utf8()),
        pa.field("chunk_id", pa.int32()),
    ])


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
        try:
            return (
                self.table.search()
                .select(["text", "source", "chunk_id"])
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
        for chunk in chunks:
            vec = chunk.get("vector")
            if vec is not None and len(vec) != self.vector_dim:
                raise ValueError(
                    f"Vector dim mismatch: got {len(vec)}, expected {self.vector_dim}"
                )
        self.table.add(chunks)
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

    def reconstruct_document(self, source: str) -> str:
        import re

        try:
            results = (
                self.table.search([0.0] * self.vector_dim)
                .limit(10000)
                .where(f"source = '{source}'")
                .to_list()
            )
            if not results:
                return "Document contents not found in vector store."
            results.sort(key=lambda x: x["chunk_id"])
            reconstructed_text = "\n\n".join([r["text"] for r in results])
            reconstructed_text = re.sub(r"(?<!\n)\n(?!\n)", " ", reconstructed_text)
            reconstructed_text = re.sub(r" +", " ", reconstructed_text)
            return reconstructed_text
        except Exception as e:
            logger.error(f"Failed to reconstruct {source}: {e}")
            return f"Error loading document: {str(e)}"
