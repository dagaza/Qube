"""Background worker: wipe LanceDB and re-embed library + memory rows."""
from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtCore import QThread, pyqtSignal

from core.embedding_modes import ModeId, get_mode_spec, normalize_mode_id
from core.reindex_state import set_reindex_in_progress
from core.router_centroid_install import clear_router_embedding_state, install_router_centroids

logger = logging.getLogger("Qube.ReindexWorker")

_BATCH_SIZE = 16


class ReindexWorker(QThread):
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    reindex_complete = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        *,
        embedder: Any,
        store: Any,
        cognitive_router: Any = None,
        target_mode: str | None = None,
        reload_embedder: bool = True,
    ):
        super().__init__()
        self.target_mode = normalize_mode_id(target_mode) if target_mode else None
        self.reload_embedder = bool(reload_embedder)
        self.embedder = embedder
        self.store = store
        self.cognitive_router = cognitive_router

    def run(self) -> None:
        set_reindex_in_progress(True)
        try:
            self.status_update.emit("Exporting indexed content…")
            exported = self.store.export_all_rows()
            total = max(1, len(exported))

            if self.reload_embedder and self.target_mode is not None:
                spec = get_mode_spec(self.target_mode)
                self.status_update.emit(f"Loading {spec.label} embedding model…")
                self.embedder.reload(mode_id=self.target_mode)
            else:
                self.status_update.emit("Reprocessing with current embedding model…")

            self.status_update.emit("Rebuilding vector index…")
            self.store.recreate_for_dim(self.embedder.vector_dim)

            batch_records: list[dict] = []
            processed = 0
            last_pct = -1
            for row in exported:
                text = (row.get("text") or "").strip()
                source = row.get("source") or ""
                chunk_id = int(row.get("chunk_id") or 0)
                if not text or not source:
                    continue
                batch_records.append({"text": text, "source": source, "chunk_id": chunk_id})
                if len(batch_records) >= _BATCH_SIZE:
                    self._flush_batch(batch_records)
                    batch_records = []
                processed += 1
                pct = int(processed / total * 100)
                if pct != last_pct or processed == total:
                    last_pct = pct
                    self.progress_update.emit(pct)
                    self.status_update.emit(
                        f"Re-embedding chunks {processed:,} / {total:,} ({pct}%)"
                    )

            if batch_records:
                self._flush_batch(batch_records)

            self.store.rebuild_fts_index()

            if self.cognitive_router is not None:
                clear_router_embedding_state(self.cognitive_router)
                install_router_centroids(
                    self.cognitive_router,
                    self.embedder,
                    force=True,
                )

            self.progress_update.emit(100)
            self.status_update.emit("Reprocessing complete.")
            self.reindex_complete.emit(self.target_mode or "")
        except Exception as exc:
            logger.exception("Reindex failed")
            self.error_occurred.emit(str(exc))
        finally:
            set_reindex_in_progress(False)

    def _flush_batch(self, rows: list[dict]) -> None:
        texts = [r["text"] for r in rows]
        vectors = self.embedder.embed(texts)
        records = []
        for row, vector in zip(rows, vectors):
            records.append(
                {
                    "vector": vector.tolist(),
                    "text": row["text"],
                    "source": row["source"],
                    "chunk_id": row["chunk_id"],
                }
            )
        self.store.add_chunks(records, rebuild_fts=False)
