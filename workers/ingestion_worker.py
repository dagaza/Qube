# workers/ingestion_worker.py
from PyQt6.QtCore import QThread, pyqtSignal
from pathlib import Path
from rag.embedder import EmbeddingModel, MAX_EMBED_CHARS
from rag.store import DocumentStore
import logging

from core.app_settings import get_sidecar_ingest_blurb_enabled
from core.chunking.chunk_metadata import chunk_record_to_meta_json
from core.chunking.embed_context import library_chunk_embed_text
from core.chunking.ingest_pipeline import chunk_document_for_ingest
from core.chunking.semantic_ingest import chunk_document_for_precision_ingest
from core.knowledge.document.builders.library_builder import build_document_from_path
from core.library_ingest_modes import is_precision_ingest_mode
from core.library_pro_features import resolve_import_ingest_mode

logger = logging.getLogger("Qube.RAG")


class IngestionWorker(QThread):
    progress_update = pyqtSignal(int)       
    file_done = pyqtSignal(str)             
    ingestion_complete = pyqtSignal(int)    
    error_occurred = pyqtSignal(str)        

    def __init__(
        self,
        file_paths: list[Path],
        embedder: EmbeddingModel,
        store: DocumentStore,
        db_manager,
        folder_id: str | None = None,
        sidecar_worker=None,
        ingest_mode: str | None = None,
    ):
        super().__init__()
        self.file_paths = file_paths
        self.embedder = embedder
        self.store = store
        self.db = db_manager
        self.ingest_mode = resolve_import_ingest_mode(ingest_mode)
        resolved = folder_id or db_manager.get_main_library_folder_id()
        if not db_manager.library_folder_allows_user_ingest(resolved):
            logger.warning(
                "Ingestion target folder does not allow user uploads; using Main."
            )
            resolved = db_manager.get_main_library_folder_id()
        self.folder_id = resolved
        self.sidecar_worker = sidecar_worker

    def run(self):
        total_chunks = 0
        total_files = len(self.file_paths)

        if self.embedder is None:
            msg = (
                "Search models are not ready. Open Settings → Knowledge → Search quality "
                "and tap Prepare search models, then try again."
            )
            logger.warning("Ingestion aborted: embedder not loaded.")
            self.error_occurred.emit(msg)
            self.ingestion_complete.emit(0)
            return

        logger.info(f"Starting ingestion sequence for {total_files} files.")
        
        for i, path in enumerate(self.file_paths):
            try:
                source = path.name
                
                file_size_kb = round(path.stat().st_size / 1024, 2)
                
                logger.info(f"Processing: {source} ({file_size_kb} KB)")
                self.file_done.emit(f"Reading {source}...")
                
                if self.store.source_exists(source):
                    self.file_done.emit(f"Skipped (already indexed): {source}")
                    self.progress_update.emit(int((i + 1) / total_files * 100))
                    continue

                document = build_document_from_path(path)
                if is_precision_ingest_mode(self.ingest_mode):
                    self.file_done.emit(f"Precision ingest (semantic split): {source}...")
                    chunk_records = chunk_document_for_precision_ingest(
                        document,
                        self.embedder,
                    )
                else:
                    chunk_records = chunk_document_for_ingest(document)

                if not chunk_records:
                    self.error_occurred.emit(f"No readable text found in {source}.")
                    self.progress_update.emit(int((i + 1) / total_files * 100))
                    continue

                chunks = [record.body[:MAX_EMBED_CHARS] for record in chunk_records]

                self.file_done.emit(f"Embedding {len(chunks)} chunks from {source}...")

                batch_size = 32
                records = []
                
                embed_inputs = [
                    library_chunk_embed_text(
                        source,
                        record.body,
                        section_heading=record.heading,
                        breadcrumb=record.breadcrumb,
                    )[:MAX_EMBED_CHARS]
                    for record in chunk_records
                ]

                for b_start in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[b_start:b_start + batch_size]
                    batch_embed = embed_inputs[b_start:b_start + batch_size]
                    vectors = self.embedder.embed(batch_embed)

                    for j, (record, vector) in enumerate(zip(chunk_records[b_start:b_start + batch_size], vectors)):
                        text = record.body[:MAX_EMBED_CHARS]
                        records.append({
                            "vector": vector.tolist(),
                            "text": text,
                            "source": source,
                            "chunk_id": b_start + j,
                            "meta_json": chunk_record_to_meta_json(record),
                        })
                        
                    file_base_progress = (i / total_files) * 100
                    chunk_progress = ((b_start + len(batch_chunks)) / len(chunks)) * (100 / total_files)
                    self.progress_update.emit(int(file_base_progress + chunk_progress))

                self.store.add_chunks(records)
                
                self.db.add_document_metadata(
                    source,
                    file_size_kb,
                    len(chunks),
                    folder_id=self.folder_id,
                    ingest_mode=self.ingest_mode,
                )

                if (
                    self.sidecar_worker
                    and get_sidecar_ingest_blurb_enabled()
                    and chunks
                ):
                    sample = chunks[0][:2500]
                    self.sidecar_worker.enqueue_ingest_blurb(source, sample)
                
                total_chunks += len(records)
                logger.info(f"Indexed {source}: {len(records)} chunks saved to LanceDB, metadata logged to SQLite.")
                self.file_done.emit(f"Indexed: {source}")

            except Exception as e:
                self.error_occurred.emit(f"Failed on {path.name}: {str(e)}")
                logger.error(f"Ingestion crashed on {path.name}", exc_info=True)

            self.progress_update.emit(int((i + 1) / total_files * 100))

        logger.info(f"Ingestion complete. Added {total_chunks} new chunks.")
        self.ingestion_complete.emit(total_chunks)
