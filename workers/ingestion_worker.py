from PyQt6.QtCore import QThread, pyqtSignal
from pathlib import Path
from rag.parsers import parse_file
from rag.chunker import chunk_text
from rag.embedder import EmbeddingModel
from rag.store import DocumentStore
import logging

# FIX 1: Give this its own logger identity so terminal output is clear
logger = logging.getLogger("Qube.RAG")

class IngestionWorker(QThread):
    progress_update = pyqtSignal(int)       
    file_done = pyqtSignal(str)             
    ingestion_complete = pyqtSignal(int)    
    error_occurred = pyqtSignal(str)        

    def __init__(self, file_paths: list[Path], embedder: EmbeddingModel, store: DocumentStore):
        super().__init__()
        self.file_paths = file_paths
        self.embedder = embedder
        self.store = store

    def run(self):
        total_chunks = 0
        total_files = len(self.file_paths)
        
        logger.info(f"Starting ingestion sequence for {total_files} files.")
        
        for i, path in enumerate(self.file_paths):
            try:
                source = path.name
                logger.info(f"Processing: {source}")
                
                # Update UI immediately so the user knows it started
                self.file_done.emit(f"Reading {source}...")
                
                if self.store.source_exists(source):
                    self.file_done.emit(f"Skipped (already indexed): {source}")
                    self.progress_update.emit(int((i + 1) / total_files * 100))
                    logger.warning(f"File {source} is already in the database. Skipping.")
                    continue

                # 1. Parsing
                raw_sections = parse_file(path)
                chunks = []
                for section in raw_sections:
                    chunks.extend(chunk_text(section))

                # FIX 2: Stop early if the document yielded no text
                if not chunks:
                    error_msg = f"No readable text found in {source}."
                    self.error_occurred.emit(error_msg)
                    logger.error(error_msg)
                    self.progress_update.emit(int((i + 1) / total_files * 100))
                    continue

                logger.debug(f"Successfully extracted {len(chunks)} chunks from {source}. Starting embedding.")
                self.file_done.emit(f"Embedding {len(chunks)} chunks from {source}...")

                # 2. Embedding
                batch_size = 32
                records = []
                
                for b_start in range(0, len(chunks), batch_size):
                    batch = chunks[b_start:b_start + batch_size]
                    vectors = self.embedder.embed(batch)
                    
                    for j, (text, vector) in enumerate(zip(batch, vectors)):
                        records.append({
                            "vector": vector.tolist(),
                            "text": text,
                            "source": source,
                            "chunk_id": b_start + j,
                        })
                        
                    # FIX 3: Micro-updates for the UI progress bar during heavy embedding
                    # Calculates partial progress of the current file being processed
                    file_base_progress = (i / total_files) * 100
                    chunk_progress = ((b_start + len(batch)) / len(chunks)) * (100 / total_files)
                    self.progress_update.emit(int(file_base_progress + chunk_progress))

                # 3. Storage
                self.store.add_chunks(records)
                total_chunks += len(records)
                
                logger.info(f"Successfully committed {len(records)} vectors to database for {source}.")
                self.file_done.emit(f"Indexed: {source}")

            except Exception as e:
                self.error_occurred.emit(f"Failed on {path.name}: {str(e)}")
                logger.error(f"Ingestion crashed on {path.name}", exc_info=True)

            self.progress_update.emit(int((i + 1) / total_files * 100))

        logger.info(f"Ingestion sequence complete. Total new chunks added: {total_chunks}")
        self.ingestion_complete.emit(total_chunks)