# workers/ingestion_worker.py
from PyQt6.QtCore import QThread, pyqtSignal
from pathlib import Path
from rag.parsers import parse_file
from rag.embedder import EmbeddingModel
from rag.store import DocumentStore
import logging

logger = logging.getLogger("Qube.RAG")

class IngestionWorker(QThread):
    progress_update = pyqtSignal(int)       
    file_done = pyqtSignal(str)             
    ingestion_complete = pyqtSignal(int)    
    error_occurred = pyqtSignal(str)        

    def __init__(self, file_paths: list[Path], embedder: EmbeddingModel, store: DocumentStore, db_manager):
        super().__init__()
        self.file_paths = file_paths
        self.embedder = embedder
        self.store = store
        self.db = db_manager

    # 🔑 THE FIX: The new Semantic Chunker to protect llama.cpp
    def semantic_chunk_text(self, text: str, max_chars: int = 1500, overlap: int = 200) -> list[str]:
        """
        Slices massive documents into bite-sized, overlapping paragraphs 
        that the C++ engine can process at lightning speed safely.
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + max_chars
            
            if end >= text_length:
                chunks.append(text[start:].strip())
                break
            
            # Prefer paragraph breaks, then line breaks, then sentence breaks
            last_double_newline = text.rfind('\n\n', start, end)
            last_newline = text.rfind('\n', start, end)
            last_period = text.rfind('. ', start, end)
            
            if last_double_newline > start + (max_chars // 2):
                end = last_double_newline + 2
            elif last_newline > start + (max_chars // 2):
                end = last_newline + 1
            elif last_period > start + (max_chars // 2):
                end = last_period + 2
            else:
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space + 1
                
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)
                
            start = end - overlap
            
        return chunks

    def run(self):
        total_chunks = 0
        total_files = len(self.file_paths)
        
        logger.info(f"Starting ingestion sequence for {total_files} files.")
        
        for i, path in enumerate(self.file_paths):
            try:
                source = path.name
                
                # Retrieve file size in KB
                file_size_kb = round(path.stat().st_size / 1024, 2)
                
                logger.info(f"Processing: {source} ({file_size_kb} KB)")
                self.file_done.emit(f"Reading {source}...")
                
                if self.store.source_exists(source):
                    self.file_done.emit(f"Skipped (already indexed): {source}")
                    self.progress_update.emit(int((i + 1) / total_files * 100))
                    continue

                raw_sections = parse_file(path)
                chunks = []
                
                # 🔑 THE FIX: Route the raw text through our safe chunker
                for section in raw_sections:
                    chunks.extend(self.semantic_chunk_text(section))

                if not chunks:
                    self.error_occurred.emit(f"No readable text found in {source}.")
                    self.progress_update.emit(int((i + 1) / total_files * 100))
                    continue

                self.file_done.emit(f"Embedding {len(chunks)} chunks from {source}...")

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
                        
                    file_base_progress = (i / total_files) * 100
                    chunk_progress = ((b_start + len(batch)) / len(chunks)) * (100 / total_files)
                    self.progress_update.emit(int(file_base_progress + chunk_progress))

                # Write vectors to LanceDB
                self.store.add_chunks(records)
                
                # Write metadata to SQLite for the UI Library Tab
                self.db.add_document_metadata(source, file_size_kb, len(chunks))
                
                total_chunks += len(records)
                logger.info(f"Indexed {source}: {len(records)} chunks saved to LanceDB, metadata logged to SQLite.")
                self.file_done.emit(f"Indexed: {source}")

            except Exception as e:
                self.error_occurred.emit(f"Failed on {path.name}: {str(e)}")
                logger.error(f"Ingestion crashed on {path.name}", exc_info=True)

            self.progress_update.emit(int((i + 1) / total_files * 100))

        logger.info(f"Ingestion complete. Added {total_chunks} new chunks.")
        self.ingestion_complete.emit(total_chunks)