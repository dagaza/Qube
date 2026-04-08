# rag/store.py
import lancedb
import pyarrow as pa
import numpy as np
from pathlib import Path
from datetime import timedelta
import logging

logger = logging.getLogger("Qube.RAG.Store")

DB_PATH = Path("data/lancedb")
TABLE_NAME = "documents"
VECTOR_DIM = 384  # all-MiniLM-L6-v2 output dimension

SCHEMA = pa.schema([
    pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),
    pa.field("text", pa.utf8()),
    pa.field("source", pa.utf8()),     
    pa.field("chunk_id", pa.int32()),
])

class DocumentStore:
    def __init__(self):
        DB_PATH.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(DB_PATH))
        if TABLE_NAME in self.db.table_names():
            self.table = self.db.open_table(TABLE_NAME)
        else:
            self.table = self.db.create_table(TABLE_NAME, schema=SCHEMA)

    def get_all_indexed_sources(self) -> list[str]:
        """Queries the LanceDB table for all unique document filenames."""
        try:
            # Use native .to_list() instead of .to_pandas() to avoid heavy dependencies
            results = self.table.search().select(["source"]).limit(100000).to_list()
            
            # Extract unique sources using a Python set comprehension
            unique_sources = {item['source'] for item in results if 'source' in item}
            return list(unique_sources)
            
        except Exception as e:
            import logging
            logger = logging.getLogger("Qube.RAG")
            logger.warning(f"Failed to query LanceDB for unique sources: {e}")
            
        return []

    def add_chunks(self, chunks: list[dict]):
        self.table.add(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        return self.table.search(query_vector).limit(top_k).select(["text", "source", "chunk_id"]).to_list()

    def source_exists(self, source: str) -> bool:
        try:
            result = self.table.search([0.0] * VECTOR_DIM).limit(1).where(f"source = '{source}'").to_list()
            return len(result) > 0
        except Exception:
            return False

    # --- NEW CAPABILITIES FOR LIBRARY UI ---

    def delete_document(self, source_name: str):
        """Permanently removes vectors and aggressively reclaims disk space for individual files."""
        import logging
        from datetime import timedelta
        logger = logging.getLogger("Qube.RAG")
        
        try:
            # 1. Logically delete the rows associated with the specific file
            self.table.delete(f'source = "{source_name}"')
            logger.info(f"Logical delete complete for '{source_name}'.")
            
            # 2. Aggressively delete ONLY the unreferenced *.lance fragments
            try:
                # Modern LanceDB optimize API
                if hasattr(self.table, 'optimize'):
                    self.table.optimize(cleanup_older_than=timedelta(seconds=0))
                    logger.info(f"HARD DELETE: Pruned specific *.lance fragment for '{source_name}'.")
                
                # Legacy fallback
                else:
                    if hasattr(self.table, 'compact_files'):
                        self.table.compact_files()
                    if hasattr(self.table, 'cleanup_old_versions'):
                        self.table.cleanup_old_versions(older_than=timedelta(seconds=0))
                    logger.info(f"HARD DELETE: Executed garbage collection for '{source_name}'.")
                    
            except Exception as cleanup_err:
                logger.warning(f"Rows deleted, but physical disk cleanup was bypassed: {cleanup_err}")

            # Notice: The "drop_table" nuclear option has been removed!

        except Exception as e:
            logger.error(f"Failed to delete vectors for '{source_name}' from LanceDB: {e}")

    def rename_document(self, old_source: str, new_source: str) -> bool:
        """Updates the source filename for all chunks belonging to a document."""
        import logging
        logger = logging.getLogger("Qube.RAG")
        
        try:
            # 1. Attempt the modern LanceDB direct update
            self.table.update(where=f"source = '{old_source}'", values={"source": new_source})
            logger.info(f"Successfully renamed vectors from '{old_source}' to '{new_source}'.")
            return True
            
        except Exception as e:
            logger.warning(f"Native update failed or not supported, attempting manual rewrite: {e}")
            
            # 2. Bulletproof Fallback: Read -> Modify -> Insert -> Delete
            try:
                # Fetch all old records (using a dummy vector and high limit)
                records = self.table.search([0.0] * VECTOR_DIM).limit(100000).where(f"source = '{old_source}'").to_list()
                
                if not records:
                    logger.warning(f"No vectors found for '{old_source}' during rename.")
                    return False
                
                # Update source in memory and strip out LanceDB search artifacts
                for r in records:
                    r["source"] = new_source
                    r.pop("_distance", None) # Remove search distance if it was attached
                    
                # Add the cloned, renamed records
                self.table.add(records)
                
                # Delete the old records
                self.table.delete(f"source = '{old_source}'")
                logger.info(f"Fallback rename complete for '{new_source}'.")
                return True
                
            except Exception as fallback_err:
                logger.error(f"CRITICAL: Complete failure to rename document in Vector Store: {fallback_err}")
                return False

    def reconstruct_document(self, source: str) -> str:
        """Grabs all chunks for a file and stitches them back together for reading."""
        import re # 🔑 Imported for text cleaning
        
        try:
            # We fetch up to 10,000 chunks for the specific file
            results = self.table.search([0.0] * VECTOR_DIM).limit(10000).where(f"source = '{source}'").to_list()
            if not results:
                return "Document contents not found in vector store."
            
            # Sort by chunk_id so the text reads in the correct original order
            results.sort(key=lambda x: x["chunk_id"])
            
            # 1. Stitch it back together
            reconstructed_text = "\n\n".join([r["text"] for r in results])
            
            # 2. 🔑 THE FIX: PDF Hard Line Break Cleaner
            # This regex finds single newlines that are NOT surrounded by other newlines 
            # and replaces them with a space, while leaving \n\n (paragraphs) intact.
            reconstructed_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', reconstructed_text)
            
            # 3. Clean up any double spaces created by the merge
            reconstructed_text = re.sub(r' +', ' ', reconstructed_text)
            
            return reconstructed_text
            
        except Exception as e:
            logger.error(f"Failed to reconstruct {source}: {e}")
            return f"Error loading document: {str(e)}"