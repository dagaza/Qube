# rag/store.py
import lancedb
import pyarrow as pa
import numpy as np
from pathlib import Path
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

    def delete_document(self, source: str):
        """Erases all vector chunks associated with a specific file."""
        try:
            self.table.delete(f"source = '{source}'")
            logger.info(f"Successfully purged all vectors for: {source}")
        except Exception as e:
            logger.error(f"Failed to delete {source} from LanceDB: {e}")

    def reconstruct_document(self, source: str) -> str:
        """Grabs all chunks for a file and stitches them back together for reading."""
        try:
            # We fetch up to 10,000 chunks for the specific file
            results = self.table.search([0.0] * VECTOR_DIM).limit(10000).where(f"source = '{source}'").to_list()
            if not results:
                return "Document contents not found in vector store."
            
            # Sort by chunk_id so the text reads in the correct original order
            results.sort(key=lambda x: x["chunk_id"])
            
            # Stitch it back together
            reconstructed_text = "\n\n".join([r["text"] for r in results])
            return reconstructed_text
        except Exception as e:
            logger.error(f"Failed to reconstruct {source}: {e}")
            return f"Error loading document: {str(e)}"