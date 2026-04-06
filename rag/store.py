# rag/store.py
import lancedb
import pyarrow as pa
import numpy as np
from pathlib import Path

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
        """chunks: [{"vector": np.array, "text": str, "source": str, "chunk_id": int}]"""
        self.table.add(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        results = (
            self.table.search(query_vector)
            .limit(top_k)
            .select(["text", "source", "chunk_id"])
            .to_list()
        )
        return results

    def source_exists(self, source: str) -> bool:
        """Avoid re-ingesting the same file"""
        try:
            result = self.table.search([0.0] * VECTOR_DIM).limit(1).where(
                f"source = '{source}'"
            ).to_list()
            return len(result) > 0
        except Exception:
            return False