"""DocumentStore dynamic dimension tests."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

from rag.store import DocumentStore


def test_recreate_for_dim_changes_table_dimension():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "lance"
        store = DocumentStore(db_path=db_path, expected_vector_dim=512, quiet=True)
        store.add_chunks(
            [{
                "vector": [0.1] * 512,
                "text": "hello",
                "source": "doc.txt",
                "chunk_id": 0,
            }],
            rebuild_fts=False,
        )
        assert store.vector_dim == 512

        store.recreate_for_dim(384)
        assert store.vector_dim == 384
        assert store.get_all_indexed_sources() == []
