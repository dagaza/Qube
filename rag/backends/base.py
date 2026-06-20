"""Embedding backend protocol."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EmbeddingBackend(Protocol):
    backend_id: str
    vector_dim: int
    display_name: str

    def embed_query(self, text: str) -> np.ndarray: ...

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def unload(self) -> None: ...

    def get_inference_transparency(self) -> dict: ...
