# rag/embedder.py
from __future__ import annotations

import gc
import logging
from typing import Any

import numpy as np

from core.app_settings import get_embedding_mode
from core.embedding_modes import DEFAULT_MODE, ModeId, get_mode_spec, normalize_mode_id
from core.embedding_models import mark_embedding_preset_available, resolve_active_gguf_path
from rag.backends.fastembed_backend import FastembedBackend
from rag.embed_utils import MAX_EMBED_CHARS, truncate_for_embed

logger = logging.getLogger("Qube.RAG.Embedder")

# Back-compat alias used by ingestion_worker
_truncate_for_embed = truncate_for_embed


class EmbeddingModel:
    """Facade over one active embedding backend (mode preset or GGUF override)."""

    def __init__(self, mode_id: str | None = None, model_path: str | None = None):
        self._mode_id: ModeId | None = None
        self._model_path = ""
        self._backend: Any = None
        self._load(mode_id=mode_id, model_path=model_path)

    @property
    def active_model_path(self) -> str:
        if self._model_path:
            return self._model_path
        if self._mode_id:
            return get_mode_spec(self._mode_id).fastembed_model
        return ""

    @property
    def active_mode_id(self) -> str | None:
        return self._mode_id

    @property
    def expected_vector_dim(self) -> int:
        return int(self._backend.vector_dim)

    @property
    def vector_dim(self) -> int:
        return self.expected_vector_dim

    def get_inference_transparency(self) -> dict:
        if self._backend is None:
            return {"loaded": False, "role": "embedder"}
        return self._backend.get_inference_transparency()

    def reload(
        self,
        mode_id: str | None = None,
        model_path: str | None = None,
    ) -> None:
        if self._backend is not None:
            try:
                self._backend.unload()
            except Exception:
                logger.debug("Backend unload failed during reload", exc_info=True)
        self._backend = None
        gc.collect()
        self._load(mode_id=mode_id, model_path=model_path)

    def _load(self, *, mode_id: str | None, model_path: str | None) -> None:
        gguf_path = (model_path or resolve_active_gguf_path() or "").strip()
        if gguf_path:
            from rag.backends.gguf_backend import GgufEmbeddingBackend

            self._model_path = gguf_path
            self._mode_id = None
            self._backend = GgufEmbeddingBackend(gguf_path)
            return

        from core.app_settings import get_embedding_mode

        resolved_mode = normalize_mode_id(mode_id or get_embedding_mode() or DEFAULT_MODE)
        self._mode_id = resolved_mode
        self._model_path = ""
        self._backend = FastembedBackend(get_mode_spec(resolved_mode))
        mark_embedding_preset_available(resolved_mode)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            dim = self.expected_vector_dim
            return np.zeros((0, dim), dtype=np.float32)
        return self._backend.embed_documents(texts)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def embed_query(self, query: str) -> np.ndarray:
        return self._backend.embed_query(query)
