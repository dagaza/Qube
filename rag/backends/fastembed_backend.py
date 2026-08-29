"""ONNX embedding backend via fastembed."""
from __future__ import annotations

import gc
import logging
from typing import Any

import numpy as np

from core.embedding_modes import EmbeddingModeSpec, get_mode_spec
from rag.embed_utils import MAX_EMBED_CHARS, truncate_for_embed

logger = logging.getLogger("Qube.RAG.FastembedBackend")


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 0:
        return vec
    return vec / norm


class FastembedBackend:
    def __init__(self, spec: EmbeddingModeSpec | None = None):
        self._spec = spec or get_mode_spec()
        self.backend_id = f"fastembed:{self._spec.mode_id}"
        self.vector_dim = int(self._spec.vector_dim)
        self.display_name = f"{self._spec.label} ({self._spec.fastembed_model})"
        self._model: Any = None
        self._load()

    def _load(self) -> None:
        from core.paths import configure_user_model_paths
        from fastembed import TextEmbedding

        configure_user_model_paths()

        supported = {
            entry["model"]
            for entry in TextEmbedding.list_supported_models()
        }
        model_name = self._spec.fastembed_model
        if model_name not in supported:
            raise ValueError(
                f"Embedding preset {self._spec.label!r} uses unsupported fastembed "
                f"model {model_name!r}. Choose a model from "
                "TextEmbedding.list_supported_models()."
            )

        load_kwargs: dict[str, str] = {}
        from core.bootstrap_search_download import resolve_qube_preset_path

        preset_path = resolve_qube_preset_path(self._spec.mode_id)
        if preset_path:
            load_kwargs["specific_model_path"] = preset_path

        logger.info(
            "Loading fastembed model mode=%s model=%s preset_path=%s",
            self._spec.mode_id,
            model_name,
            preset_path or "(fastembed cache)",
        )
        self._model = TextEmbedding(model_name=model_name, **load_kwargs)
        gc.collect()

    def unload(self) -> None:
        self._model = None
        gc.collect()

    def get_inference_transparency(self) -> dict:
        return {
            "loaded": self._model is not None,
            "role": "embedder",
            "backend": "fastembed",
            "mode_id": self._spec.mode_id,
            "model_name": self._spec.fastembed_model,
            "vector_dim": self.vector_dim,
        }

    def _format_query(self, text: str) -> str:
        return truncate_for_embed(text)

    def _format_document(self, text: str) -> str:
        return truncate_for_embed(text)

    def _embed_texts(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.vector_dim), dtype=np.float32)
        if self._model is None:
            raise RuntimeError("Fastembed backend is not loaded.")

        formatter = self._format_query if is_query else self._format_document
        formatted = [formatter(t) for t in texts]
        vectors: list[np.ndarray] = []
        for embedding in self._model.embed(formatted, batch_size=8):
            vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
            if vec.shape[0] != self.vector_dim:
                raise ValueError(
                    f"Expected dim {self.vector_dim}, got {vec.shape[0]}"
                )
            vectors.append(_normalize(vec))
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"fastembed returned {len(vectors)} vectors for {len(texts)} texts"
            )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_texts([text], is_query=True)[0]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed_texts(texts, is_query=False)
