"""GGUF embedding backend via llama.cpp (advanced override only)."""
from __future__ import annotations

import gc
import logging
import multiprocessing
import os
from typing import Any

import numpy as np

from core.inference_transparency import log_inference_transparency, snapshot_from_loaded_llama
from rag.embed_utils import MAX_EMBED_CHARS, init_llama_embed, truncate_for_embed

logger = logging.getLogger("Qube.RAG.GgufBackend")


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 0:
        return vec
    return vec / norm


class GgufEmbeddingBackend:
    def __init__(self, model_path: str):
        if not model_path or not os.path.isfile(model_path):
            raise FileNotFoundError(f"Embedding GGUF not found: {model_path!r}")

        self._model_path = model_path
        self.backend_id = "gguf"
        self.display_name = os.path.basename(model_path)
        self._physical_cores = max(1, multiprocessing.cpu_count() // 2)
        self._backend = "unknown"
        self._requested_n_gpu_layers = 0
        self._inference_transparency: dict = {}
        self.model: Any | None = None
        self.vector_dim = 0
        self._load()

    def _load(self) -> None:
        try:
            self._requested_n_gpu_layers = -1
            self.model = init_llama_embed(self._model_path, -1, self._physical_cores)
            probe = self.model.create_embedding("hardware_test")
            self._backend = "gpu"
        except Exception as exc:
            logger.warning("GGUF embedder GPU init failed, falling back to CPU: %s", exc)
            self._requested_n_gpu_layers = 0
            self.model = init_llama_embed(self._model_path, 0, self._physical_cores)
            probe = self.model.create_embedding("hardware_test")
            self._backend = "cpu"

        vec = np.asarray(probe["data"][0]["embedding"], dtype=np.float32)
        self.vector_dim = int(vec.shape[0])
        self._capture_transparency()

    def unload(self) -> None:
        self.model = None
        self._inference_transparency = {}
        gc.collect()

    def _capture_transparency(self) -> None:
        if self.model is None:
            self._inference_transparency = {"loaded": False, "role": "embedder"}
            return
        try:
            snap = snapshot_from_loaded_llama(
                self.model,
                model_path=self._model_path,
                requested_n_gpu_layers=self._requested_n_gpu_layers,
                n_ctx=2048,
                n_threads=self._physical_cores,
                role="embedder",
            )
            snap["backend"] = self._backend
            self._inference_transparency = snap
            log_inference_transparency(logger, role="Embedder", snapshot=snap)
        except Exception as exc:
            logger.debug("GGUF transparency capture failed: %s", exc)
            self._inference_transparency = {
                "loaded": True,
                "role": "embedder",
                "model_basename": os.path.basename(self._model_path),
                "backend": self._backend,
            }

    def get_inference_transparency(self) -> dict:
        snap = dict(self._inference_transparency)
        snap.setdefault("role", "embedder")
        snap["backend"] = self._backend
        snap["vector_dim"] = self.vector_dim
        return snap

    def _embed_one(self, text: str, *, prefix: str) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("GGUF backend is not loaded.")
        safe = truncate_for_embed(text)
        response = self.model.create_embedding(f"{prefix}{safe}")
        vec = np.asarray(response["data"][0]["embedding"], dtype=np.float32)
        return _normalize(vec)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_one(text, prefix="search_query: ")

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.vector_dim), dtype=np.float32)
        rows = [self._embed_one(t, prefix="search_document: ") for t in texts]
        return np.asarray(rows, dtype=np.float32)
