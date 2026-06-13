# rag/embedder.py
import gc
import numpy as np
from llama_cpp import Llama
import os
import multiprocessing
import logging

from core.embedding_models import (
    EXPECTED_VECTOR_DIM,
    migrate_legacy_embedding_layout,
    resolve_active_embedding_path,
)

logger = logging.getLogger("Qube.RAG.Embedder")

# Hard cap on characters passed to llama.cpp embedding (token count must stay ≤ n_ctx / n_ubatch).
# Dense code / CJK can inflate tokens; keep this conservative to avoid GGML_ASSERT on n_ubatch.
MAX_EMBED_CHARS = 2400

# nomic-embed-text-v1.5 GGUF reports n_ctx_train≈2048; larger n_ctx breaks llama_context on many builds.
_LLAMA_CTX = 2048
_LLAMA_CTX_FALLBACKS = (2048, 1024, 512)


def _llama_embed_kwargs(*, n_ctx: int | None = None) -> dict:
    """Shared context/batch sizing so n_ubatch >= max single-sequence tokens (llama.cpp requirement)."""
    n = _LLAMA_CTX if n_ctx is None else int(n_ctx)
    return {"n_ctx": n, "n_batch": n, "n_ubatch": n}


def _truncate_for_embed(text: str) -> str:
    if len(text) <= MAX_EMBED_CHARS:
        return text
    logger.warning(
        "Embedding input truncated from %d to %d chars (MAX_EMBED_CHARS)",
        len(text),
        MAX_EMBED_CHARS,
    )
    return text[:MAX_EMBED_CHARS]


def _init_llama_embed(model_path: str, n_gpu_layers: int, physical_cores: int) -> Llama:
    """Construct Llama for embeddings; retries smaller ``n_ctx`` and without ``n_ubatch`` if needed."""
    last_error: Exception | None = None
    for n_ctx in _LLAMA_CTX_FALLBACKS:
        base = dict(
            model_path=model_path,
            embedding=True,
            **_llama_embed_kwargs(n_ctx=n_ctx),
            n_threads=physical_cores,
            verbose=False,
            n_gpu_layers=n_gpu_layers,
        )
        try:
            return Llama(**base)
        except TypeError as e:
            err = str(e).lower()
            if "n_ubatch" in err or "unexpected keyword" in err:
                base.pop("n_ubatch", None)
                logger.warning(
                    "Llama() has no n_ubatch; retrying (upgrade llama-cpp-python to match llama.cpp batch fixes)"
                )
                try:
                    return Llama(**base)
                except Exception as retry_exc:
                    last_error = retry_exc
                    logger.warning("Embedder init failed at n_ctx=%s: %s", n_ctx, retry_exc)
                    continue
            raise
        except Exception as e:
            last_error = e
            logger.warning("Embedder init failed at n_ctx=%s: %s", n_ctx, e)
            continue

    if last_error is not None:
        raise last_error
    raise RuntimeError("Embedder init failed with no error detail")


class EmbeddingModel:
    def __init__(self, model_path: str | None = None):
        migrate_legacy_embedding_layout()
        self._model_path = ""
        self.model: Llama | None = None
        self._physical_cores = max(1, multiprocessing.cpu_count() // 2)
        self._load(model_path or resolve_active_embedding_path())

    @property
    def active_model_path(self) -> str:
        return self._model_path

    @property
    def expected_vector_dim(self) -> int:
        return EXPECTED_VECTOR_DIM

    def reload(self, model_path: str | None = None) -> None:
        """Unload and reload the embedder from settings or an explicit path."""
        self.model = None
        gc.collect()
        self._load(model_path or resolve_active_embedding_path())

    def _load(self, model_path: str) -> None:
        if not model_path or not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Embedding model not found at {model_path!r}. "
                f"Place {os.path.basename(model_path or '')} under ~/.qube/models/embedding/."
            )

        self._model_path = model_path
        logger.info("Probing user hardware for embedding model: %s", os.path.basename(model_path))

        try:
            self.model = _init_llama_embed(model_path, -1, self._physical_cores)
            self.model.create_embedding("hardware_test")
            logger.info("GPU acceleration engaged successfully!")

        except Exception as e:
            logger.warning(
                "GPU init failed (Likely missing drivers). Falling back to CPU. Error: %s",
                e,
            )
            self.model = _init_llama_embed(model_path, 0, self._physical_cores)
            logger.info("Running on CPU mode.")

    def embed(self, texts: list[str]) -> np.ndarray:
        """Rock-solid sequential embedding to bypass llama.cpp batching bugs."""
        embeddings = []

        for text in texts:
            safe_text = _truncate_for_embed(text)
            formatted_text = f"search_document: {safe_text}"
            
            try:
                # 2. Process one by one safely
                response = self.model.create_embedding(formatted_text)
                
                # 🔑 THE FIX: Extract, convert, and normalize the vector
                vec = np.array(response["data"][0]["embedding"], dtype=np.float32)
                embeddings.append(self._normalize(vec))
                
            except Exception as e:
                # Keep your existing logger/print statement here
                print(f"CRITICAL: Chunk failed. Inserting blank vector. Error: {e}")
                embeddings.append([0.0] * EXPECTED_VECTOR_DIM)
                
        return np.array(embeddings, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """Single string embedding for convenience."""
        return self.embed([text])[0]
        
    def embed_query(self, query: str) -> np.ndarray:
        """Use this specifically in your LLM search tool!"""
        q = _truncate_for_embed(query)
        formatted_query = f"search_query: {q}"
        response = self.model.create_embedding(formatted_query)
        vec = np.array(response["data"][0]["embedding"], dtype=np.float32)
        return self._normalize(vec) # 🔑 THE FIX
    
    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
