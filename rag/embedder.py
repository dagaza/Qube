# rag/embedder.py
import numpy as np
from llama_cpp import Llama
import os
import multiprocessing
import logging

logger = logging.getLogger("Qube.RAG.Embedder")

# Hard cap on characters passed to llama.cpp embedding (token count must stay ≤ n_ctx / n_ubatch).
# Dense code / CJK can inflate tokens; keep this conservative to avoid GGML_ASSERT on n_ubatch.
MAX_EMBED_CHARS = 4000

_LLAMA_CTX = 8192


def _llama_embed_kwargs() -> dict:
    """Shared context/batch sizing so n_ubatch >= max single-sequence tokens (llama.cpp requirement)."""
    return {
        "n_ctx": _LLAMA_CTX,
        "n_batch": _LLAMA_CTX,
        "n_ubatch": _LLAMA_CTX,
    }


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
    """Construct Llama for embeddings; retries without n_ubatch if the binding is too old."""
    base = dict(
        model_path=model_path,
        embedding=True,
        **_llama_embed_kwargs(),
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
            return Llama(**base)
        raise


class EmbeddingModel:
    def __init__(self):
        model_path = os.path.join("models", "nomic-embed-text-v1.5.Q4_K_M.gguf")
        physical_cores = max(1, multiprocessing.cpu_count() // 2)

        logger.info("Probing user hardware...")

        try:
            self.model = _init_llama_embed(model_path, -1, physical_cores)
            self.model.create_embedding("hardware_test")
            logger.info("GPU acceleration engaged successfully!")

        except Exception as e:
            logger.warning(f"GPU init failed (Likely missing drivers). Falling back to CPU. Error: {e}")
            self.model = _init_llama_embed(model_path, 0, physical_cores)
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
                embeddings.append([0.0] * 768)
                
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