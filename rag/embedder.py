# rag/embedder.py
import numpy as np
from llama_cpp import Llama
import os
import multiprocessing
import logging

logger = logging.getLogger("Qube.RAG.Embedder")

class EmbeddingModel:
    def __init__(self):
        model_path = os.path.join("models", "nomic-embed-text-v1.5.Q4_K_M.gguf")
        physical_cores = max(1, multiprocessing.cpu_count() // 2)

        logger.info("Probing user hardware...")

        try:
            # 🏎️ THE FAST PATH: Try Vulkan GPU first
            self.model = Llama(
                model_path=model_path,
                embedding=True,
                n_gpu_layers=-1, 
                n_ctx=8192,
                n_batch=8192,
                n_threads=physical_cores,
                verbose=False
            )
            self.model.create_embedding("hardware_test")
            logger.info("GPU acceleration engaged successfully!")

        except Exception as e:
            # 🐢 THE SAFE PATH: No GPU or bad drivers? Fall back to CPU instantly.
            logger.warning(f"GPU init failed (Likely missing drivers). Falling back to CPU. Error: {e}")
            
            self.model = Llama(
                model_path=model_path,
                embedding=True,
                n_gpu_layers=0, 
                n_ctx=8192,
                n_batch=8192,
                n_threads=physical_cores,
                verbose=False
            )
            logger.info("Running on CPU mode.")

    def embed(self, texts: list[str]) -> np.ndarray:
        """Rock-solid sequential embedding to bypass llama.cpp batching bugs."""
        embeddings = []
        
        for text in texts:
            safe_text = text[:25000] 
            formatted_text = f"search_document: {safe_text}"
            
            try:
                response = self.model.create_embedding(formatted_text)
                embeddings.append(response["data"][0]["embedding"])
            except Exception as e:
                logger.error(f"CRITICAL: Chunk failed. Inserting blank vector. Error: {e}")
                embeddings.append([0.0] * 768)
                
        return np.array(embeddings, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """Single string embedding for convenience."""
        return self.embed([text])[0]
        
    def embed_query(self, query: str) -> np.ndarray:
        """Use this specifically in your LLM search tool!"""
        formatted_query = f"search_query: {query}"
        response = self.model.create_embedding(formatted_query)
        return np.array(response["data"][0]["embedding"], dtype=np.float32)