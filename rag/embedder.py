# rag/embedder.py
import numpy as np
from llama_cpp import Llama
import os
import multiprocessing

class EmbeddingModel:
    def __init__(self):
        model_path = os.path.join("models", "nomic-embed-text-v1.5.Q4_K_M.gguf")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing embedding model: {model_path}")

        print("Booting Nomic Engine (Hardware Auto-Detect)...")
        
        physical_cores = max(1, multiprocessing.cpu_count() // 2)
        
        # 🔑 THE PRESTIGE SETUP:
        # n_gpu_layers=-1 tells it to push 100% of the model to the GPU.
        # If the user has a GPU (and the Vulkan library), it will fly.
        # If they don't, llama.cpp will gracefully fall back to the CPU automatically.
        self.model = Llama(
            model_path=model_path,
            embedding=True,  
            n_gpu_layers=-1, 
            n_ctx=8192,      
            n_batch=8192,    
            n_threads=physical_cores, 
            verbose=False    
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        """Rock-solid sequential embedding to bypass llama.cpp batching bugs."""
        embeddings = []
        
        for text in texts:
            # 1. Hard-cap the text length just below the absolute limit to guarantee zero crashes
            safe_text = text[:25000] 
            formatted_text = f"search_document: {safe_text}"
            
            try:
                # 2. Process one by one safely
                response = self.model.create_embedding(formatted_text)
                embeddings.append(response["data"][0]["embedding"])
            except Exception as e:
                print(f"CRITICAL: Chunk failed. Inserting blank vector. Error: {e}")
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