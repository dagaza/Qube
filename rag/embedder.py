# rag/embedder.py
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from pathlib import Path

# Adjust path if your working directory differs
MODEL_PATH = Path("models/minilm_onnx")

class EmbeddingModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
        self.session = ort.InferenceSession(
            str(MODEL_PATH / "model.onnx"),
            providers=["CPUExecutionProvider"]
        )

    def embed(self, texts: list[str]):
        # 1. Tokenize the text as normal
        encoded = dict(self.tokenizer(texts, padding=True, truncation=True, return_tensors="np"))
        
        # 2. FIX: Ask ONNX what inputs it actually wants
        expected_inputs = [i.name for i in self.session.get_inputs()]
        
        # 3. FIX: Filter the dictionary so we drop 'token_type_ids' (or anything else ONNX hates)
        onnx_inputs = {k: v for k, v in encoded.items() if k in expected_inputs}
        
        # 4. Run the session with the clean inputs
        outputs = self.session.run(None, onnx_inputs)
        # Mean pooling over token embeddings
        embeddings = outputs[0].mean(axis=1)
        # L2 normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return (embeddings / norms).astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]