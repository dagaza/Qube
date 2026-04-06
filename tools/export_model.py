# export_model.py
from sentence_transformers import SentenceTransformer
from pathlib import Path
import subprocess

print("Downloading PyTorch model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
export_path = Path("models/minilm")
export_path.mkdir(parents=True, exist_ok=True)
model.save(str(export_path))

print("Converting to ONNX via Optimum...")
subprocess.run([
    "optimum-cli", "export", "onnx", 
    "--model", "models/minilm", 
    "--task", "sentence-similarity", 
    "models/minilm_onnx"
])

print("✅ ONNX Export Complete! You can now delete the 'models/minilm' PyTorch folder to save space.")