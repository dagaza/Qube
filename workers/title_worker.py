# --- workers/title_worker.py ---
from PyQt6.QtCore import QThread, pyqtSignal
from llama_cpp import Llama
import os

class TitleWorker(QThread):
    # Signal to tell the UI when a title is ready (session_id, new_title)
    title_generated = pyqtSignal(str, str)

    def __init__(self, db_manager):
        super().__init__()
        self.db = db_manager
        
        # Point to the Qwen micro-model
        model_path = os.path.join("models", "qwen2-0_5b-instruct-q4_k_m.gguf")
        
        if os.path.exists(model_path):
            print("Loading Qwen2-0.5B Titling AI on CPU...")
            self.model = Llama(
                model_path=model_path,
                n_gpu_layers=0,  # Forces CPU-only to protect your GPU VRAM
                n_ctx=512,       # Tiny context window for instant memory allocation
                verbose=False    # Hides the C++ terminal spam
            )
        else:
            self.model = None
            print(f"Warning: Titling AI not found at {model_path}. Auto-titling disabled.")

    def run_titling(self, user_prompt, session_id):
        """Generates a smart title using Qwen's world knowledge."""
        if not self.model:
            return

        # The "Ruthless" Prompt Engineering using ChatML
        prompt = (
            "<|im_start|>system\n"
            "You are an automated titling engine. Extract the thematic core of the user's message into a 2-4 word title. You are strictly forbidden from using conversational filler, punctuation, or quotation marks. Output ONLY the title text.<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        try:
            # Low temperature = strictly factual extraction
            output = self.model(
                prompt,
                max_tokens=10,
                temperature=0.2, 
                stop=["<|im_end|>", "\n"] 
            )
            
            # Clean up the output
            new_title = output['choices'][0]['text'].strip()
            new_title = new_title.replace('"', '').replace("'", "")
            new_title = new_title.title() # "color of the sky" -> "Color Of The Sky"

            # Update DB and notify UI
            if new_title and self.db.rename_session(session_id, new_title):
                self.title_generated.emit(session_id, new_title)
                
        except Exception as e:
            print(f"Titling AI Error: {e}")