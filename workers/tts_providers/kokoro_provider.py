import os
import requests
import numpy as np
import asyncio
import threading
import queue
import logging
from .base_provider import BaseTTSProvider

logger = logging.getLogger("Qube.TTS.Kokoro")

def ensure_model_exists(model_path: str):
    """Downloads the Kokoro model and voices if they don't exist."""
    base_dir = os.path.dirname(model_path)
    os.makedirs(base_dir, exist_ok=True)
    
    onnx_path = os.path.join(base_dir, "kokoro-v1.0.onnx")
    bin_path = os.path.join(base_dir, "voices-v1.0.bin")
    
    files_to_check = {
        onnx_path: "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v1.0.onnx",
        bin_path: "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices-v1.0.bin"
    }
    
    for file_path, url in files_to_check.items():
        if not os.path.exists(file_path):
            print(f"[SYSTEM] Downloading missing required file: {os.path.basename(file_path)}...")
            response = requests.get(url, stream=True)
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"[SYSTEM] Download complete: {os.path.basename(file_path)}")

class KokoroProvider(BaseTTSProvider):
    def __init__(self):
        super().__init__()
        self.engine = None
        self.supports_voice_selection = True
        self.sample_rate = 24000
        
    def load_model(self, model_path: str) -> bool:
        try:
            ensure_model_exists(model_path)
            from kokoro_onnx import Kokoro
            
            base_dir = os.path.dirname(model_path)
            voices_path = os.path.join(base_dir, "voices-v1.0.bin")
            self.engine = Kokoro(model_path, voices_path)
            
            voices_data = np.load(voices_path, allow_pickle=False)
            self.available_voices = list(voices_data.files)
            if self.available_voices:
                self.current_voice = self.available_voices[0]
                
            self.is_loaded = True
            return True
        except Exception as e:
            logger.error(f"Failed to load Kokoro: {e}")
            return False

    def set_voice(self, voice_name: str) -> bool:
        if voice_name in self.available_voices:
            self.current_voice = voice_name
            return True
        return False

    def generate_audio_stream(self, text: str, cancel_flag_callback):
        """Generates audio while constantly checking the kill-switch."""
        audio_queue = queue.Queue()

        async def fetch_stream():
            try:
                stream = self.engine.create_stream(text, voice=self.current_voice, speed=1.0, lang="en-us")
                async for samples, _ in stream:
                    # Check kill-switch before processing the chunk
                    if cancel_flag_callback():
                        break
                    pcm_data = (samples * 32767).astype(np.int16).tobytes()
                    audio_queue.put(pcm_data)
            except Exception as e:
                audio_queue.put(e)
            finally:
                audio_queue.put(None)

        threading.Thread(target=lambda: asyncio.run(fetch_stream()), daemon=True).start()

        while True:
            # Check kill-switch while waiting for audio
            if cancel_flag_callback():
                break
                
            try:
                chunk = audio_queue.get(timeout=0.1) # 100ms timeout to keep checking the flag
                if chunk is None: break
                if isinstance(chunk, Exception): raise chunk
                yield chunk
            except queue.Empty:
                continue