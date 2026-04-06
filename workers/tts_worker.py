from PyQt6.QtCore import QThread, pyqtSignal
import queue
import pyaudio
import numpy as np
import os
import requests
import time
import logging
logger = logging.getLogger("Qube.Audio")

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

class PiperAdapter:
    def __init__(self, model_path):
        from piper.voice import PiperVoice
        self.voice = PiperVoice.load(model_path, use_cuda=False)
        self.sample_rate = self.voice.config.sample_rate
        self.available_voices = ["Default"]

    def synthesize(self, text, voice_name):
        for chunk in self.voice.synthesize(text):
            pcm_data = getattr(chunk, 'audio_int16_bytes', getattr(chunk, 'pcm', None))
            if pcm_data:
                yield pcm_data

class KokoroAdapter:
    def __init__(self, model_path):
        from kokoro_onnx import Kokoro
        import os
        import numpy as np

        base_dir = os.path.dirname(model_path)
        voices_path = os.path.join(base_dir, "voices-v1.0.bin")
        
        # --- 3. Call the downloader right here! ---
        # This guarantees the files exist before Kokoro tries to read them.
        ensure_model_exists(model_path)
        
        if not os.path.exists(voices_path):
            raise FileNotFoundError(f"Kokoro voices file not found at {voices_path}")
        
        self.engine = Kokoro(model_path, voices_path)
        self.sample_rate = 24000
        
        voices_data = np.load(voices_path, allow_pickle=False)
        self.available_voices = voices_data.files

    def synthesize(self, text, voice_name):
        import asyncio
        import threading
        import queue
        
        audio_queue = queue.Queue()

        async def fetch_stream():
            try:
                # Kokoro-ONNX supports streaming chunks
                stream = self.engine.create_stream(text, voice=voice_name, speed=1.0, lang="en-us")
                async for samples, _ in stream:
                    pcm_data = (samples * 32767).astype(np.int16).tobytes()
                    audio_queue.put(pcm_data)
            except Exception as e:
                audio_queue.put(e)
            finally:
                audio_queue.put(None) 

        def run_async():
            asyncio.run(fetch_stream())

        threading.Thread(target=run_async, daemon=True).start()

        while True:
            chunk = audio_queue.get()
            if chunk is None: break
            if isinstance(chunk, Exception): raise chunk
            yield chunk


class TTSWorker(QThread):
    status_update = pyqtSignal(str)
    model_loaded = pyqtSignal(str, list) 
    tts_latency = pyqtSignal(float) 

    def __init__(self, initial_model=""):
        super().__init__()
        self.sentence_queue = queue.Queue()
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.active_adapter = None 
        self.active_voice_name = "Default"
        self.current_device_index = None 
        
        # --- NEW: Voice Bypass Flag ---
        self.is_muted = False
        
        if initial_model:
            self.load_voice(initial_model)

    # --- NEW: Mute Toggle Method ---
    def set_mute(self, muted: bool):
        self.is_muted = muted
        state = "Muted" if muted else "Active"
        self.status_update.emit(f"TTS Voice is now {state}")

    def set_device(self, index):
        self.current_device_index = index
        if self.active_adapter:
            self.load_voice(self.model_path) 
            
    def set_voice(self, voice_name):
        self.active_voice_name = voice_name
        self.status_update.emit(f"Voice set to: {voice_name}")

    def load_voice(self, model_path):
        self.model_path = model_path
        filename = os.path.basename(model_path).lower()
        
        try:
            if "kokoro" in filename:
                self.active_adapter = KokoroAdapter(model_path)
            elif "piper" in filename or os.path.exists(model_path + ".json"):
                self.active_adapter = PiperAdapter(model_path)
            else:
                self.status_update.emit("Error: Unsupported Model Architecture.")
                return

            self.active_voice_name = self.active_adapter.available_voices[0]
            
            if self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except: pass

            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.active_adapter.sample_rate,
                output=True,
                output_device_index=self.current_device_index,
                frames_per_buffer=1024
            )
            
            self.model_loaded.emit(os.path.basename(model_path), self.active_adapter.available_voices)
            self.status_update.emit(f"TTS Engine Ready ({self.active_adapter.sample_rate}Hz)")
            
        except Exception as e:
            self.status_update.emit(f"Failed to load model: {e}")

    def add_to_queue(self, text):
        self.sentence_queue.put(text)
        if not self.isRunning():
            self.start()

    def run(self):
        while not self.sentence_queue.empty():
            text = self.sentence_queue.get()
            
            # --- NEW: Bypass Logic ---
            if self.is_muted:
                self.sentence_queue.task_done()
                continue
            
            if not self.active_adapter or not self.stream:
                self.sentence_queue.task_done()
                continue

            self.status_update.emit("🔊 Speaking...")
            first_chunk_played = False 
            start_time = time.time()   

            try:
                for pcm_data in self.active_adapter.synthesize(text, self.active_voice_name):
                    if not first_chunk_played:
                        self.tts_latency.emit((time.time() - start_time) * 1000)
                        first_chunk_played = True
                    self.stream.write(pcm_data)
                            
            except Exception as e:
                self.status_update.emit(f"Audio Error: {e}")
                    
            self.sentence_queue.task_done()
        self.status_update.emit("Idle")