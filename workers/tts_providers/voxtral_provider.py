import os
import sys
import torch
import logging
import queue
import threading
if sys.platform == "linux":
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
from .base_provider import BaseTTSProvider

logger = logging.getLogger("Qube.TTS.Voxtral")

class VoxtralProvider(BaseTTSProvider):
    def __init__(self):
        super().__init__()
        self.model = None
        self.sample_rate = 24000
        # 🔑 UNIVERSAL DEVICE DETECTION (2026 Cross-Platform Standard)
        if torch.cuda.is_available():
            # This covers NVIDIA (CUDA) AND AMD on Linux (ROCm)
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # This covers Apple Silicon (M1/M2/M3/M4)
            self.device = "mps"
        else:
            # Check for Windows-specific DirectML (AMD/Intel acceleration)
            try:
                import importlib
                tdml = importlib.import_module("torch_directml")
                if tdml.is_available():
                    self.device = tdml.device()
                else:
                    self.device = "cpu"
            except (ImportError, ModuleNotFoundError):
                self.device = "cpu"
        
        logger.info(f"Qube TTS Engine assigned to: {self.device}")
        self.reference_audio_path = None

    def load_model(self, model_path: str) -> bool:
        try:
            logger.info("Initializing Voxtral TTS in Quantized Mode...")
            from transformers import AutoModelForTextToSpectrogram, BitsAndBytesConfig
            
            # 🔑 THE SECRET SAUCE: 4-Bit Quantization
            # This crushes the 4B parameter model down to ~3GB of RAM!
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
            self.model = AutoModelForTextToSpectrogram.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto" # Let HuggingFace balance the RAM/VRAM
            )
            
            self.available_voices = ["Voxtral_Base", "Custom_Clone"]
            self.current_voice = "Voxtral_Base"
            self.is_loaded = True
            return True
        except ImportError:
            logger.error("Missing 'bitsandbytes' or 'transformers'. Cannot load Voxtral.")
            return False
        except Exception as e:
            logger.error(f"Voxtral failed to load: {e}")
            return False

    def set_voice(self, voice_name: str) -> bool:
        if voice_name.endswith(".wav") and os.path.exists(voice_name):
            self.reference_audio_path = voice_name
            self.current_voice = "Custom_Clone"
            return True
        elif voice_name in self.available_voices:
            self.current_voice = voice_name
            self.reference_audio_path = None
            return True
        return False

    def generate_audio_stream(self, text: str, cancel_flag_callback):
        """Streams Voxtral audio chunks securely."""
        audio_queue = queue.Queue()

        def inference_thread():
            try:
                # Simulated Voxtral streaming API
                stream = self.model.synthesize_stream(
                    text=text,
                    clone_from=self.reference_audio_path
                )
                for audio_tensor in stream:
                    if cancel_flag_callback():
                        logger.info("Voxtral generation aborted by user.")
                        break
                    pcm_data = (audio_tensor.cpu().numpy() * 32767).astype(np.int16).tobytes()
                    audio_queue.put(pcm_data)
            except Exception as e:
                audio_queue.put(e)
            finally:
                audio_queue.put(None)

        threading.Thread(target=inference_thread, daemon=True).start()

        while True:
            if cancel_flag_callback():
                break
            try:
                chunk = audio_queue.get(timeout=0.1)
                if chunk is None: break
                if isinstance(chunk, Exception): raise chunk
                yield chunk
            except queue.Empty:
                continue

    def unload_model(self):
        """Hard flush of the 4B parameter weights."""
        if self.model:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect() # Force Python garbage collection
        super().unload_model()