import requests
import queue
import threading
import logging
from .base_provider import BaseTTSProvider

logger = logging.getLogger("Qube.TTS.API")

class OpenAIProvider(BaseTTSProvider):
    def __init__(self):
        super().__init__()
        self.api_url = "http://localhost:5050/v1/audio/speech" # Fallback default
        self.sample_rate = 24000
        # Standard OpenAI voices, though local APIs might ignore this or use their own
        self.available_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer", "custom"]
        self.current_voice = "alloy"

    def load_model(self, endpoint_url: str) -> bool:
        """
        For the API provider, 'loading the model' just means setting the target URL.
        """
        try:
            if endpoint_url.startswith("http"):
                self.api_url = endpoint_url
                logger.info(f"API Provider connected to endpoint: {self.api_url}")
                self.is_loaded = True
                return True
            else:
                logger.error("Invalid API URL provided.")
                return False
        except Exception as e:
            logger.error(f"Failed to set API endpoint: {e}")
            return False

    def set_voice(self, voice_name: str) -> bool:
        """Passes the requested voice name directly to the payload."""
        self.current_voice = voice_name
        return True

    def generate_audio_stream(self, text: str, cancel_flag_callback):
        """Fires the JSON payload to the server and streams the returning bytes."""
        audio_queue = queue.Queue()

        def fetch_api():
            try:
                payload = {
                    "model": "tts-1",
                    "input": text,
                    "voice": self.current_voice,
                    "response_format": "pcm" # Request raw PCM bytes for direct PyAudio streaming
                }
                
                # We use stream=True so we don't have to wait for the whole sentence to download
                response = requests.post(self.api_url, json=payload, stream=True, timeout=5)
                response.raise_for_status()

                # Stream the returning audio bytes in tiny 4KB chunks
                for chunk in response.iter_content(chunk_size=4096):
                    if cancel_flag_callback():
                        logger.info("API Stream severed by user interruption.")
                        break
                    if chunk:
                        audio_queue.put(chunk)
                        
            except requests.exceptions.RequestException as e:
                logger.error(f"Network error communicating with external TTS API: {e}")
                audio_queue.put(e)
            except Exception as e:
                logger.error(f"Unexpected API Provider error: {e}")
                audio_queue.put(e)
            finally:
                audio_queue.put(None)

        threading.Thread(target=fetch_api, daemon=True).start()

        # The main loop that feeds bytes back to the TTSWorker
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