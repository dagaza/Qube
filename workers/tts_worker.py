from PyQt6.QtCore import QThread, pyqtSignal
import queue
import pyaudio
import os
import time
import logging

from workers.tts_providers.kokoro_provider import KokoroProvider
from workers.tts_providers.f5_provider import F5Provider
from workers.tts_providers.voxtral_provider import VoxtralProvider
from workers.tts_providers.openai_provider import OpenAIProvider

logger = logging.getLogger("Qube.Audio")


class TTSWorker(QThread):
    status_update = pyqtSignal(str)
    model_loaded = pyqtSignal(str, list, bool)
    tts_latency = pyqtSignal(float)

    class VoiceContext:
        def __init__(self):
            self.audio_path = None
            self.text = ""
            self.name = "default"

    def __init__(self, initial_model=""):
        super().__init__()

        self.sentence_queue = queue.Queue()
        self.audio = pyaudio.PyAudio()
        self.stream = None

        self.active_provider = None
        self.current_model_path = None
        self.current_device_index = None
        self.is_muted = False

        self.voice = self.VoiceContext()

        if initial_model:
            self.load_voice(initial_model)

    # --------------------------------------------------
    # SAFE VOICE SET
    # --------------------------------------------------
    def set_voice(self, voice_path: str):
        logger.info(f"[VOICE] Set voice: {voice_path}")

        self.voice.audio_path = voice_path
        self.voice.name = os.path.basename(voice_path)

        if not self.active_provider:
            return

        try:
            # ONLY pass what provider supports
            if hasattr(self.active_provider, "set_voice"):
                self.active_provider.set_voice(voice_path)
        except Exception as e:
            logger.error(f"Voice set failed: {e}")

    # --------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------
    def load_voice(self, model_path_or_url):
        self.current_model_path = model_path_or_url
        target_id = str(model_path_or_url).lower()

        try:
            if self.active_provider:
                try:
                    self.active_provider.unload_model()
                except:
                    pass

            if target_id.startswith("http"):
                self.active_provider = OpenAIProvider()

            elif "kokoro" in target_id:
                self.active_provider = KokoroProvider()

            elif "f5" in target_id:
                self.active_provider = F5Provider()

            elif "voxtral" in target_id:
                self.active_provider = VoxtralProvider()

            else:
                self.status_update.emit("Unsupported TTS model")
                return

            if not self.active_provider.load_model(model_path_or_url):
                self.status_update.emit("Failed to load TTS model")
                return

            if self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except:
                    pass

            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.active_provider.sample_rate,
                output=True,
                output_device_index=self.current_device_index,
                frames_per_buffer=1024
            )

            # restore voice safely
            if self.voice.audio_path:
                self.set_voice(self.voice.audio_path)

            voices = getattr(self.active_provider, "available_voices", [])
            supports = getattr(self.active_provider, "supports_voice_selection", False)

            model_name = os.path.basename(model_path_or_url)

            self.model_loaded.emit(model_name, voices, supports)
            self.status_update.emit(f"TTS Ready ({self.active_provider.sample_rate}Hz)")

        except Exception as e:
            logger.error(f"Load error: {e}", exc_info=True)
            self.status_update.emit(f"Load error: {str(e)[:40]}")

    # --------------------------------------------------
    # QUEUE
    # --------------------------------------------------
    def add_to_queue(self, text):
        self.sentence_queue.put(text)
        if not self.isRunning():
            self.start()

    # --------------------------------------------------
    # RUN LOOP
    # --------------------------------------------------
    def run(self):
        while not self.sentence_queue.empty():
            self._interrupt_tts = False
            text = self.sentence_queue.get()

            if self.is_muted or not self.active_provider:
                self.sentence_queue.task_done()
                continue

            self.status_update.emit("🔊 Speaking...")

            start_time = time.time()
            first_chunk = False

            try:
                cancel = lambda: getattr(self, "_interrupt_tts", False)

                stream = self.active_provider.generate_audio_stream(
                    text,
                    cancel
                )

                for pcm in stream:
                    if cancel():
                        break

                    if not first_chunk:
                        self.tts_latency.emit((time.time() - start_time) * 1000)
                        first_chunk = True

                    for i in range(0, len(pcm), 4096):
                        if cancel():
                            break
                        self.stream.write(pcm[i:i+4096])

            except Exception as e:
                logger.error(f"TTS runtime error: {e}", exc_info=True)
                self.status_update.emit(f"Audio Error: {e}")

            self.sentence_queue.task_done()

    def stop_playback(self):
        self._interrupt_tts = True
        with self.sentence_queue.mutex:
            self.sentence_queue.queue.clear()