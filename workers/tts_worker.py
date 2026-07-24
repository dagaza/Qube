from PyQt6.QtCore import QThread, pyqtSignal
import queue
import pyaudio
import numpy as np
import os
import requests
import time
import logging
logger = logging.getLogger("Qube.Audio")

# Queued after the last sentence of an LLM turn so playback_finished tracks real end-of-audio.
_END_OF_LLM_TURN = object()
# Wake the consumer so run() can exit on app shutdown.
_TTS_SHUTDOWN = object()
# Settings voice preview — bypasses mute and avoids chat UI playback signals.
_VOICE_PREVIEW = object()
_PLAYBACK_LEVEL_EMIT_INTERVAL_S = 0.04


def _pcm_peak_level(pcm: bytes) -> float:
    """Normalized 0–1 loudness from int16 PCM (companion visualizer)."""
    if not pcm:
        return 0.0
    arr = np.frombuffer(pcm, dtype=np.int16)
    if arr.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(arr.astype(np.float32) ** 2))) / 32768.0
    return min(1.0, rms * 2.8)


def ensure_bundled_kokoro_assets(model_path: str, *, allow_download: bool = False) -> None:
    """Ensure Kokoro ONNX + voices exist locally; download only when explicitly allowed."""
    from core.tts_models import (
        BUNDLED_DEFAULT_FILENAME,
        BUNDLED_VOICES_FILENAME,
        bundled_default_path,
        is_protected_tts_model,
    )

    if not is_protected_tts_model(model_path):
        return

    base_dir = os.path.dirname(bundled_default_path())
    os.makedirs(base_dir, exist_ok=True)

    onnx_path = os.path.join(base_dir, BUNDLED_DEFAULT_FILENAME)
    bin_path = os.path.join(base_dir, BUNDLED_VOICES_FILENAME)

    files_to_check = {
        onnx_path: "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v1.0.onnx",
        bin_path: "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices-v1.0.bin",
    }

    for file_path, url in files_to_check.items():
        if not os.path.exists(file_path):
            if not allow_download:
                raise FileNotFoundError(
                    f"Missing Kokoro asset: {os.path.basename(file_path)} "
                    "(download from Settings → Voice & Audio or first-run bootstrap)."
                )
            print(f"[SYSTEM] Downloading missing required file: {os.path.basename(file_path)}...")
            response = requests.get(url, stream=True)
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"[SYSTEM] Download complete: {os.path.basename(file_path)}")


def ensure_model_exists(model_path: str) -> None:
    """Backward-compatible alias; allows network fetch when caller expects bootstrap."""
    ensure_bundled_kokoro_assets(model_path, allow_download=True)

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

        from core.tts_models import BUNDLED_VOICES_FILENAME, is_protected_tts_model

        base_dir = os.path.dirname(model_path)
        voices_path = os.path.join(base_dir, BUNDLED_VOICES_FILENAME)

        if is_protected_tts_model(model_path):
            ensure_bundled_kokoro_assets(model_path, allow_download=False)
        
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
    playback_started = pyqtSignal(str)
    playback_finished = pyqtSignal()
    turn_settled = pyqtSignal()
    playback_level = pyqtSignal(float)

    def __init__(self, initial_model=""):
        super().__init__()
        self.sentence_queue = queue.Queue()
        self.audio = pyaudio.PyAudio()
        self.pyaudio_instance = None
        self.stream = None
        self.active_adapter = None
        self.active_voice_name = "Default"
        self.current_device_index = None
        self._last_playback_level_emit = 0.0
        
        # --- NEW: Voice Bypass Flag ---
        self.is_muted = False
        self._playback_active = False
        self._last_queued_tts_key = ""
        
        if initial_model:
            self.load_voice(initial_model)

    # --- NEW: Mute Toggle Method ---
    @property
    def is_playing(self) -> bool:
        return self._playback_active

    def set_mute(self, muted: bool):
        self.is_muted = muted
        logger.info("[TTS] Mute toggled -> %s", "ON" if muted else "OFF")
        # If user disables voice while audio is in-flight, cut playback immediately.
        if muted:
            self.stop_playback()
        state = "Muted" if muted else "Active"
        self.status_update.emit(f"TTS Voice is now {state}")

    def set_device(self, index):
        self.current_device_index = index
        if self.active_adapter:
            self.load_voice(self.model_path) 
            
    def set_voice(self, voice_name):
        self.active_voice_name = voice_name
        self.status_update.emit(f"Voice set to: {voice_name}")

    def _normalize_tts_queue_key(self, text: str) -> str:
        return " ".join(str(text or "").split()).strip().lower()

    def load_voice(self, model_path) -> bool:
        """Load a TTS ONNX model. Returns True on success; restores the prior adapter on failure."""
        from core.tts_models import UNSUPPORTED_TTS_ARCHITECTURE_MSG, classify_tts_architecture

        previous_adapter = self.active_adapter
        previous_path = getattr(self, "model_path", "")
        previous_voice = self.active_voice_name
        previous_stream = self.stream

        architecture = classify_tts_architecture(model_path)
        if architecture is None:
            self.status_update.emit(f"TTS error: {UNSUPPORTED_TTS_ARCHITECTURE_MSG}")
            return False

        try:
            if architecture == "kokoro":
                new_adapter = KokoroAdapter(model_path)
            else:
                new_adapter = PiperAdapter(model_path)

            if previous_stream:
                try:
                    previous_stream.stop_stream()
                    previous_stream.close()
                except Exception:
                    pass

            new_stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=new_adapter.sample_rate,
                output=True,
                output_device_index=self.current_device_index,
                frames_per_buffer=1024,
            )

            self.model_path = model_path
            self.active_adapter = new_adapter
            self.stream = new_stream
            from core.tts_models import resolve_default_tts_voice

            self.active_voice_name = resolve_default_tts_voice(
                self.active_adapter.available_voices
            )

            self.model_loaded.emit(os.path.basename(model_path), self.active_adapter.available_voices)
            self.status_update.emit(f"TTS Engine Ready ({self.active_adapter.sample_rate}Hz)")
            return True

        except Exception as e:
            self.active_adapter = previous_adapter
            self.model_path = previous_path
            self.active_voice_name = previous_voice
            self.stream = previous_stream
            logger.exception("[TTS] Failed to load model %s: %s", model_path, e)
            self.status_update.emit(f"Failed to load TTS model: {e}")
            return False

    def add_to_queue(self, text, session_id="default"):
        """Modified to accept a session_id for the Memory Brain."""
        # 🔑 THE FAILSAFE: Never queue empty text!
        if not text or not text.strip():
            return

        queue_key = self._normalize_tts_queue_key(text)
        if queue_key and queue_key == self._last_queued_tts_key:
            logger.debug("[TTS] Skipping duplicate queued sentence.")
            return
        self._last_queued_tts_key = queue_key

        # Store as a tuple so the session_id travels with the text
        self.sentence_queue.put((text, session_id))
        if not self.isRunning():
            self.start()

    def queue_voice_preview(self, text: str) -> None:
        """Play a short sample in Settings; ignores mute and chat playback UI."""
        if not text or not text.strip() or not self.active_adapter:
            return
        self._interrupt_tts = True
        self._last_queued_tts_key = ""
        if hasattr(self, "sentence_queue"):
            try:
                with self.sentence_queue.mutex:
                    self.sentence_queue.queue.clear()
            except Exception:
                pass
        self.sentence_queue.put((_VOICE_PREVIEW, text.strip()))
        if not self.isRunning():
            self.start()

    def enqueue_turn_complete(self, _session_id: str | None = None) -> None:
        """Call once per LLM response after streaming ends (with or without TTS chunks)."""
        self.sentence_queue.put(_END_OF_LLM_TURN)
        if not self.isRunning():
            self.start()

    def request_graceful_stop(self) -> None:
        """Unblocks run() so the thread can exit during app shutdown."""
        logger.info("[TTS] Graceful stop requested.")
        try:
            self.sentence_queue.put_nowait(_TTS_SHUTDOWN)
        except Exception:
            pass

    def _signal_playback_finished(self) -> None:
        if not self._playback_active:
            return
        self._playback_active = False
        self.playback_level.emit(0.0)
        self.playback_finished.emit()

    def run(self):
        import pyaudio
        import time

        _SYNTHESIS_CAP_S = 180.0

        try:
            while True:
                item = self.sentence_queue.get()

                if item is _TTS_SHUTDOWN:
                    logger.info("[TTS] Shutdown sentinel received; exiting run loop.")
                    self._signal_playback_finished()
                    break

                if item is _END_OF_LLM_TURN:
                    logger.debug("[TTS] End-of-turn sentinel received.")
                    self._last_queued_tts_key = ""
                    if self._playback_active:
                        self._signal_playback_finished()
                    else:
                        # No audio was output (muted, missing adapter, empty response).
                        # Still signal end-of-turn so main.py can return to Idle.
                        self.playback_level.emit(0.0)
                        self.playback_finished.emit()
                    self.turn_settled.emit()
                    continue

                self._interrupt_tts = False

                voice_preview = False
                if isinstance(item, tuple) and len(item) == 2 and item[0] is _VOICE_PREVIEW:
                    text, session_id = item[1], "__voice_preview__"
                    voice_preview = True
                elif isinstance(item, tuple):
                    text, session_id = item
                else:
                    text, session_id = item, "default"

                logger.info(f"[TTS] Preparing to speak: '{text[:40]}...'")

                if (
                    not voice_preview
                    and (getattr(self, 'is_muted', False) or not self.active_adapter)
                ):
                    logger.info(
                        "[TTS] Skipping speech (muted=%s, adapter_ready=%s).",
                        bool(getattr(self, "is_muted", False)),
                        bool(self.active_adapter),
                    )
                    continue

                if getattr(self, 'stream', None) is None:
                    try:
                        if getattr(self, 'pyaudio_instance', None) is None:
                            self.pyaudio_instance = pyaudio.PyAudio()

                        sample_rate = (
                            self.active_adapter.sample_rate
                            if self.active_adapter is not None
                            else 24000
                        )
                        self.stream = self.pyaudio_instance.open(
                            format=pyaudio.paInt16,
                            channels=1,
                            rate=sample_rate,
                            output=True,
                            output_device_index=self.current_device_index,
                        )
                    except Exception as e:
                        self.status_update.emit(f"Failed to rebuild stream: {e}")
                        continue

                self.status_update.emit("🔊 Previewing voice..." if voice_preview else "🔊 Speaking...")
                first_chunk_played = False
                start_time = time.time()
                syn_deadline = time.time() + _SYNTHESIS_CAP_S

                try:
                    for pcm_data in self.active_adapter.synthesize(text, self.active_voice_name):
                        if time.time() > syn_deadline:
                            logger.error("[TTS] Synthesis exceeded time cap; stopping sentence.")
                            break
                        if getattr(self, '_interrupt_tts', False):
                            break

                        if not first_chunk_played:
                            self.tts_latency.emit((time.time() - start_time) * 1000)
                            if not voice_preview:
                                self.playback_started.emit(session_id)
                                self._playback_active = True
                            first_chunk_played = True

                        CHUNK_SIZE = 4096
                        for i in range(0, len(pcm_data), CHUNK_SIZE):
                            if getattr(self, '_interrupt_tts', False):
                                logger.debug("Micro-chunker caught interruption! Stopping playback.")
                                break
                            chunk = pcm_data[i:i + CHUNK_SIZE]
                            now = time.time()
                            if now - self._last_playback_level_emit >= _PLAYBACK_LEVEL_EMIT_INTERVAL_S:
                                self.playback_level.emit(_pcm_peak_level(chunk))
                                self._last_playback_level_emit = now
                            self.stream.write(chunk)

                except Exception as e:
                    self.status_update.emit(f"Audio Error: {e}")

                if voice_preview:
                    self.playback_level.emit(0.0)
                    continue

        except Exception as e:
            logger.exception("[TTS] run loop failed: %s", e)
            self._signal_playback_finished()

    def stop_playback(self):
        """Thread-safe kill switch. Empties queue and flags the loop to stop."""
        dropped = None
        if hasattr(self, "sentence_queue"):
            try:
                dropped = self.sentence_queue.qsize()
            except Exception:
                dropped = None
        logger.info(
            "[TTS] Interruption received. Clearing queue (queued_items_before_clear=%s).",
            dropped if dropped is not None else "unknown",
        )
        self._interrupt_tts = True
        self.playback_level.emit(0.0)
        self._last_queued_tts_key = ""

        if hasattr(self, 'sentence_queue'):
            try:
                with self.sentence_queue.mutex:
                    self.sentence_queue.queue.clear()
            except Exception:
                pass
        # Unblock run() if it is waiting on sentence_queue.get()
        try:
            self.sentence_queue.put_nowait(_END_OF_LLM_TURN)
        except Exception:
            pass

        # ❌ We DO NOT close PyAudio here! It causes Segfaults!

    def close_audio_resources(self) -> None:
        """Release audio handles only after playback loop has stopped."""
        if self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        if self.audio is not None:
            try:
                self.audio.terminate()
            except Exception:
                pass
            self.audio = None
        if self.pyaudio_instance is not None:
            try:
                self.pyaudio_instance.terminate()
            except Exception:
                pass
            self.pyaudio_instance = None
        logger.info("[TTS] Output audio resources closed.")