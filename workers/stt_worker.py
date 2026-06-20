from PyQt6.QtCore import QThread, pyqtSignal
import gc
import numpy as np
import time
from faster_whisper import WhisperModel
import logging
import os

from core.stt_models import (
    BUNDLED_STT_MODEL_ID,
    get_stt_models_dir,
    resolve_active_stt_model_spec,
)

logger = logging.getLogger("Qube.Audio")

class STTWorker(QThread):
    transcription_ready = pyqtSignal(str)
    status_update = pyqtSignal(str)
    stt_latency = pyqtSignal(float) 

    def __init__(self, *, eager_load: bool = True):
        super().__init__()
        self.stt_model = None
        self._active_spec = BUNDLED_STT_MODEL_ID
        if eager_load:
            self._load_model()

    def _load_model(self) -> None:
        spec = resolve_active_stt_model_spec()
        self._active_spec = spec
        self.status_update.emit("BOOT: Loading Whisper Weights...")
        if os.path.isdir(spec):
            logger.info("[STT] Loading custom model from %s", spec)
            self.stt_model = WhisperModel(spec, device="cpu", compute_type="int8")
        else:
            logger.info("[STT] Loading bundled Whisper model: %s", spec)
            self.stt_model = WhisperModel(
                spec,
                device="cpu",
                compute_type="int8",
                download_root=get_stt_models_dir(),
            )
        self.status_update.emit("STT Engine Ready")

    def reload_from_settings(self) -> None:
        if self.isRunning():
            self.requestInterruption()
            self.wait(5000)
        try:
            self.stt_model = None
            gc.collect()
            self._load_model()
        except Exception as e:
            logger.error("[STT] Reload failed: %s", e)
            self.status_update.emit(f"STT reload failed: {e}")

    def process_audio(self, raw_audio_bytes):
        self.audio_data = raw_audio_bytes
        self.start() 

    def run(self):
        if self.isInterruptionRequested():
            self.status_update.emit("STT: cancelled")
            return
        if self.stt_model is None:
            self.status_update.emit("STT: model not loaded")
            return
        self.status_update.emit("Transcribing...")
        start_time = time.time() 
        audio_int16 = np.frombuffer(self.audio_data, np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        
        segments, _ = self.stt_model.transcribe(audio_float32, beam_size=5, language="en")
        
        full_text = ""
        for segment in segments:
            full_text += segment.text + " "

        latency = (time.time() - start_time) * 1000 
        self.stt_latency.emit(latency)

        self.transcription_ready.emit(full_text.strip())
