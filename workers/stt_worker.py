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
    is_protected_stt_model,
    resolve_active_stt_model_spec,
    resolve_bundled_whisper_load_path,
)

logger = logging.getLogger("Qube.Audio")

class STTWorker(QThread):
    transcription_ready = pyqtSignal(str, int)
    transcription_failed = pyqtSignal(str, int)
    status_update = pyqtSignal(str)
    stt_latency = pyqtSignal(float)

    def __init__(self, *, eager_load: bool = True):
        super().__init__()
        self.stt_model = None
        self._active_spec = BUNDLED_STT_MODEL_ID
        self._job_epoch = 0
        self.audio_data = b""
        if eager_load:
            self._load_model()

    def _load_model(self) -> None:
        spec = resolve_active_stt_model_spec()
        self._active_spec = spec
        self.status_update.emit("BOOT: Loading Whisper Weights...")
        if os.path.isdir(spec) and not is_protected_stt_model(spec):
            logger.info("[STT] Loading custom model from %s", spec)
            self.stt_model = WhisperModel(spec, device="cpu", compute_type="int8")
        elif is_protected_stt_model(spec):
            load_path = resolve_bundled_whisper_load_path()
            if load_path is not None:
                logger.info("[STT] Loading bundled Whisper model from %s", load_path)
                self.stt_model = WhisperModel(
                    str(load_path),
                    device="cpu",
                    compute_type="int8",
                )
            else:
                logger.info("[STT] Loading bundled Whisper model: %s", spec)
                self.stt_model = WhisperModel(
                    spec,
                    device="cpu",
                    compute_type="int8",
                    download_root=get_stt_models_dir(),
                )
        else:
            logger.info("[STT] Loading Whisper model from %s", spec)
            self.stt_model = WhisperModel(spec, device="cpu", compute_type="int8")
        self.status_update.emit("STT Engine Ready")

    def _ensure_model_loaded(self) -> bool:
        """Load Whisper on first use when boot skipped eager load."""
        if self.stt_model is not None:
            return True
        try:
            logger.info("[STT] Lazy-loading Whisper model.")
            self._load_model()
        except Exception:
            logger.exception("[STT] Failed to lazy-load Whisper model.")
            self.stt_model = None
        return self.stt_model is not None

    def reload_from_settings(self) -> None:
        if self.isRunning():
            self.cancel_transcription()
            self.wait(5000)
        try:
            self.stt_model = None
            gc.collect()
            self._load_model()
        except Exception as e:
            logger.error("[STT] Reload failed: %s", e)
            self.status_update.emit(f"STT reload failed: {e}")

    def process_audio(self, raw_audio_bytes: bytes, job_epoch: int) -> None:
        """Queue audio for transcription; supersede any in-flight job."""
        self.audio_data = raw_audio_bytes or b""
        self._job_epoch = int(job_epoch)
        if self.isRunning():
            logger.info(
                "[STT] Superseding in-flight transcription (new job_epoch=%s).",
                self._job_epoch,
            )
            self.requestInterruption()
            if not self.wait(5000):
                logger.warning("[STT] Prior transcription thread did not exit within 5s.")
        self.start()

    def cancel_transcription(self) -> None:
        """Best-effort cancel for an in-flight ``run()`` (Stop / shutdown)."""
        if not self.isRunning():
            return
        logger.info(
            "[STT] Cancel requested (job_epoch=%s).",
            getattr(self, "_job_epoch", 0),
        )
        self.requestInterruption()

    def run(self):
        job_epoch = int(getattr(self, "_job_epoch", 0))
        if self.isInterruptionRequested():
            logger.info("[STT] Transcription skipped (cancelled before start, job_epoch=%s).", job_epoch)
            return
        if not self.audio_data:
            logger.info("[STT] Transcription skipped — empty audio buffer (job_epoch=%s).", job_epoch)
            self.transcription_ready.emit("", job_epoch)
            return
        if not self._ensure_model_loaded():
            reason = "Speech-to-text model not loaded"
            logger.warning(
                "[STT] Transcription aborted — %s (job_epoch=%s).",
                reason,
                job_epoch,
            )
            self.transcription_failed.emit(reason, job_epoch)
            return

        self.status_update.emit("Working...")
        start_time = time.time()
        audio_int16 = np.frombuffer(self.audio_data, np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        sample_count = int(audio_float32.shape[0]) if audio_float32.size else 0
        logger.info(
            "[STT] Transcription started (job_epoch=%s, samples=%d).",
            job_epoch,
            sample_count,
        )

        segments, _ = self.stt_model.transcribe(audio_float32, beam_size=5, language="en")

        latency_ms = (time.time() - start_time) * 1000
        if self.isInterruptionRequested():
            logger.info(
                "[STT] Transcription cancelled after decode (job_epoch=%s, latency_ms=%.0f).",
                job_epoch,
                latency_ms,
            )
            return

        full_text = ""
        for segment in segments:
            full_text += segment.text + " "

        text = full_text.strip()
        preview = text[:80]
        logger.info(
            "[STT] Transcription complete (job_epoch=%s, latency_ms=%.0f, chars=%d, preview=%r).",
            job_epoch,
            latency_ms,
            len(text),
            preview,
        )
        self.stt_latency.emit(latency_ms)
        self.transcription_ready.emit(text, job_epoch)
