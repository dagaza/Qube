from PyQt6.QtCore import QThread, pyqtSignal
import numpy as np
import time
from faster_whisper import WhisperModel
import logging
logger = logging.getLogger("Qube.Audio")

class STTWorker(QThread):
    transcription_ready = pyqtSignal(str)
    status_update = pyqtSignal(str)
    stt_latency = pyqtSignal(float) 

    def __init__(self):
        super().__init__()
        self.status_update.emit("BOOT: Loading Whisper Weights...")
        # This tells Whisper to check your local folder first
        self.stt_model = WhisperModel("small", device="cpu", compute_type="int8", download_root="models/stt")
        self.status_update.emit("STT Engine Ready")

    def process_audio(self, raw_audio_bytes):
        self.audio_data = raw_audio_bytes
        self.start() 

    def run(self):
        if self.isInterruptionRequested():
            self.status_update.emit("STT: cancelled")
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