from PyQt6.QtCore import QThread, pyqtSignal, QMutex
import pyaudio
import numpy as np
import time
import glob
import os
from openwakeword.model import Model
import logging

logger = logging.getLogger("Qube.Audio")

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280

class AudioListenerWorker(QThread):
    status_update = pyqtSignal(str)
    audio_captured = pyqtSignal(bytes)

    def __init__(self):
        super().__init__()
        
        wakeword_dir = os.path.join("models", "wakeword")
        custom_models = glob.glob(os.path.join(wakeword_dir, "*.tflite"))
        
        if custom_models:
            self.oww_model = Model(wakeword_models=custom_models)
            self.status_update.emit(f"Loaded {len(custom_models)} custom wake words.")
        else:
            self.oww_model = Model() 
            self.status_update.emit("Using default wake words.")
        
        self.mutex = QMutex()
        self.input_device_index = None  
        self.running = True
        self.stream = None
        self.audio = pyaudio.PyAudio()
        self.current_rate = 16000 
        self.current_volume = 0
        
        # UI Adjustable UX parameters
        self.silence_timeout = 2.0  
        # We now use a pure 0-100 percentage scale. 2% is a great default.
        self.speech_threshold = 2

    def set_silence_timeout(self, seconds: float):
        self.silence_timeout = seconds

    def set_speech_threshold(self, threshold: int):
        self.speech_threshold = threshold

    def set_input_device(self, index):
        self.mutex.lock()  
        try:
            self.input_device_index = index
            self.status_update.emit(f"Switching to device {index}...")
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
        finally:
            self.mutex.unlock()  

    def _record_until_silence(self):
        """The Sliding-Window Recording Logic (with real-time 48kHz resampling)"""
        self.status_update.emit("🎙️ RECORDING...")
        logger.info("Wake word detected. Opening recording buffer...")
        recording = []
        
        INITIAL_TIMEOUT = 5.0    
        silence_start_time = time.time()
        has_spoken = False

        read_chunk = 1024 if self.current_rate == 16000 else 1024 * 3

        while self.running:
            data = self.stream.read(read_chunk, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            # --- REAL-TIME TIME-WARP FIX ---
            if self.current_rate == 48000:
                audio_data = audio_data[::3]
                
            recording.append(audio_data.tobytes())
            
            # 1. Calculate raw peak volume
            try:
                peak = np.max(np.abs(audio_data))
                # 2. Convert to a pure 0-100 percentage
                current_vol_pct = min(100, int((peak / 32767.0) * 100))
            except ValueError:
                current_vol_pct = 0
                
            # Update the UI VU Meter
            self.current_volume = current_vol_pct

            # Tripwire logic: Compare the percentage against your slider!
            if current_vol_pct >= self.speech_threshold:
                silence_start_time = time.time()
                has_spoken = True

            elapsed_silence = time.time() - silence_start_time
            
            if not has_spoken and elapsed_silence > INITIAL_TIMEOUT:
                self.status_update.emit("Idle")
                return 
                
            if has_spoken and elapsed_silence > self.silence_timeout:
                self.status_update.emit("Transcribing...")
                logger.info(f"Silence timeout reached ({self.silence_timeout}s). Closing buffer.")
                
                audio_bytes = b''.join(recording)
                self.audio_captured.emit(audio_bytes)
                return

    def run(self):
        self.status_update.emit("BOOT: Loading Wake Word...")
        last_trigger_time = 0.0

        while self.running:
            self.mutex.lock()
            try:
                if self.stream is None:
                    try:
                        self.current_rate = 16000
                        self.stream = self.audio.open(format=FORMAT, channels=1, rate=16000,
                                                    input=True, frames_per_buffer=CHUNK,
                                                    input_device_index=self.input_device_index)
                        self.status_update.emit("Listening (16kHz)...")
                    except:
                        try:
                            self.current_rate = 48000
                            self.stream = self.audio.open(format=FORMAT, channels=1, rate=48000,
                                                        input=True, frames_per_buffer=CHUNK * 3,
                                                        input_device_index=self.input_device_index)
                            self.status_update.emit("Listening (48kHz HW Fallback)...")
                        except Exception as e:
                            self.status_update.emit(f"Mic Error: {e}")
                            self.mutex.unlock()
                            time.sleep(2)
                            continue

                read_chunk = CHUNK if self.current_rate == 16000 else CHUNK * 3
                data = self.stream.read(read_chunk, exception_on_overflow=False)
            finally:
                self.mutex.unlock()

            # Process audio for Wake Word detection
            audio_data = np.frombuffer(data, dtype=np.int16)
            if self.current_rate == 48000:
                audio_data = audio_data[::3] 
                
            try:
                # Apply the exact same accurate percentage math while idle
                peak = np.max(np.abs(audio_data))
                self.current_volume = min(100, int((peak / 32767.0) * 100))
            except ValueError:
                self.current_volume = 0
            
            self.oww_model.predict(audio_data)

            for wakeword_name in self.oww_model.prediction_buffer.keys():
                if len(self.oww_model.prediction_buffer[wakeword_name]) > 0:
                    score = list(self.oww_model.prediction_buffer[wakeword_name])[-1]
                    if score > 0.5:
                        if (time.time() - last_trigger_time) > 2.0:
                            self.mutex.lock()
                            try:
                                self._record_until_silence()
                            finally:
                                self.mutex.unlock()
                                last_trigger_time = time.time()
                                self.status_update.emit("Processing...")
                            break