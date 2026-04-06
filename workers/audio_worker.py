from PyQt6.QtCore import QThread, pyqtSignal, QMutex
import pyaudio
import numpy as np
import time
import glob
import os
import openwakeword
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
    # --- NEW: Signal to populate the UI dropdown ---
    wakewords_ready = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.mutex = QMutex()
        self.input_device_index = None  
        self.running = True
        self.stream = None
        self.audio = pyaudio.PyAudio()
        self.current_rate = 16000 
        self.current_volume = 0
        
        self.silence_timeout = 2.0  
        self.speech_threshold = 2
        
        # --- FIX: Dynamic Model Discovery using exact file paths ---
        self.available_wakewords = self._discover_wakewords()
        
        # Default to the first available (usually 'Alexa')
        if self.available_wakewords:
            self.active_wakeword_name = list(self.available_wakewords.keys())[0]
            active_path = self.available_wakewords[self.active_wakeword_name]
            self.oww_model = Model(wakeword_model_paths=[active_path])
        else:
            logger.error("CRITICAL: No wakeword models found anywhere!")

    def _discover_wakewords(self) -> dict:
        """Scans for default and custom wakewords, returning a clean UI Name -> File Path mapping."""
        wakeword_map = {}
        
        # 1. Ask openwakeword where its default models live on the hard drive
        try:
            pretrained_paths = openwakeword.get_pretrained_model_paths()
            for path in pretrained_paths:
                # Clean up "alexa_v0.1.tflite" into "Alexa" for the UI dropdown
                clean_name = os.path.basename(path).split('_v')[0].replace('.tflite', '').replace('.onnx', '').capitalize()
                wakeword_map[clean_name] = path
        except Exception as e:
            logger.warning(f"Could not load pre-trained wakewords: {e}")

        # 2. Append any custom models from your local models/wakeword folder
        wakeword_dir = os.path.join("models", "wakeword")
        os.makedirs(wakeword_dir, exist_ok=True)
        custom_paths = glob.glob(os.path.join(wakeword_dir, "*.tflite")) + glob.glob(os.path.join(wakeword_dir, "*.onnx"))
        
        for path in custom_paths:
            clean_name = os.path.basename(path).replace('.tflite', '').replace('.onnx', '')
            wakeword_map[f"Custom: {clean_name}"] = path
            
        return wakeword_map

    def emit_available_wakewords(self):
        """Emits just the clean names (keys) to populate the UI ComboBox."""
        self.wakewords_ready.emit(list(self.available_wakewords.keys()))

    def set_wakeword(self, ui_name: str):
        """Receives the clean UI name, grabs the real file path, and hot-swaps the model."""
        self.mutex.lock()
        try:
            target_path = self.available_wakewords.get(ui_name)
            if not target_path:
                logger.error(f"Wakeword '{ui_name}' not found in the mapping dictionary.")
                return
                
            self.status_update.emit(f"Loading Wakeword: {ui_name}...")
            
            # Feed the exact, absolute path to the ONNX runtime
            self.oww_model = Model(wakeword_model_paths=[target_path])
            self.active_wakeword_name = ui_name
            
            logger.info(f"Successfully hot-swapped wakeword to: {ui_name}")
            self.status_update.emit("Idle")
        except Exception as e:
            logger.error(f"Failed to load wakeword {ui_name}: {e}")
            self.status_update.emit("Error loading Wakeword")
        finally:
            self.mutex.unlock()

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
            
            if self.current_rate == 48000:
                audio_data = audio_data[::3]
                
            recording.append(audio_data.tobytes())
            
            try:
                peak = np.max(np.abs(audio_data))
                current_vol_pct = min(100, int((peak / 32767.0) * 100))
            except ValueError:
                current_vol_pct = 0
                
            self.current_volume = current_vol_pct

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

            audio_data = np.frombuffer(data, dtype=np.int16)
            if self.current_rate == 48000:
                audio_data = audio_data[::3] 
                
            try:
                peak = np.max(np.abs(audio_data))
                self.current_volume = min(100, int((peak / 32767.0) * 100))
            except ValueError:
                self.current_volume = 0
            
            # --- NEW: Mutex protection for hot-swapping during inference ---
            self.mutex.lock()
            try:
                self.oww_model.predict(audio_data)
                
                for wakeword_name in self.oww_model.prediction_buffer.keys():
                    if len(self.oww_model.prediction_buffer[wakeword_name]) > 0:
                        score = list(self.oww_model.prediction_buffer[wakeword_name])[-1]
                        if score > 0.5:
                            if (time.time() - last_trigger_time) > 2.0:
                                self._record_until_silence()
                                last_trigger_time = time.time()
                                self.status_update.emit("Processing...")
                                break
            finally:
                self.mutex.unlock()