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

    # 🔑 NEW: Signal to broadcast live audio levels (0.0 to 1.0)
    volume_update = pyqtSignal(float)

    wakeword_detected = pyqtSignal() # 🔑 THE NEW SIGNAL

    def __init__(self):
        super().__init__()
        self.mutex = QMutex()
        self.input_device_index = None  
        self.running = True
        self.is_paused = False
        self.stream = None
        self.audio = pyaudio.PyAudio()
        self.current_rate = 16000 
        self.current_volume = 0
        self.ignore_mic_until = 0.0 # 🔑 NEW: Hardware mute timer
        
        self.silence_timeout = 2.0  
        self.speech_threshold = 2
        
        # --- FIX: Dynamic Model Discovery using exact file paths ---
        self.available_wakewords = self._discover_wakewords()
        
        # --- NEW: Thread-safe loading flags ---
        self.pending_wakeword = None
        self.pending_device_index = None
        self.oww_model = None
        
        # Default to the first available (usually 'Alexa')
        if self.available_wakewords:
            self.active_wakeword_name = list(self.available_wakewords.keys())[0]
            # Tell the run loop to load this path immediately when it starts
            self.pending_wakeword = self.available_wakewords[self.active_wakeword_name]
        else:
            self.active_wakeword_name = None
            logger.error("CRITICAL: No wakeword models found anywhere!")

    def set_paused(self, paused: bool):
        """Thread-safe request to pause the audio stream."""
        self.is_paused = paused
        if paused:
            self.status_update.emit("Voice Input Deactivated")
        else:
            self.status_update.emit("Boot: Reconnecting Mic...")

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
        """Receives the clean UI name and flags the background thread to hot-swap."""
        self.mutex.lock()
        try:
            target_path = self.available_wakewords.get(ui_name)
            if not target_path:
                logger.error(f"Wakeword '{ui_name}' not found in the mapping dictionary.")
                return
                
            self.active_wakeword_name = ui_name
            self.pending_wakeword = target_path  # The background thread will pick this up!
            logger.info(f"Wakeword swap requested: {ui_name}")
            
        except Exception as e:
            logger.error(f"Failed to request wakeword {ui_name}: {e}")
        finally:
            self.mutex.unlock()

    def set_silence_timeout(self, seconds: float):
        self.silence_timeout = seconds

    def set_speech_threshold(self, threshold: int):
        self.speech_threshold = threshold

    def set_input_device(self, index):
        """Thread-safe request to swap the audio device."""
        self.pending_device_index = index
        self.status_update.emit(f"Swapping to device {index}...")

    def _record_until_silence(self):
        # 🔑 THE FIX: Imports must go at the absolute top of the scope!
        import time
        import numpy as np
        
        self.status_update.emit("🎙️ RECORDING...")
        logger.info("Wake word detected. Opening recording buffer...")

        # 🔑 THE TEMPORAL GATE SETUP
        # This ignores the first X seconds of audio to let speakers go silent.
        gate_start_time = time.time()
        DELAY_BUFFER = 0.65  # Adjust to 1.0 if you still hear "phantom" syllables
        
        # 🔑 THE ECHO FLUSH: Instantly throw away the mic's hardware buffer!
        try:
            if getattr(self, 'stream', None) and self.stream.get_read_available() > 0:
                self.stream.read(self.stream.get_read_available(), exception_on_overflow=False)
        except Exception as e:
            logger.debug(f"Safely ignored PyAudio buffer flush error: {e}")

        recording = []
        
        INITIAL_TIMEOUT = 5.0    
        silence_start_time = time.time()
        has_spoken = False

        read_chunk = 1024 if self.current_rate == 16000 else 1024 * 3

        while self.running:
            data = self.stream.read(read_chunk, exception_on_overflow=False)

            # 🔑 THE TEMPORAL GATE: Skip everything for the first 0.8 seconds
            if time.time() - gate_start_time < DELAY_BUFFER:
                # Reset the silence timer so the 5s timeout starts AFTER the gate opens
                silence_start_time = time.time() 
                continue

            audio_data = np.frombuffer(data, dtype=np.int16)
            
            if self.current_rate == 48000:
                audio_data = audio_data[::3]
                
            recording.append(audio_data.tobytes())
            
            try:
                peak = np.max(np.abs(audio_data))
                
                # Calculate and emit during active recording
                normalized_level = min(1.0, peak / 32767.0)
                self.volume_update.emit(float(normalized_level))
                
                current_vol_pct = int(normalized_level * 100)
            except ValueError:
                current_vol_pct = 0
                self.volume_update.emit(0.0)
                
            self.current_volume = current_vol_pct

            if current_vol_pct >= self.speech_threshold:
                silence_start_time = time.time()
                has_spoken = True
                
            elapsed_silence = time.time() - silence_start_time
            
            # If nothing is heard after the gate opens for 5 seconds, go idle
            if not has_spoken and elapsed_silence > INITIAL_TIMEOUT:
                self.status_update.emit("Idle")
                return 
                
            if has_spoken and elapsed_silence > self.silence_timeout:
                self.status_update.emit("Transcribing...")
                logger.info(f"Silence timeout reached ({self.silence_timeout}s). Closing buffer.")
                audio_bytes = b''.join(recording)
                self.audio_captured.emit(audio_bytes)
                return

    def _open_mic_with_warmup(self):
        """
        Opens the microphone stream and pre-warms the OWW model's feature buffer.

        The OWW LSTM needs ~76 frames of audio before its rolling mel-spectrogram
        window is full and produces non-zero scores. Previously the ALSA warmup flush
        discarded those chunks unused, so the model was effectively deaf for the first
        ~1 second after every (re)open. Feeding the flush chunks through predict()
        solves this for both cold boot and hot-swaps.

        Returns True on success, False on total failure.
        """
        # Try 16kHz first, fall back to 48kHz
        configs = [
            (16000, CHUNK),
            (48000, CHUNK * 3),
        ]

        for rate, chunk_size in configs:
            try:
                self.current_rate = rate
                self.stream = self.audio.open(
                    format=FORMAT, channels=1, rate=rate,
                    input=True, frames_per_buffer=chunk_size,
                    input_device_index=self.input_device_index
                )
                logger.info(f"Mic opened at {rate}Hz on device {self.input_device_index}")
                self.status_update.emit("Idle")

                # Flush ALSA startup corruption AND pre-warm the OWW feature buffer
                try:
                    for _ in range(15):
                        flush_data = self.stream.read(chunk_size, exception_on_overflow=False)
                        if self.oww_model:
                            flush_audio = np.frombuffer(flush_data, dtype=np.int16)
                            if rate == 48000:
                                flush_audio = flush_audio[::3]
                            self.oww_model.predict(flush_audio)
                except Exception:
                    pass
                logger.info("Hardware buffer flush complete. OWW model pre-warmed.")
                return True

            except Exception as e:
                logger.warning(f"Mic open failed at {rate}Hz: {e}")
                if self.stream:
                    try:
                        self.stream.stop_stream()
                        self.stream.close()
                    except Exception:
                        pass
                    self.stream = None

        self.status_update.emit("Mic Error: ALSA Lock")
        return False

    def run(self):
        last_trigger_time = 0.0
        time.sleep(2.5)

        while self.running:
            # --- 0. Thread-Safe Teardown (For Pauses & Device Swaps) ---
            needs_reboot = False
            
            if getattr(self, 'pending_device_index', None) is not None:
                self.input_device_index = self.pending_device_index
                self.pending_device_index = None
                needs_reboot = True
                logger.info(f"Target input device updated to {self.input_device_index}")
                
            if getattr(self, 'is_paused', False) and self.stream is not None:
                needs_reboot = True

            if needs_reboot and self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None
                logger.info("Hardware audio stream safely released by background thread.")

            # --- 1. Check for Pause State ---
            if getattr(self, 'is_paused', False):
                time.sleep(0.5)
                continue

            # --- 2. Safely Load the ONNX Model ---
            if self.pending_wakeword:
                self.status_update.emit("Loading Wakeword...")

                # Close the mic stream before reloading the model.
                # This is essential for hot-swaps: the new model must open a fresh
                # stream so the warmup flush pre-warms *its* feature buffer from
                # scratch. If the stream stays open, the new model inherits a stale
                # rolling buffer from the previous model and scores stay at 0.0.
                # On cold boot self.stream is already None, so this is a safe no-op.
                if self.stream is not None:
                    try:
                        self.stream.stop_stream()
                        self.stream.close()
                    except Exception:
                        pass
                    self.stream = None
                    logger.info("Audio stream closed for clean wakeword hot-swap.")

                try:
                    logger.info(f"Loading ONNX Model: {os.path.basename(self.pending_wakeword)}")
                    self.oww_model = Model(wakeword_model_paths=[self.pending_wakeword])
                    logger.info("ONNX Model initialized successfully.")
                except Exception as e:
                    logger.error("CRITICAL: Failed to load wakeword model!", exc_info=True)

                self.pending_wakeword = None
                self.status_update.emit("Idle")
                continue

            if self.oww_model is None:
                time.sleep(0.5)
                continue

            # --- 3. Open the Microphone & Read Audio ---
            if self.stream is None:
                if not self._open_mic_with_warmup():
                    time.sleep(2)
                    continue

            # Read the live, clean audio buffer
            try:
                read_chunk = CHUNK if self.current_rate == 16000 else CHUNK * 3
                data = self.stream.read(read_chunk, exception_on_overflow=False)
            except Exception as read_err:
                logger.error(f"Failed to read audio stream (Zombie Stream detected): {read_err}")
                if self.stream:
                    try:
                        self.stream.stop_stream()
                        self.stream.close()
                    except Exception:
                        pass
                    self.stream = None
                time.sleep(1)
                continue

            # --- 4. Process Audio ---
            audio_data = np.frombuffer(data, dtype=np.int16)
            if self.current_rate == 48000:
                audio_data = audio_data[::3] 
                
            try:
                peak = np.max(np.abs(audio_data))
                
                # 🔑 THE FIX: Calculate the 0.0 to 1.0 float and emit it
                normalized_level = min(1.0, peak / 32767.0)
                self.volume_update.emit(float(normalized_level))
                
                # Keep your existing integer logic for the logger
                self.current_volume = int(normalized_level * 100)
                
                # Volume Logger
                if self.current_volume > 0 and (time.time() - getattr(self, '_last_log_time', 0) > 5.0):
                    logger.debug(f"Audio Buffer Active. Peak Volume: {self.current_volume}%")
                    self._last_log_time = time.time()
                    
            except ValueError:
                self.current_volume = 0
                self.volume_update.emit(0.0)
            
            # --- 5. Hardened Inference Engine ---
            try:
                self.oww_model.predict(audio_data)
                
                for wakeword_name in self.oww_model.prediction_buffer.keys():
                    if len(self.oww_model.prediction_buffer[wakeword_name]) > 0:
                        score = list(self.oww_model.prediction_buffer[wakeword_name])[-1]
                        if score > 0.5:
                            # 🔑 THE DEBOUNCE FIX: Put the trigger inside the 2-second cooldown!
                            if (time.time() - last_trigger_time) > 2.0:
                                logger.info(f"Wakeword '{wakeword_name}' triggered with score {score}!")
                                
                                # 1. NOW it only tells main.py once!
                                self.wakeword_detected.emit()
                                
                                # 2. Drop into recording
                                self._record_until_silence()
                                last_trigger_time = time.time()
                                break
            except Exception as e:
                logger.error("CRITICAL: Inference engine crashed during predict()!", exc_info=True)

    def _close_audio_resources(self) -> None:
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
        logger.info("[Audio] Input audio resources closed.")

    def stop(self):
        """Cooperative stop for shutdown; releases native audio handles."""
        logger.info("[Audio] Stop requested.")
        self.running = False
        self.is_paused = True
        self._close_audio_resources()