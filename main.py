import sys
import os

from PyQt6 import QtCore
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont
from qt_material import apply_stylesheet

from workers import AudioListenerWorker, STTWorker, LLMWorker, TTSWorker
from workers.ingestion_worker import IngestionWorker  # noqa: F401 – kept for parity
from core.gpu_monitor import GPUMonitor
from rag.embedder import EmbeddingModel
from rag.store import DocumentStore
from ui.main_window import MainWindow
from ui.settings_dialog import SettingsDialog

import logging

# --- QUBE TERMINAL LOGGER SETUP ---
logging.basicConfig(
    level=logging.DEBUG,  # Set to INFO in production to hide the noise
    format='%(asctime)s.%(msecs)03d | %(levelname)-8s | [%(name)s] %(message)s',
    datefmt='%H:%M:%S'
)

# Create the main app logger
logger = logging.getLogger("Qube.Core")
logger.info("Terminal logging initialized. Booting sequence started.")

class Qube:
    def __init__(self):
        # -- 1. Shared services ------------------------------------------
        self.embedder = EmbeddingModel()
        self.store    = DocumentStore()

        # -- 2. Background workers ---------------------------------------
        self.audio_worker = AudioListenerWorker()
        self.stt_worker   = STTWorker()
        self.llm_worker   = LLMWorker(self.embedder, self.store)
        self.tts_worker   = TTSWorker()
        self.gpu_monitor  = GPUMonitor()

        workers = {
            "audio": self.audio_worker,
            "stt":   self.stt_worker,
            "llm":   self.llm_worker,
            "tts":   self.tts_worker,
        }

        # -- 3. UI -------------------------------------------------------
        # Create the main window
        self.window = MainWindow(workers=workers, gpu_monitor=self.gpu_monitor)

        # FIX: Instantiate the SettingsDialog here in the Orchestrator.
        # We pass self.window as the parent (for UI centering) 
        # and self.llm_worker as the specific dependency.
        self.settings_dialog = SettingsDialog(
            self.window, 
            self.llm_worker, 
            self.embedder, 
            self.store,
            self.audio_worker  # <-- Inject the audio worker here
        )
        
        # Manually wire the window's button to open this specific dialog
        self.window.settings_btn.clicked.connect(self.settings_dialog.exec)

        # -- 4. Wire signals ---------------------------------------------
        self._connect_signals()

        # -- 5. Start continuous processes --------------------------------
        self.audio_worker.start()

        # FIX 2: Now that the UI is listening, tell the worker to load the model!
        import os
        tts_path = os.path.join("models", "tts", "kokoro-v1.0.onnx")
        self.tts_worker.load_voice(tts_path)
    # ------------------------------------------------------------------ #
    #  Signal wiring                                                       #
    # ------------------------------------------------------------------ #

    def _connect_signals(self) -> None:
        """
        The 'Traffic Cop' that routes signals between workers and the UI.
        Nothing outside this method should call connect() – all cross-layer
        wiring lives here so it is trivial to audit and change.
        """
        w = self.window  # shorthand

        # Audio → STT
        self.audio_worker.audio_captured.connect(self.stt_worker.process_audio)

        # STT → UI & LLM
        self.stt_worker.transcription_ready.connect(w.log_user_message)
        self.stt_worker.transcription_ready.connect(self.llm_worker.generate_response)

        # LLM → UI (streaming tokens)
        self.llm_worker.token_streamed.connect(w.log_agent_token)

        # LLM → TTS (completed sentences)
        self.llm_worker.sentence_ready.connect(self.tts_worker.add_to_queue)

        # Status updates from all workers → UI status bar
        self.audio_worker.status_update.connect(w.update_status)
        self.stt_worker.status_update.connect(w.update_status)
        self.llm_worker.status_update.connect(w.update_status)
        self.tts_worker.status_update.connect(w.update_status)

        # Latency waterfall
        self.stt_worker.stt_latency.connect(w.update_stt_latency)
        self.llm_worker.ttft_latency.connect(w.update_ttft_latency)
        self.tts_worker.tts_latency.connect(w.update_tts_latency)

        # TTS model loaded → voice dropdown
        self.tts_worker.model_loaded.connect(w.update_voice_dropdown)

        # RAG context usage → indicator
        self.llm_worker.context_retrieved.connect(w.update_rag_indicator)

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def show(self) -> None:
        self.window.show()


if __name__ == "__main__":
    # PyQt6 high DPI handling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    
    # Force a clean, modern font globally
    app_font = QFont("Segoe UI", 10) 
    app_font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(app_font)

    # NEW: Apply the Material Design dark theme
    apply_stylesheet(app, theme='dark_blue.xml')

    qube = Qube()
    qube.show()
    sys.exit(app.exec())