import sys
import os

from PyQt6 import QtCore
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont, QFontDatabase, QIcon

from workers import AudioListenerWorker, STTWorker, LLMWorker, TTSWorker
from workers.ingestion_worker import IngestionWorker 
from core.gpu_monitor import GPUMonitor
from rag.embedder import EmbeddingModel
from rag.store import DocumentStore
from ui.main_window import MainWindow
from core.database import DatabaseManager

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
        self.db_manager = DatabaseManager()
        self.audio_worker = AudioListenerWorker()
        self.stt_worker   = STTWorker()
        self.llm_worker   = LLMWorker(self.embedder, self.store, self.db_manager)
        self.tts_worker   = TTSWorker()
        self.gpu_monitor  = GPUMonitor()

        workers = {
            "audio": self.audio_worker,
            "stt":   self.stt_worker,
            "llm":   self.llm_worker,
            "tts":   self.tts_worker,
            "db": self.db_manager,
            "store": self.store
        }

        # -- 3. UI -------------------------------------------------------
        # Create the main window
        self.window = MainWindow(workers=workers, gpu_monitor=self.gpu_monitor)

        # -- 4. Wire signals ---------------------------------------------
        self._connect_signals()

        self._sync_databases()

        # -- 5. Start continuous processes --------------------------------
        
        # Start the worker with OS default (None) hardware routing (PipeWire)
        self.audio_worker.start()

        # Load the TTS Kokoro Model
        import os
        tts_path = os.path.join("models", "tts", "kokoro-v1.0.onnx")
        self.tts_worker.load_voice(tts_path)
    # ------------------------------------------------------------------ #
    #  Signal wiring                                                       #
    # ------------------------------------------------------------------ #

    def _connect_signals(self):
        w = self.window
        
        # 1. Global Shell Routing (Top Bar Status & RAG Dot)
        self.audio_worker.status_update.connect(w.update_status)
        self.stt_worker.status_update.connect(w.update_status)
        self.llm_worker.status_update.connect(w.update_status)
        self.tts_worker.status_update.connect(w.update_status)
        self.llm_worker.context_retrieved.connect(w.update_rag_indicator)

        # 2. Settings View Routing
        self.tts_worker.model_loaded.connect(self.window.update_global_voice_dropdown)

        # 3. Conversations View Routing (Transcript)
        # 🔑 UPDATED NAME HERE
        self.llm_worker.token_streamed.connect(w.conversations_view.log_agent_token)
        
        # 4. Background Data Pipeline (Audio -> STT -> LLM -> TTS)
        self.audio_worker.audio_captured.connect(self.stt_worker.process_audio)
        self.stt_worker.transcription_ready.connect(self._handle_voice_prompt)
        self.llm_worker.sentence_ready.connect(self.tts_worker.add_to_queue)

        # 5. Library View Routing
        # 🔑 UPDATED NAME HERE
        w.library_view.ingest_requested.connect(self._start_ingestion)

        # 6. Telemetry View Routing
        if hasattr(self.stt_worker, 'stt_latency'):
            self.stt_worker.stt_latency.connect(w.update_stt_latency)
            
        if hasattr(self.llm_worker, 'ttft_latency'):
            self.llm_worker.ttft_latency.connect(w.update_ttft_latency)
            
        if hasattr(self.tts_worker, 'tts_latency'):
            self.tts_worker.tts_latency.connect(w.update_tts_latency)

    def _handle_voice_prompt(self, text: str):
        """Safely bridges STT voice input to the LLM and the UI."""
        # 1. Grab the session ID from the UI, or create a fallback
        # 🔑 UPDATED NAMES HERE
        session_id = getattr(self.window.conversations_view, 'active_session_id', None)
        if not session_id:
            session_id = self.db_manager.create_session("Voice Chat")
            self.window.conversations_view.active_session_id = session_id
            self.window.conversations_view._refresh_history_list()

        # 2. Show what you just said in the UI (so you know the STT heard you correctly!)
        self.window.conversations_view.log_user_message(text)

        # 3. Send it to the LLM
        self.llm_worker.generate_response(text, session_id)
    
    def _sync_databases(self):
        """
        Self-healing mechanism: Scans LanceDB for embeddings and ensures 
        they are registered in the SQLite UI library.
        """
        logger.info("Running pre-flight database synchronization...")
        
        # Get what actually exists in the vector store
        lancedb_sources = self.store.get_all_indexed_sources()
        
        # Get what the UI thinks exists
        sqlite_docs = [doc['filename'] for doc in self.db_manager.get_library_documents()]
        
        # Calculate what is missing from the UI
        missing_from_ui = set(lancedb_sources) - set(sqlite_docs)
        
        if missing_from_ui:
            logger.warning(f"Found {len(missing_from_ui)} ghost files in LanceDB. Healing UI registry...")
            for source in missing_from_ui:
                # Add a dummy record to SQLite so the UI can see it and delete it if needed
                self.db_manager.add_document_metadata(source, file_size_kb=0, chunk_count=0)
                
            logger.info("Database synchronization complete.")

    def _start_ingestion(self, file_paths: list):
        """Spawns a background thread to safely embed documents without freezing the UI."""
        self.window.update_status("Ingesting Documents...")
        
        # Instantiate the worker with the required dependencies
        self.ingestion_worker = IngestionWorker(
            file_paths, 
            self.embedder, 
            self.store, 
            self.db_manager
        )
        
        # Wire the worker's progress signals back to the Library UI
        # 🔑 UPDATED NAMES HERE
        self.ingestion_worker.progress_update.connect(self.window.library_view.update_ingestion_progress)
        self.ingestion_worker.file_done.connect(self.window.update_status)
        self.ingestion_worker.ingestion_complete.connect(self.window.library_view.complete_ingestion)
        
        # Route backend errors directly to the UI popup
        self.ingestion_worker.error_occurred.connect(self.window.library_view.show_error)
        
        # Keep the terminal log as a backup
        self.ingestion_worker.error_occurred.connect(lambda err: logger.error(f"Ingestion Error: {err}"))
        
        # Fire it up!
        self.ingestion_worker.start()

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def show(self) -> None:
        self.window.show()


if __name__ == "__main__":
    # Optional: The Windows Taskbar App ID fix we discussed
    if sys.platform == 'win32':
        myappid = 'dagaza.qube.app.1.0'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    # 1. PyQt6 high DPI handling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("assets/qube_logo_256.png"))
    # 2. 🔑 THE PRESTIGE FONT LOADER
    font_files = [
        "assets/fonts/Inter-Regular.ttf",
        "assets/fonts/Inter-Italic.ttf",
        "assets/fonts/Inter-Medium.ttf",
        "assets/fonts/Inter-MediumItalic.ttf",
        "assets/fonts/Inter-SemiBold.ttf",
        "assets/fonts/Inter-SemiBoldItalic.ttf",
        "assets/fonts/Inter-Bold.ttf",
        "assets/fonts/Inter-BoldItalic.ttf"
    ]
    
    font_family = None
    for font_file in font_files:
        font_id = QFontDatabase.addApplicationFont(font_file)
        if font_id != -1 and font_family is None:
            font_family = QFontDatabase.applicationFontFamilies(font_id)[0]

    # Apply the Inter font globally if successfully loaded
    if font_family:
        app.setFont(QFont(font_family, 10))
    else:
        # 🔑 THE FIX: Fallback to Segoe UI ONLY if Inter fails to load
        logger.warning("Custom Inter font failed to load. Falling back to Segoe UI.")
        app_font = QFont("Segoe UI", 10) 
        app_font.setStyleHint(QFont.StyleHint.SansSerif)
        app.setFont(app_font)

    # 3. Load the Global Structural Stylesheet
    # This interprets the ObjectNames and Classes we just added to the views.
    style_path = os.path.join("assets", "styles", "base.qss")
    if os.path.exists(style_path):
        with open(style_path, "r") as f:
            custom_style = f.read()
            # We append our structure to the qt_material base styles
            app.setStyleSheet(app.styleSheet() + custom_style)
        logger.info(f"Custom structural stylesheet loaded from {style_path}")
    else:
        logger.warning(f"Structural stylesheet NOT found at {style_path}. UI may look unorganized.")

    # 4. Boot the Qube Assistant
    qube = Qube()
    qube.show()
    sys.exit(app.exec())