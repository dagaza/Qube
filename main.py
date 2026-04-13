import sys
import os
os.environ["QUBE_LLM_DEBUG"] = "1"

from PyQt6 import QtCore
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont, QFontDatabase, QIcon

from core.richtext_styles import apply_app_link_palette

from workers import AudioListenerWorker, STTWorker, LLMWorker, TTSWorker
from workers.native_llama_engine import NativeLlamaEngine
from workers.ingestion_worker import IngestionWorker 
from core.gpu_monitor import GPUMonitor
from rag.embedder import EmbeddingModel
from rag.store import DocumentStore
from ui.main_window import MainWindow
from core.database import DatabaseManager
from core.app_settings import (
    get_enable_memory_enrichment,
    get_engine_mode,
    get_auto_load_last_model_on_startup,
    get_internal_model_path,
)
from workers.enrichment_worker import EnrichmentWorker
from workers.internet_worker import InternetWorker

import logging

from core.logging_bootstrap import init_llm_debug_logging
from core.tooltip_wrap import install_wrapping_tooltips

# --- QUBE TERMINAL LOGGER SETUP ---
logging.basicConfig(
    level=logging.DEBUG,  # Set to INFO in production to hide the noise
    format='%(asctime)s.%(msecs)03d | %(levelname)-8s | [%(name)s] %(message)s',
    datefmt='%H:%M:%S'
)

# LLM introspection (Qube.NativeLLM.Debug) -> logs/llm_debug.log only; not the terminal
init_llm_debug_logging()

# Create the main app logger
logger = logging.getLogger("Qube.Core")
logger.info("Terminal logging initialized. Booting sequence started.")

class Qube:
    def __init__(self):
        # -- 1. Shared services ------------------------------------------
        self.embedder = EmbeddingModel()
        self.store    = DocumentStore()
        self.db_manager = DatabaseManager()
        
        # -- 2. Background workers ---------------------------------------
        self.audio_worker = AudioListenerWorker()
        self.stt_worker   = STTWorker()
        self.native_llama_engine = NativeLlamaEngine()
        self.native_llama_engine.start()

        self.llm_worker = LLMWorker(
            self.embedder,
            self.store,
            self.db_manager,
            native_engine=self.native_llama_engine,
        )
        self.tts_worker   = TTSWorker()
        self.gpu_monitor  = GPUMonitor()

        self.active_internet_worker = None

        # --- 3. Instantiate the Async Brain (Memory v3) -----------------
        # 🔑 FIX: Corrected variable names to match self.store and self.db_manager
        self.enrichment_worker = EnrichmentWorker(
            llm=self.llm_worker,
            embedder=self.embedder,
            store=self.store,
            db=self.db_manager
        )
        self.enrichment_worker.set_enabled(get_enable_memory_enrichment())

        # --- 4. THE GOLDEN WIRE (Signal -> Slot) ------------------------
        # response_finished wiring lives in _connect_signals (needs MainWindow + TTS sentinel).

        # Start the memory worker thread
        self.enrichment_worker.start()

        workers = {
            "audio": self.audio_worker,
            "stt":   self.stt_worker,
            "llm":   self.llm_worker,
            "tts":   self.tts_worker,
            "db": self.db_manager,
            "store": self.store,
            "native_engine": self.native_llama_engine,
        }

        # -- 5. UI -------------------------------------------------------
        self.window = MainWindow(
            workers=workers,
            gpu_monitor=self.gpu_monitor,
            native_engine=self.native_llama_engine,
        )

        # -- 6. Wire signals ---------------------------------------------
        self._connect_signals()
        self._sync_databases()

        if (
            get_engine_mode() == "internal"
            and get_auto_load_last_model_on_startup()
            and bool(get_internal_model_path())
        ):
            self.llm_worker.refresh_native_model_from_settings()

        # -- 7. Start continuous processes --------------------------------
        self.audio_worker.start()

        # Load the TTS Kokoro Model
        tts_path = os.path.join("models", "tts", "kokoro-v1.0.onnx")
        self.tts_worker.load_voice(tts_path)

    # ------------------------------------------------------------------ #
    #  Signal wiring                                                       #
    # ------------------------------------------------------------------ #

    def _connect_signals(self):
        w = self.window
        
        # Global Shell Routing
        self.audio_worker.status_update.connect(w.update_status)
        self.stt_worker.status_update.connect(w.update_status)
        self.llm_worker.status_update.connect(w.update_status)
        self.native_llama_engine.status_update.connect(w.update_status)
        self.tts_worker.status_update.connect(w.update_status)
        self.llm_worker.context_retrieved.connect(w.update_rag_indicator)
        self.tts_worker.playback_finished.connect(self._handle_tts_finished)
        self.tts_worker.playback_started.connect(w.conversations_view.on_tts_playback_started)
        self.tts_worker.playback_finished.connect(w.conversations_view.on_tts_playback_finished)

        # Settings View Routing
        self.tts_worker.model_loaded.connect(self.window.update_global_voice_dropdown)
        if hasattr(self.window, 'settings_view') and hasattr(self.window.settings_view, 'rag_toggle'):
            self.window.settings_view.rag_toggle.toggled.connect(self.on_rag_toggle_changed)
        if hasattr(self.window, 'settings_view') and hasattr(self.window.settings_view, 'memory_enrichment_changed'):
            self.window.settings_view.memory_enrichment_changed.connect(self.enrichment_worker.set_enabled)
        if hasattr(self.window, 'settings_view') and hasattr(self.window.settings_view, 'engine_mode_changed'):
            self.window.settings_view.engine_mode_changed.connect(self._on_engine_mode_changed)
        self.native_llama_engine.load_finished.connect(self._on_native_model_load_finished)
        if (
            hasattr(self.window, "model_manager_view")
            and hasattr(self.window, "settings_view")
            and hasattr(self.window.model_manager_view, "native_library_changed")
            and hasattr(self.window.settings_view, "refresh_native_local_library")
        ):
            self.window.model_manager_view.native_library_changed.connect(
                self.window.settings_view.refresh_native_local_library
            )

        # Conversations View Routing
        self.llm_worker.token_streamed.connect(w.conversations_view.log_agent_token)
        self.llm_worker.sources_found.connect(w.conversations_view.on_sources_found)
        # 🔑 THE FIXES: Send the live status to the text box, and unlock it when finished!
        self.llm_worker.status_update.connect(w.conversations_view.update_action_placeholder)
        self.llm_worker.response_finished.connect(self._on_llm_response_finished)
        w.conversations_view.set_stop_requested_callback(self.stop_active_response)

        # Background Data Pipeline
        self.audio_worker.audio_captured.connect(self.stt_worker.process_audio)
        self.audio_worker.wakeword_detected.connect(self._handle_user_interruption)
        self.stt_worker.transcription_ready.connect(self._handle_voice_prompt)
        
        # 🔑 UI BRIDGE: Ensure the session_id is passed from the LLM to the TTS
        self.llm_worker.sentence_ready.connect(self.tts_worker.add_to_queue)

        # Library View Routing
        w.library_view.ingest_requested.connect(self._start_ingestion)

        # Telemetry View Routing
        if hasattr(self.stt_worker, 'stt_latency'):
            self.stt_worker.stt_latency.connect(w.update_stt_latency)
        if hasattr(self.llm_worker, 'ttft_latency'):
            self.llm_worker.ttft_latency.connect(w.update_ttft_latency)
        if hasattr(self.tts_worker, 'tts_latency'):
            self.tts_worker.tts_latency.connect(w.update_tts_latency)
        if hasattr(self.llm_worker, 'router_telemetry_updated') and hasattr(w, 'telemetry_view'):
            if hasattr(w.telemetry_view, 'update_router_telemetry'):
                self.llm_worker.router_telemetry_updated.connect(w.telemetry_view.update_router_telemetry)

    def _on_llm_response_finished(self, session_id: str, text: str) -> None:
        """Unlock chat, queue memory extraction, and mark end of LLM turn for TTS (sentinel)."""
        logger.info(
            "[Main] LLM turn finished (session_id=%s, chars=%d).",
            session_id,
            len(text or ""),
        )
        if hasattr(self, 'window') and hasattr(self.window, 'conversations_view'):
            self.window.conversations_view.on_llm_response_finished()
        if hasattr(self, 'enrichment_worker'):
            self.enrichment_worker.enqueue(session_id)
        if hasattr(self, 'tts_worker'):
            self.tts_worker.enqueue_turn_complete(session_id)

    def _handle_voice_prompt(self, text: str):
        session_id = getattr(self.window.conversations_view, 'active_session_id', None)
        if not session_id:
            session_id = self.db_manager.create_session("Voice Chat")
            self.window.conversations_view.active_session_id = session_id
            self.window.conversations_view._refresh_history_list()

        # 🔑 FIX: Lock the UI while processing a voice command
        self.window.conversations_view.set_input_enabled(False)

        self.window.conversations_view.log_user_message(text)
        self.llm_worker.generate_response(text, session_id)

    def _handle_user_interruption(self):
        logger = logging.getLogger("Qube.Main")
        logger.info("User interruption detected! Slamming on the brakes.")
        
        if hasattr(self, 'llm_worker') and self.llm_worker.isRunning():
            self.llm_worker.cancel_generation()
        if hasattr(self, 'tts_worker') and self.tts_worker.isRunning():
            self.tts_worker.stop_playback()
            
        if hasattr(self, 'window'):
            self.window.update_status("LISTENING...")
            if hasattr(self.window.conversations_view, "on_generation_stopped"):
                self.window.conversations_view.on_generation_stopped()
            else:
                # Backward-safe fallback
                self.window.conversations_view.set_input_enabled(True)
                if hasattr(self.window.conversations_view, "clear_stale_agent_pointer"):
                    self.window.conversations_view.clear_stale_agent_pointer()

        logger.debug("Deaf window closed. Ready to accept new voice commands.")

    def stop_active_response(self):
        """Manual UI stop: immediately cancel LLM + TTS and unlock text input."""
        logger.info("[Main] Manual Stop requested from chat UI.")
        if hasattr(self, 'llm_worker') and self.llm_worker.isRunning():
            self.llm_worker.cancel_generation()
        if hasattr(self, 'tts_worker') and self.tts_worker.isRunning():
            self.tts_worker.stop_playback()
        if hasattr(self, 'window') and hasattr(self.window, 'conversations_view'):
            self.window.conversations_view.on_generation_stopped()
        if hasattr(self, 'window'):
            self.window.update_status("Idle")
    
    def _handle_tts_finished(self):
        """Safely resets the UI state based on the current microphone status."""
        if hasattr(self, 'window'):
            # 1. Determine the correct safe state
            if getattr(self.audio_worker, 'is_paused', False):
                safe_status = "Voice Input Deactivated"
            else:
                safe_status = "Idle"
                
            # 2. Update the internal window state
            self.window.update_status(safe_status)
            
            # 3. 🔑 THE FIX: Forcefully broadcast the safe status through the worker's 
            # signal pipeline so the Top Bar and Input Box catch the update!
            if hasattr(self, 'tts_worker'):
                self.tts_worker.status_update.emit(safe_status)

    def _handle_internet_search(self, query: str):
        """Spawns the async internet worker and connects it to the UI."""
        # Update UI status to show we are searching
        self.window.update_status("Searching the Web...")
        
        # Kill the old one if it's somehow still running
        if self.active_internet_worker and self.active_internet_worker.isRunning():
            self.active_internet_worker.stop()
            self.active_internet_worker.wait()
        
        # Instantiate the worker
        self.active_internet_worker = InternetWorker(query)
        
        # 1. Connect Result: Send the text to the chat window
        self.active_internet_worker.search_result.connect(
            lambda res: self.window.conversations_view.log_agent_token(f"\n\n**Web Search Results:**\n{res}")
        )
        
        # 2. Connect Error: Show a warning if the search fails
        self.active_internet_worker.search_error.connect(
            lambda err: logger.error(f"Web Search Failed: {err}")
        )
        
        # 3. Clean up: Reset status when finished
        self.active_internet_worker.finished.connect(lambda: self.window.update_status("Idle"))
        
        # Start the thread
        self.active_internet_worker.start()

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
    #  UI State Handlers                                                   #
    # ------------------------------------------------------------------ #

    def _on_engine_mode_changed(self, mode: str) -> None:
        """Switch between localhost OpenAI server and in-process llama.cpp."""
        if hasattr(self, "llm_worker"):
            self.llm_worker.set_engine_mode(str(mode))
            if (
                str(mode).lower().strip() == "internal"
                and get_auto_load_last_model_on_startup()
                and bool(get_internal_model_path())
            ):
                self.llm_worker.refresh_native_model_from_settings()
        self._refresh_conversations_think_toggle()

    def _on_native_model_load_finished(self, ok: bool, message: str) -> None:
        """Update Think toggle when internal GGUF load completes."""
        self._refresh_conversations_think_toggle()

    def _refresh_conversations_think_toggle(self) -> None:
        cv = getattr(getattr(self, "window", None), "conversations_view", None)
        if cv is not None and hasattr(cv, "refresh_think_toggle"):
            cv.refresh_think_toggle()

    def on_rag_toggle_changed(self, is_enabled: bool):
        """Updates the LLM worker when the user flips the RAG switch."""
        if hasattr(self, 'llm_worker'):
            self.llm_worker.mcp_rag_enabled = is_enabled
            logger.debug(f"RAG Engine manually set to: {is_enabled}")

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def show(self) -> None:
        self.window.show()

    def _graceful_shutdown(self):
        """Called automatically when the application is closing."""
        logger.info("Initiating graceful shutdown...")

        # 0. Model Manager — Hub search/README/list/download QThreads can block exit if still running
        if hasattr(self.window, "model_manager_view"):
            self.window.model_manager_view.shutdown_hf_workers()

        # 1. Stop transient workers (Internet & Ingestion)
        if self.active_internet_worker and self.active_internet_worker.isRunning():
            self.active_internet_worker.stop()
            self.active_internet_worker.wait(2000) # Wait up to 2 seconds for it to close safely
            
        if hasattr(self, 'ingestion_worker') and self.ingestion_worker.isRunning():
            self.ingestion_worker.stop()
            self.ingestion_worker.wait(2000)

        # 2. Stop the core background loop (Enrichment/Memory)
        if hasattr(self, 'enrichment_worker') and self.enrichment_worker.isRunning():
            self.enrichment_worker.stop()
            self.enrichment_worker.wait(2000)

        if hasattr(self, 'tts_worker'):
            # Cut any in-flight audio first, then request cooperative thread exit.
            self.tts_worker.stop_playback()
            self.tts_worker.request_graceful_stop()
            tts_exited = self.tts_worker.wait(2000)
            if not tts_exited:
                logger.warning("[Shutdown] TTS worker did not exit within timeout.")
                # One more cooperative nudge before giving up on native handle teardown.
                self.tts_worker.request_graceful_stop()
                tts_exited = self.tts_worker.wait(3000)
                if not tts_exited:
                    logger.error("[Shutdown] TTS worker still active; skipping audio handle close to avoid crash.")
            else:
                logger.info("[Shutdown] TTS worker exited cleanly.")
            if tts_exited and hasattr(self.tts_worker, "close_audio_resources"):
                self.tts_worker.close_audio_resources()

        if hasattr(self, "native_llama_engine"):
            self.native_llama_engine.stop_engine()

        # 3. Stop all core hardware/LLM workers
        for name, worker in self.window.workers.items():
            # 🔑 THE FIX: Ask if the object is a thread before asking if it's running!
            if hasattr(worker, 'isRunning') and worker.isRunning():
                logger.debug(f"Stopping {name} worker...")
                if hasattr(worker, 'stop'):
                    worker.stop() 
                elif hasattr(worker, 'cancel_generation'):
                    worker.cancel_generation() 

                # Only ask Qt event-loop threads to quit; custom while-loop workers stop via flags.
                if hasattr(worker, "quit") and name not in ("audio", "tts", "native_engine"):
                    worker.quit()
                worker.wait(2000) 
            
            # 🔑 BONUS: Safely close database connections if they exist
            elif hasattr(worker, 'close'):
                logger.debug(f"Closing {name} connection...")
                worker.close()

        logger.info("All threads safely terminated. Goodbye!")


if __name__ == "__main__":
    # Optional: The Windows Taskbar App ID fix we discussed
    if sys.platform == "win32":
        import ctypes

        myappid = "dagaza.qube.app.1.0"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    # 1. PyQt6 high DPI handling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    install_wrapping_tooltips(300)
    app.setWindowIcon(QIcon("assets/qube_logo_256.png"))
    apply_app_link_palette(app)
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
    app.aboutToQuit.connect(qube._graceful_shutdown)
    qube.show()
    sys.exit(app.exec())


    