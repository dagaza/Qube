import sys
import os
from collections.abc import Callable
from pathlib import Path
os.environ["QUBE_LLM_DEBUG"] = "1"

from core.__version__ import __version__

from PyQt6 import QtCore
from core.qube_tooltip import QubeApplication, qube_tooltip_set_theme
from PyQt6.QtGui import QFont, QFontDatabase, QIcon

from core.richtext_styles import apply_app_link_palette

from workers import AudioListenerWorker, STTWorker, LLMWorker, TTSWorker
from workers.native_llama_engine import NativeLlamaEngine
from workers.sidecar_llm_worker import SidecarLlmWorker
from core.sidecar_llm import SidecarLlmClient
from workers.ingestion_worker import IngestionWorker 
from core.gpu_monitor import GPUMonitor
from rag.embedder import EmbeddingModel
from rag.store import DocumentStore
from ui.main_window import MainWindow
from ui.splash_overlay import bootstrap_with_splash, start_phased_qube_build
from core.database import DatabaseManager
from core.app_settings import (
    ensure_engine_mode_initialized,
    get_enable_memory_enrichment,
    get_enable_memory_promotion,
    get_enable_memory_consolidation,
    get_enable_memory_v7_salvage,
    get_engine_mode,
    get_auto_load_last_model_on_startup,
    get_internal_model_path,
    get_audio_input_device_index,
    get_audio_output_device_index,
    get_notifications_show_preview,
    KEY_AUDIO_INPUT_DEVICE,
    KEY_AUDIO_OUTPUT_DEVICE,
    KEY_ENGINE_MODE,
    KEY_MEMORY_ENRICHMENT,
    KEY_NATIVE_CHAT_FORMAT,
    KEY_NATIVE_CPU_THREADS,
    KEY_NATIVE_GPU_LAYERS,
    KEY_NATIVE_MODEL_PATH,
    KEY_WAKEWORD_ACTIVE_ID,
    KEY_WAKEWORD_THRESHOLDS,
)
from core.notification_types import (
    enrichment_complete_event,
    ingestion_complete_event,
    stt_failed_event,
    turn_complete_event,
)
from workers.enrichment_worker import EnrichmentWorker
from workers.memory_reflection_worker import MemoryReflectionWorker
from workers.memory_promotion_worker import MemoryPromotionWorker
from workers.memory_consolidation_worker import MemoryConsolidationWorker
from workers.internet_worker import InternetWorker

import logging

from core.logging_bootstrap import init_llm_debug_logging, init_routing_debug_logging
from core.boot_args import parse_boot_args
from core.paths import install_root, resource_path

# --- QUBE TERMINAL LOGGER SETUP ---
logging.basicConfig(
    level=logging.DEBUG,  # Set to INFO in production to hide the noise
    format='%(asctime)s.%(msecs)03d | %(levelname)-8s | [%(name)s] %(message)s',
    datefmt='%H:%M:%S'
)

# LLM introspection (Qube.NativeLLM.Debug) -> logs/llm_debug.log only; not the terminal
init_llm_debug_logging()
# Routing explainability (Qube.RoutingDebug) -> logs/routing_debug.log only; not the terminal
init_routing_debug_logging()

# Create the main app logger
logger = logging.getLogger("Qube.Core")
logger.info("Terminal logging initialized. Booting sequence started.")

class Qube:
    def __init__(
        self,
        enable_routing_debug_tool: bool = False,
        *,
        embedder: EmbeddingModel | None = None,
        startup_tick: Callable[[str], None] | None = None,
    ):
        tick = startup_tick or (lambda _msg: None)
        self._boot_storage(tick, embedder)  # startup_tick optional; splash uses a fixed label
        self._boot_core_workers(tick)
        self._boot_memory_workers(tick)
        self._boot_main_window(tick, enable_routing_debug_tool)
        self._boot_connect_and_sync(tick)
        self._boot_autoload_model(tick)
        self._boot_runtime(tick)

    def _boot_storage(
        self,
        tick: Callable[[str], None],
        embedder: EmbeddingModel | None,
    ) -> None:
        if embedder is not None:
            self.embedder = embedder
        else:
            tick("Loading embeddings…")
            self.embedder = EmbeddingModel()
        tick("Preparing storage…")
        self.store = DocumentStore()
        self.db_manager = DatabaseManager()

    def _boot_core_workers(self, tick: Callable[[str], None]) -> None:
        tick("Starting core services…")
        self.audio_worker = AudioListenerWorker()
        self.stt_worker = STTWorker()
        self.native_llama_engine = NativeLlamaEngine()
        self.native_llama_engine.start()

        self.sidecar_worker = SidecarLlmWorker(self.db_manager)
        self.sidecar_worker.start()
        self.sidecar_client = SidecarLlmClient(self.sidecar_worker)

        self.llm_worker = LLMWorker(
            self.embedder,
            self.store,
            self.db_manager,
            native_engine=self.native_llama_engine,
            sidecar_client=self.sidecar_client,
        )
        self.tts_worker = TTSWorker()
        self.gpu_monitor = GPUMonitor()
        self.active_internet_worker = None

    def _boot_memory_workers(self, tick: Callable[[str], None]) -> None:
        tick("Starting memory services…")
        self.enrichment_worker = EnrichmentWorker(
            extraction_llm=self.llm_worker,
            cognition_llm=self.sidecar_client,
            embedder=self.embedder,
            store=self.store,
            db=self.db_manager,
        )
        self.enrichment_worker.set_enabled(get_enable_memory_enrichment())
        self.enrichment_worker.start()

        self.memory_reflection_worker = MemoryReflectionWorker(
            llm=self.sidecar_client,
            store=self.store,
        )
        self.memory_reflection_worker.set_enabled(get_enable_memory_enrichment())
        self.memory_reflection_worker.start()

        self.memory_promotion_worker = MemoryPromotionWorker(store=self.store)
        self.memory_promotion_worker.set_enabled(
            get_enable_memory_enrichment() and get_enable_memory_promotion()
        )
        self.memory_promotion_worker.start()

        self.memory_consolidation_worker = MemoryConsolidationWorker(store=self.store)
        self.memory_consolidation_worker.set_enabled(get_enable_memory_consolidation())
        self.memory_consolidation_worker.start()

    def _workers_for_main_window(self) -> dict:
        return {
            "audio": self.audio_worker,
            "stt": self.stt_worker,
            "llm": self.llm_worker,
            "tts": self.tts_worker,
            "db": self.db_manager,
            "store": self.store,
            "embedder": self.embedder,
            "native_engine": self.native_llama_engine,
            "sidecar": self.sidecar_client,
            "sidecar_worker": self.sidecar_worker,
        }

    def _boot_main_window(
        self,
        tick: Callable[[str], None],
        enable_routing_debug_tool: bool,
    ) -> None:
        tick("Building interface…")
        self.window = MainWindow(
            workers=self._workers_for_main_window(),
            gpu_monitor=self.gpu_monitor,
            native_engine=self.native_llama_engine,
            enable_routing_debug_tool=enable_routing_debug_tool,
        )

    def _boot_connect_and_sync(self, tick: Callable[[str], None]) -> None:
        tick("Connecting services…")
        self._connect_signals()
        self._wire_notification_adapters()
        self.sidecar_worker.ingest_blurb_ready.connect(self._on_ingest_blurb_ready)
        self._sync_databases()

    def _boot_autoload_model(self, tick: Callable[[str], None]) -> None:
        if (
            get_engine_mode() == "internal"
            and get_auto_load_last_model_on_startup()
            and bool(get_internal_model_path())
        ):
            tick("Loading language model…")
            self.llm_worker.refresh_native_model_from_settings()

    def _boot_runtime(self, tick: Callable[[str], None]) -> None:
        tick("Starting audio and voice…")
        self.audio_worker.start()
        tts_path = os.path.join("models", "tts", "kokoro-v1.0.onnx")
        self.tts_worker.load_voice(tts_path)
        tick("Ready")
        self._pending_enrichment_context = {}
        self._pending_turn_session_id: str | None = None

    # ------------------------------------------------------------------ #
    #  Signal wiring                                                       #
    # ------------------------------------------------------------------ #

    def _wire_notification_adapters(self) -> None:
        """Translate worker lifecycle events into NotificationService emits."""
        if hasattr(self.enrichment_worker, "extraction_finished"):
            self.enrichment_worker.extraction_finished.connect(self._on_enrichment_finished)

    def _on_enrichment_finished(self, session_id: str, facts_stored: int) -> None:
        self.window.emit_notification(
            enrichment_complete_event(session_id=session_id, facts_stored=facts_stored)
        )

    def _notify_turn_complete_if_hidden(self, session_id: str, final_text: str) -> None:
        preview = ""
        if get_notifications_show_preview() and final_text:
            preview = final_text.strip()[:120]
        event = turn_complete_event(session_id=session_id, preview=preview)
        tts_enabled = bool(
            getattr(self.tts_worker, "is_muted", False) is False
            and getattr(self.window, "voice_bypass_toggle", None)
            and self.window.voice_bypass_toggle.isChecked()
        )
        if tts_enabled:
            self.window.notification_service.queue_turn_complete(event, wait_for_tts=True)
        else:
            self.window.notification_service.emit(event)

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
        self.tts_worker.turn_settled.connect(w.conversations_view.on_tts_turn_settled)

        # Settings View Routing
        self.tts_worker.model_loaded.connect(self.window.update_global_voice_dropdown)
        if hasattr(self.window, 'settings_view') and hasattr(self.window.settings_view, 'rag_toggle'):
            self.window.settings_view.rag_toggle.toggled.connect(self.on_rag_toggle_changed)
        if hasattr(self.window, 'settings_view') and hasattr(self.window.settings_view, 'memory_enrichment_changed'):
            self.window.settings_view.memory_enrichment_changed.connect(self.enrichment_worker.set_enabled)
            self.window.settings_view.memory_enrichment_changed.connect(
                self.memory_reflection_worker.set_enabled
            )
        if hasattr(self.window, 'settings_view') and hasattr(self.window.settings_view, 'memory_promotion_changed'):
            self.window.settings_view.memory_promotion_changed.connect(
                self.memory_promotion_worker.set_enabled
            )
        if hasattr(self.window, 'settings_view') and hasattr(self.window.settings_view, 'memory_consolidation_changed'):
            self.window.settings_view.memory_consolidation_changed.connect(
                self.memory_consolidation_worker.set_enabled
            )
        if hasattr(self.window, 'settings_view') and hasattr(self.window.settings_view, 'engine_mode_changed'):
            self.window.settings_view.engine_mode_changed.connect(self._on_engine_mode_changed)
        if hasattr(self.window, 'settings_view') and hasattr(
            self.window.settings_view, 'external_settings_reloaded'
        ):
            self.window.settings_view.external_settings_reloaded.connect(
                self._on_external_settings_reloaded
            )
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
        self.llm_worker.token_streamed.connect(w.conversations_view.on_llm_token_streamed)
        self.llm_worker.sources_found.connect(w.conversations_view.on_sources_found)
        # 🔑 THE FIXES: Send the live status to the text box, and unlock it when finished!
        self.llm_worker.status_update.connect(w.conversations_view.update_action_placeholder)
        self.llm_worker.response_finished.connect(self._on_llm_response_finished)
        # Phase B memory enrichment: per-turn rich context (rag chunk ids +
        # message ids) is emitted just before response_finished. Capture it
        # on self and hand it to the enrichment worker in
        # _on_llm_response_finished so provenance is exact.
        self.llm_worker.enrichment_context_ready.connect(self._on_enrichment_context_ready)
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
        if hasattr(self.tts_worker, 'playback_level'):
            self.tts_worker.playback_level.connect(w.on_tts_playback_level)
        if hasattr(self.audio_worker, 'volume_update'):
            self.audio_worker.volume_update.connect(w.on_audio_volume_update)
        if hasattr(self.llm_worker, 'router_telemetry_updated') and hasattr(w, 'telemetry_view'):
            if hasattr(w.telemetry_view, 'update_router_telemetry'):
                self.llm_worker.router_telemetry_updated.connect(w.telemetry_view.update_router_telemetry)
        if hasattr(self, 'sidecar_worker') and hasattr(w, 'telemetry_view'):
            if hasattr(self.sidecar_worker, 'sidecar_telemetry_updated'):
                if hasattr(w.telemetry_view, 'update_sidecar_telemetry'):
                    self.sidecar_worker.sidecar_telemetry_updated.connect(
                        w.telemetry_view.update_sidecar_telemetry
                    )
            if hasattr(self.sidecar_worker, 'model_reload_finished'):
                tv = w.telemetry_view
                if hasattr(tv, '_refresh_sidecar_from_worker_snapshot'):
                    self.sidecar_worker.model_reload_finished.connect(
                        lambda _ok, _msg: tv._refresh_sidecar_from_worker_snapshot()
                    )
                if hasattr(w, 'settings_view') and hasattr(
                    w.settings_view, '_sync_active_cognition_label'
                ):
                    self.sidecar_worker.model_reload_finished.connect(
                        lambda _ok, _msg: w.settings_view._sync_active_cognition_label()
                    )
        if hasattr(self.llm_worker, 'sidecar_telemetry_updated') and hasattr(w, 'telemetry_view'):
            if hasattr(w.telemetry_view, 'update_sidecar_telemetry'):
                self.llm_worker.sidecar_telemetry_updated.connect(
                    w.telemetry_view.update_sidecar_telemetry
                )
        if hasattr(self.llm_worker, 'routing_debug_record_added') and hasattr(w, 'routing_debug_tool_view'):
            if w.routing_debug_tool_view is not None:
                self.llm_worker.routing_debug_record_added.connect(w.routing_debug_tool_view.add_record)

    def _on_enrichment_context_ready(self, payload: dict) -> None:
        """Cache the turn-scoped enrichment context emitted by LLMWorker.

        Stored here so ``_on_llm_response_finished`` can pass it to
        ``EnrichmentWorker.enqueue`` along with (or instead of) a bare
        session id. Signals fire on the main thread via Qt queued
        connections so a plain attribute assignment is safe.
        """
        self._pending_enrichment_context = payload or {}

    def _on_llm_response_finished(self, session_id: str, text: str) -> None:
        """Unlock chat, queue memory extraction, and mark end of LLM turn for TTS (sentinel)."""
        logger.info(
            "[Main] LLM turn finished (session_id=%s, chars=%d).",
            session_id,
            len(text or ""),
        )
        if hasattr(self, "window") and hasattr(self.window, "conversations_view"):
            self.window.conversations_view.on_llm_response_finished(session_id, text or "")
        self._pending_turn_session_id = session_id
        if hasattr(self, 'enrichment_worker') and get_enable_memory_enrichment():
            ctx = getattr(self, "_pending_enrichment_context", None) or {}
            if ctx:
                payload = dict(ctx)
                payload["session_id"] = session_id
                self.enrichment_worker.enqueue(payload)
                salvage_ids = list(payload.get("salvage_message_ids") or [])
                if salvage_ids and get_enable_memory_v7_salvage() and not payload.get("skip_enrichment"):
                    self.enrichment_worker.enqueue(
                        {
                            "session_id": session_id,
                            "enrichment_mode": "salvage",
                            "salvage_message_ids": salvage_ids,
                            "salvage_reason": payload.get("salvage_reason") or "history_window",
                        }
                    )
            else:
                self.enrichment_worker.enqueue(session_id)
        if hasattr(self, 'enrichment_worker'):
            self._pending_enrichment_context = None
        tts_will_play = bool(
            hasattr(self, "tts_worker")
            and not getattr(self.tts_worker, "is_muted", True)
            and hasattr(self.window, "voice_bypass_toggle")
            and self.window.voice_bypass_toggle.isChecked()
        )
        if not tts_will_play:
            self._notify_turn_complete_if_hidden(session_id, text or "")
            if getattr(self.audio_worker, "is_paused", False):
                self.window.update_status("Voice Input Deactivated", force=True)
            else:
                self.window.update_status("Idle", force=True)
        if hasattr(self, 'tts_worker'):
            self.tts_worker.enqueue_turn_complete(session_id)

    def _handle_voice_prompt(self, text: str):
        cleaned = (text or "").strip()
        if not cleaned:
            self.window.emit_notification(stt_failed_event())
            self.window.update_status("Idle", force=True)
            return

        session_id = getattr(self.window.conversations_view, 'active_session_id', None)
        if not session_id:
            conv_view = self.window.conversations_view
            folder_id = getattr(conv_view, "_active_folder_id", None)
            if not folder_id:
                folder_id = self.db_manager.get_main_conversation_folder_id()
            session_id = self.db_manager.create_session("Voice Chat", folder_id=folder_id)
            conv_view.active_session_id = session_id
            conv_view._refresh_history_list()

        # 🔑 FIX: Lock the UI while processing a voice command
        self.window.conversations_view.set_input_enabled(False)

        from core.composer_attachments import parse_attachments

        self.window.conversations_view.log_user_message(text, pending_assistant=True)
        clean, attachments = parse_attachments(cleaned)
        prompt = clean if clean else cleaned
        self.llm_worker.generate_response(
            prompt,
            session_id,
            attachments=attachments,
            persist_content=cleaned.strip(),
        )

    def _handle_user_interruption(self):
        logger = logging.getLogger("Qube.Main")
        logger.info("User interruption detected! Slamming on the brakes.")

        session_id = getattr(self, "_pending_turn_session_id", None)
        if session_id:
            self.window.notification_service.cancel_turn_complete(session_id)
        
        if hasattr(self, 'llm_worker') and self.llm_worker.isRunning():
            self.llm_worker.cancel_generation()
        if hasattr(self, 'tts_worker') and getattr(self.tts_worker, 'is_playing', False):
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
        session_id = getattr(self, "_pending_turn_session_id", None)
        if session_id:
            self.window.notification_service.cancel_turn_complete(session_id)
        if hasattr(self, 'llm_worker') and self.llm_worker.isRunning():
            self.llm_worker.cancel_generation()
        if hasattr(self, 'tts_worker') and getattr(self.tts_worker, 'is_playing', False):
            self.tts_worker.stop_playback()
        if hasattr(self, 'window') and hasattr(self.window, 'conversations_view'):
            self.window.conversations_view.on_generation_stopped()
        if hasattr(self, 'window'):
            self.window.update_status("Idle", force=True)
    
    def _handle_tts_finished(self):
        """Safely resets the UI state based on the current microphone status."""
        session_id = getattr(self, "_pending_turn_session_id", None)
        if session_id:
            self.window.notification_service.flush_turn_complete(session_id)
            self._pending_turn_session_id = None
        if hasattr(self, 'window'):
            # 1. Determine the correct safe state
            if getattr(self.audio_worker, 'is_paused', False):
                safe_status = "Voice Input Deactivated"
            else:
                safe_status = "Idle"
                
            # 2. Update the internal window state
            self.window.update_status(safe_status, force=True)
            
            # 3. Forcefully broadcast the safe status through the worker's
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
                from core.library_folder_policy import is_qube_managed_document_filename

                if is_qube_managed_document_filename(source):
                    folder_id = self.db_manager.get_qube_library_folder_id()
                else:
                    folder_id = self.db_manager.get_main_library_folder_id()
                # Add a dummy record to SQLite so the UI can see it and delete it if needed
                self.db_manager.add_document_metadata(
                    source,
                    file_size_kb=0,
                    chunk_count=0,
                    folder_id=folder_id,
                )
                
            logger.info("Database synchronization complete.")

    def _start_ingestion(self, file_paths: list, folder_id: str):
        """Spawns a background thread to safely embed documents without freezing the UI."""
        self.window.update_status("Ingesting Documents...")
        self.window._activity_reducer.set_background_busy(True)
        self.window._sync_tray_presence()

        self.ingestion_worker = IngestionWorker(
            file_paths,
            self.embedder,
            self.store,
            self.db_manager,
            folder_id=folder_id,
            sidecar_worker=self.sidecar_worker,
        )

        # Wire the worker's progress signals back to the Library UI
        self.ingestion_worker.progress_update.connect(self.window.library_view.update_ingestion_progress)
        self.ingestion_worker.file_done.connect(self.window.update_status)
        self.ingestion_worker.ingestion_complete.connect(self.window.library_view.complete_ingestion)
        self.ingestion_worker.ingestion_complete.connect(self._on_ingestion_complete)

        # Route backend errors directly to the UI popup
        self.ingestion_worker.error_occurred.connect(self.window.library_view.show_error)

        # Keep the terminal log as a backup
        self.ingestion_worker.error_occurred.connect(lambda err: logger.error(f"Ingestion Error: {err}"))

        # Fire it up!
        self.ingestion_worker.start()

    def _on_ingest_blurb_ready(self, filename: str, blurb: str) -> None:
        if self.db_manager.update_document_blurb(filename, blurb):
            if hasattr(self.window, "library_view"):
                self.window.library_view.refresh_library_list()

    def _on_ingestion_complete(self, chunk_count: int) -> None:
        file_count = len(getattr(self.ingestion_worker, "file_paths", []) or [])
        if file_count <= 0 and chunk_count > 0:
            file_count = 1
        if file_count > 0:
            self.window.emit_notification(ingestion_complete_event(file_count=file_count))
        self.window._activity_reducer.set_background_busy(False)
        self.window._sync_tray_presence()

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

    def _on_external_settings_reloaded(self, changed: set) -> None:
        """Apply worker/runtime updates after settings.json was edited externally."""
        if KEY_MEMORY_ENRICHMENT in changed:
            enabled = get_enable_memory_enrichment()
            if hasattr(self, "enrichment_worker"):
                self.enrichment_worker.set_enabled(enabled)
            if hasattr(self, "memory_reflection_worker"):
                self.memory_reflection_worker.set_enabled(enabled)
        if KEY_ENGINE_MODE in changed:
            self._on_engine_mode_changed(get_engine_mode())
            return
        native_keys = {
            KEY_NATIVE_MODEL_PATH,
            KEY_NATIVE_GPU_LAYERS,
            KEY_NATIVE_CPU_THREADS,
            KEY_NATIVE_CHAT_FORMAT,
        }
        if native_keys & changed and get_engine_mode() == "internal" and hasattr(self, "llm_worker"):
            self.llm_worker.refresh_native_model_from_settings()
        if KEY_AUDIO_INPUT_DEVICE in changed and hasattr(self, "audio_worker"):
            idx = get_audio_input_device_index()
            if idx is not None:
                self.audio_worker.set_input_device(idx)
        if KEY_AUDIO_OUTPUT_DEVICE in changed and hasattr(self, "tts_worker"):
            idx = get_audio_output_device_index()
            if idx is not None:
                self.tts_worker.set_device(idx)
        if (KEY_WAKEWORD_ACTIVE_ID in changed or KEY_WAKEWORD_THRESHOLDS in changed) and hasattr(
            self, "audio_worker"
        ):
            sv = getattr(self.window, "settings_view", None)
            if sv is not None and hasattr(sv, "_sync_wakeword_catalog"):
                sv._sync_wakeword_catalog(trigger="external settings")

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

        # 0b. Memory Manager — QThread is not stopped via closeEvent when the page is embedded in the stack
        mm = getattr(self.window, "memory_manager_view", None)
        if mm is not None:
            mmw = getattr(mm, "worker", None)
            if mmw is not None and hasattr(mmw, "isRunning") and mmw.isRunning():
                if hasattr(mmw, "shutdown"):
                    mmw.shutdown()
                if not mmw.wait(5000):
                    logger.warning(
                        "[Shutdown] Memory manager worker did not exit within 5s."
                    )

        # 0c. Windows GPU polling (standard library thread, not QThread)
        if hasattr(self, "gpu_monitor") and self.gpu_monitor is not None:
            try:
                self.gpu_monitor.cleanup()
            except Exception:
                pass

        if hasattr(self.window, "_companion_controller") and self.window._companion_controller is not None:
            self.window._companion_controller.shutdown()

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

        # Phase C: stop the periodic memory self-reflection worker.
        if hasattr(self, 'memory_reflection_worker') and self.memory_reflection_worker.isRunning():
            self.memory_reflection_worker.shutdown()
            self.memory_reflection_worker.wait(2000)

        if hasattr(self, 'memory_promotion_worker') and self.memory_promotion_worker.isRunning():
            self.memory_promotion_worker.shutdown()
            self.memory_promotion_worker.wait(2000)

        if hasattr(self, 'memory_consolidation_worker') and self.memory_consolidation_worker.isRunning():
            self.memory_consolidation_worker.shutdown()
            self.memory_consolidation_worker.wait(2000)

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

        if hasattr(self, "sidecar_worker") and self.sidecar_worker.isRunning():
            self.sidecar_worker.stop_engine()
            self.sidecar_worker.wait(2000)

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

        # 4. Last-chance: QThreads that ignore quit() while run() is busy (e.g. STT transcribing)
        self._finalize_running_qthreads()

        logger.info("All threads safely terminated. Goodbye!")

    def _finalize_running_qthreads(self) -> None:
        """Wait or force-terminate Qt worker threads still running after cooperative shutdown."""
        llm = getattr(self, "llm_worker", None)
        if llm is not None and llm.isRunning():
            if hasattr(llm, "cancel_generation"):
                llm.cancel_generation()
            if not llm.wait(10_000):
                logger.warning("[Shutdown] LLM worker still running after 10s wait.")

        stt = getattr(self, "stt_worker", None)
        if stt is not None and stt.isRunning():
            stt.requestInterruption()
            if not stt.wait(10_000):
                logger.warning("[Shutdown] STT worker still running; terminating thread.")
                stt.terminate()
                stt.wait(3000)

        audio = getattr(self, "audio_worker", None)
        if audio is not None and audio.isRunning():
            if hasattr(audio, "stop"):
                audio.stop()
            if not audio.wait(8000):
                logger.warning("[Shutdown] Audio worker still running; blocking until exit.")
                audio.wait()


if __name__ == "__main__":
    args = parse_boot_args()
    # Optional: The Windows Taskbar App ID fix we discussed
    if sys.platform == "win32":
        import ctypes

        myappid = f"dagaza.qube.app.{__version__}"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    # 1. PyQt6 high DPI handling
    QubeApplication.setHighDpiScaleFactorRoundingPolicy(
        QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QubeApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    repo_root = install_root()
    window_icon_path = resource_path("assets", "logos", "qube_logo_256.png")
    if not window_icon_path.is_file():
        window_icon_path = resource_path("assets", "icons", "qube_logo_256.png")
    if not window_icon_path.is_file():
        window_icon_path = resource_path("assets", "qube_logo_256.png")
    if window_icon_path.is_file():
        app.setWindowIcon(QIcon(str(window_icon_path)))
    apply_app_link_palette(app)
    # 2. 🔑 THE PRESTIGE FONT LOADER
    font_files = [
        resource_path("assets", "fonts", name)
        for name in (
            "Inter-Regular.ttf",
            "Inter-Italic.ttf",
            "Inter-Medium.ttf",
            "Inter-MediumItalic.ttf",
            "Inter-SemiBold.ttf",
            "Inter-SemiBoldItalic.ttf",
            "Inter-Bold.ttf",
            "Inter-BoldItalic.ttf",
        )
    ]
    
    font_family = None
    for font_file in font_files:
        font_id = QFontDatabase.addApplicationFont(str(font_file))
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
    style_path = resource_path("assets", "styles", "base.qss")
    if style_path.is_file():
        with open(style_path, "r") as f:
            custom_style = f.read()
            # We append our structure to the qt_material base styles
            app.setStyleSheet(app.styleSheet() + custom_style)
        logger.info(f"Custom structural stylesheet loaded from {style_path}")
    else:
        logger.warning(f"Structural stylesheet NOT found at {style_path}. UI may look unorganized.")

    # 4. Boot the Qube Assistant (first launch defaults to Internal Engine)
    ensure_engine_mode_initialized()

    def _build_qube(
        *,
        embedder: EmbeddingModel,
        on_phase,
        on_complete,
    ):
        return start_phased_qube_build(
            embedder=embedder,
            enable_routing_debug_tool=bool(args.routing_debug),
            on_phase=on_phase,
            on_complete=on_complete,
        )

    def _on_qube_ready(qube: Qube) -> None:
        qube_tooltip_set_theme(getattr(qube.window, "_is_dark_theme", True))
        app.aboutToQuit.connect(qube._graceful_shutdown)
        qube.show()

    # Keep a strong reference; otherwise StartupSplashController is GC'd and startup timers never fire.
    app._startup_splash_controller = bootstrap_with_splash(
        repo_root=repo_root,
        build_app_fn=_build_qube,
        on_ready=_on_qube_ready,
    )
    logger.info("Entering Qt event loop.")
    sys.exit(app.exec())


    