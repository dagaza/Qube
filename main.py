import sys
import os
from collections.abc import Callable
from pathlib import Path
os.environ["QUBE_LLM_DEBUG"] = "1"
os.environ["QUBE_LOG_RAW_COMPLETION"] = "1"

from core.__version__ import __version__

from PyQt6 import QtCore
from PyQt6.QtGui import QFont, QFontDatabase
from core.qube_tooltip import QubeApplication, qube_tooltip_set_theme
from ui.app_icon import apply_linux_desktop_integration, qube_window_icon

from core.richtext_styles import apply_app_link_palette

from workers import AudioListenerWorker, STTWorker, LLMWorker, TTSWorker
from workers.native_llama_engine import NativeLlamaEngine
from workers.sidecar_llm_worker import SidecarLlmWorker
from core.sidecar_llm import SidecarLlmClient
from core.auxiliary_cognition import migrate_stale_sidecar_override
from core.embedding_models import migrate_stale_embedding_override
from core.stt_models import migrate_stale_stt_override
from core.tts_models import migrate_legacy_tts_layout, migrate_stale_tts_override, resolve_boot_tts_path
from rag.embedder import EmbeddingModel
from rag.store import DocumentStore
from ui.main_window import (
    MAIN_STAGE_LIBRARY,
    MAIN_STAGE_SETTINGS,
    MAIN_STAGE_TELEMETRY,
    MainWindow,
)
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
    KEY_MCP_INTERNET_HYBRID,
    get_mcp_internet_hybrid_enabled,
)
from core.notification_types import (
    deep_research_complete_event,
    enrichment_complete_event,
    format_retry_in_progress_event,
    ingestion_complete_event,
    output_truncated_max_tokens_event,
    stt_failed_event,
    turn_complete_event,
)
from workers.enrichment_worker import EnrichmentWorker
from workers.deep_research_worker import DeepResearchWorker
from workers.ingestion_worker import IngestionWorker
from workers.reindex_worker import ReindexWorker
from workers.memory_reflection_worker import MemoryReflectionWorker
from workers.memory_promotion_worker import MemoryPromotionWorker
from workers.memory_consolidation_worker import MemoryConsolidationWorker
from workers.internet_worker import InternetWorker
from core.gpu_monitor import GPUMonitor
from core.app_settings import get_embedding_mode, set_embedding_mode, set_embedding_model_path
from core.embedding_modes import normalize_mode_id
from core.reindex_state import is_reindex_in_progress
from core.router_centroid_install import clear_router_embedding_state, install_router_centroids

import logging

from core.logging_bootstrap import (
    init_app_logging,
    init_llm_debug_logging,
    init_routing_debug_logging,
    init_skills_debug_logging,
    init_web_search_audit_logging,
    sync_diagnostic_file_sinks_from_settings,
)
from core.boot_args import parse_boot_args
from core.paths import install_root, resource_path, configure_user_model_paths

# --- QUBE TERMINAL LOGGER SETUP ---
logging.basicConfig(
    level=logging.DEBUG,  # Set to INFO in production to hide the noise
    format='%(asctime)s.%(msecs)03d | %(levelname)-8s | [%(name)s] %(message)s',
    datefmt='%H:%M:%S'
)

# LLM introspection (Qube.NativeLLM.Debug) -> ~/.qube/logs/llm_debug.log only; not the terminal
init_llm_debug_logging()
# Routing explainability (Qube.RoutingDebug) -> logs/routing_debug.log only; not the terminal
init_routing_debug_logging()
# Skill activation telemetry (Qube.SkillsDebug) -> logs/skills_debug.log only; not the terminal
init_skills_debug_logging()
# Web search audit (Qube.WebSearchAudit) -> logs/web_search.log only; not the terminal
init_web_search_audit_logging()
# General Qube.* lifecycle logs -> ~/.qube/logs/qube.log (terminal unchanged)
init_app_logging()
sync_diagnostic_file_sinks_from_settings()

# Create the main app logger
logger = logging.getLogger("Qube.Core")
logger.info("Terminal logging initialized. Booting sequence started.")

class Qube:
    def __init__(
        self,
        enable_routing_debug_tool: bool = False,
        enable_trace_diff_debug_tool: bool = False,
        *,
        embedder: EmbeddingModel | None = None,
        startup_tick: Callable[[str], None] | None = None,
        theme_manager=None,
    ):
        tick = startup_tick or (lambda _msg: None)
        self._theme_manager = theme_manager
        self._boot_storage(tick, embedder)  # startup_tick optional; splash uses a fixed label
        self._boot_core_workers(tick)
        self._boot_memory_workers(tick)
        self._boot_main_window(tick, enable_routing_debug_tool, enable_trace_diff_debug_tool)
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
            try:
                self.embedder = EmbeddingModel()
            except Exception as exc:
                logger.warning("Embedding model unavailable at startup: %s", exc)
                self.embedder = None
        tick("Preparing storage…")
        from core.embedding_modes import DEFAULT_MODE, get_mode_spec

        expected_dim = (
            self.embedder.vector_dim
            if self.embedder is not None
            else get_mode_spec(DEFAULT_MODE).vector_dim
        )
        self.store = DocumentStore(expected_vector_dim=expected_dim)
        self.db_manager = DatabaseManager()
        self.reindex_worker = None
        self._reindex_revert_embedding_mode: str | None = None
        self._reindex_target_mode: str | None = None

    def _boot_core_workers(self, tick: Callable[[str], None]) -> None:
        from core.bootstrap_manifest import BootstrapModelId
        from core.bootstrap_missing_models import stt_model_available
        from core.bootstrap_selection import get_selected_model_ids

        tick("Starting core services…")
        self.audio_worker = AudioListenerWorker()
        stt_selected = BootstrapModelId.WHISPER_SMALL in get_selected_model_ids()
        self.stt_worker = STTWorker(eager_load=stt_selected or stt_model_available())
        self.native_llama_engine = NativeLlamaEngine()
        self.native_llama_engine.start()

        migrate_stale_sidecar_override()
        migrate_stale_embedding_override()
        migrate_stale_stt_override()
        migrate_legacy_tts_layout()
        migrate_stale_tts_override()
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

        self.deep_research_worker = DeepResearchWorker(synthesis_llm=self.llm_worker)
        self.deep_research_worker.set_enabled(True)
        self.deep_research_worker.start()

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
            "deep_research": self.deep_research_worker,
        }

    def _boot_main_window(
        self,
        tick: Callable[[str], None],
        enable_routing_debug_tool: bool,
        enable_trace_diff_debug_tool: bool,
    ) -> None:
        tick("Building interface…")
        self.window = MainWindow(
            workers=self._workers_for_main_window(),
            gpu_monitor=self.gpu_monitor,
            native_engine=self.native_llama_engine,
            enable_routing_debug_tool=enable_routing_debug_tool,
            enable_trace_diff_debug_tool=enable_trace_diff_debug_tool,
            theme_manager=self._theme_manager,
        )
        if self._theme_manager is not None:
            self._theme_manager.apply(persist=False)

    def _boot_connect_and_sync(self, tick: Callable[[str], None]) -> None:
        tick("Connecting services…")
        self._connect_signals()
        self._wire_notification_adapters()
        self.sidecar_worker.ingest_blurb_ready.connect(self._on_ingest_blurb_ready)
        self._sync_databases()
        self._maybe_seed_help_corpus()
        if getattr(self.store, "dim_mismatch", False):
            logger.warning(
                "LanceDB dimension mismatch detected; starting reindex for active mode."
            )
            self._start_reindex_for_mode(get_embedding_mode())

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
        self.tts_worker.load_voice(resolve_boot_tts_path())
        tick("Ready")
        self._pending_enrichment_context = {}
        self._pending_turn_session_id: str | None = None
        self._voice_stt_epoch = 0

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

    def _invalidate_router_embedding_state(self, *, rebuild_centroids: bool = False) -> None:
        router = getattr(getattr(self, "llm_worker", None), "cognitive_router", None)
        if router is None:
            return
        clear_router_embedding_state(router)
        if rebuild_centroids and getattr(self, "embedder", None) is not None:
            install_router_centroids(router, self.embedder, force=True)
        cache = getattr(getattr(self, "llm_worker", None), "embedding_cache", None)
        if cache is not None and hasattr(cache, "reset"):
            cache.reset()

    def _reload_embedder_from_settings(self) -> None:
        """Reload GGUF override and re-embed the library with the new model."""
        self._invalidate_router_embedding_state()
        try:
            self.embedder.reload()
        except Exception as e:
            logger.error("Embedding model reload failed: %s", e)
            return
        self._start_reindex_current_embedder()

    def _begin_embedding_job_ui(self, detail: str) -> None:
        self.window.begin_background_progress(detail)
        self.window.ensure_library_view().begin_ingest_progress_ui(detail=detail)

    def _update_embedding_job_progress(self, percent: int) -> None:
        self.window.update_background_progress(percent)
        self.window.ensure_library_view().update_ingestion_progress(percent)

    def _set_embedding_job_detail(self, detail: str) -> None:
        self.window.set_background_progress_detail(detail)
        self.window.ensure_library_view().set_ingest_progress_detail(detail)
        self.window.update_status(detail)

    def _finish_embedding_job_ui(self) -> None:
        self.window.finish_background_progress()

    def _wire_reindex_worker(self, worker: ReindexWorker) -> None:
        worker.progress_update.connect(self._update_embedding_job_progress)
        worker.status_update.connect(self._set_embedding_job_detail)
        worker.error_occurred.connect(self._on_reindex_error)
        worker.reindex_complete.connect(self._on_reindex_complete)

    def _start_reindex_worker(
        self,
        *,
        target_mode: str | None = None,
        reload_embedder: bool = True,
    ) -> None:
        if is_reindex_in_progress():
            logger.info("Reindex already in progress; ignoring duplicate request.")
            return
        if self.reindex_worker is not None and self.reindex_worker.isRunning():
            return

        self._invalidate_router_embedding_state()
        detail = "Reprocessing your library and memories…"
        self.window._activity_reducer.set_background_busy(True)
        self.window._sync_tray_presence()
        self._begin_embedding_job_ui(detail)
        self.window.update_status(detail)

        router = getattr(self.llm_worker, "cognitive_router", None)
        self.reindex_worker = ReindexWorker(
            target_mode=target_mode,
            embedder=self.embedder,
            store=self.store,
            cognitive_router=router,
            reload_embedder=reload_embedder,
        )
        self._wire_reindex_worker(self.reindex_worker)
        self.reindex_worker.start()

    def _start_reindex_current_embedder(self) -> None:
        self._start_reindex_worker(reload_embedder=False)

    def _start_reindex_for_mode(self, mode_id: str) -> None:
        target_mode = normalize_mode_id(mode_id)
        self._reindex_target_mode = target_mode
        set_embedding_mode(target_mode)
        set_embedding_model_path("")
        self._start_reindex_worker(target_mode=target_mode, reload_embedder=True)

    def _on_reindex_complete(self, mode_id: str) -> None:
        from workers.intent_router import EmbeddingCache
        from core.embedding_models import clear_embedding_availability_cache

        clear_embedding_availability_cache()
        self._reindex_revert_embedding_mode = None
        self._reindex_target_mode = None
        self.llm_worker.embedder = self.embedder
        self.llm_worker.embedding_cache = EmbeddingCache(self.embedder)
        self.enrichment_worker.embedder = self.embedder
        self.store.dim_mismatch = False
        self.store.vector_dim = self.embedder.vector_dim
        w = self.window
        if isinstance(getattr(w, "workers", None), dict):
            w.workers["embedder"] = self.embedder
        mmv = self.window._memory_manager_view
        if mmv is not None:
            mmv.embedder = self.embedder
            worker = getattr(mmv, "worker", None)
            if worker is not None:
                worker.embedder = self.embedder
        self.window.ensure_library_view().finish_reindex_ui()
        self._finish_embedding_job_ui()
        self.window._activity_reducer.set_background_busy(False)
        self.window._sync_tray_presence()
        self.window.update_status("Idle", force=True)
        logger.info("Reindex complete for mode=%s", mode_id)
        from core.embedding_modes import get_mode_spec
        from core.notification_types import NotificationEvent, NotificationSeverity

        spec = get_mode_spec(mode_id) if mode_id else None
        mode_label = spec.label if spec else "updated search quality"
        self.window.emit_notification(
            NotificationEvent(
                title="Library reprocessing complete",
                body=f"Your knowledge base is ready with {mode_label} search quality.",
                severity=NotificationSeverity.SUCCESS,
                category="background",
                auto_dismiss_ms=8000,
                tray_bump=True,
            )
        )
        sv = self.window._settings_view
        if sv is not None:
            if hasattr(sv, "_sync_embedding_mode_selector"):
                sv._sync_embedding_mode_selector()
            if hasattr(sv, "_sync_active_embedding_label"):
                sv._sync_active_embedding_label()
            if hasattr(sv, "_sync_bootstrap_download_visibility"):
                sv._sync_bootstrap_download_visibility()

    def _on_reindex_error(self, message: str) -> None:
        from core.app_settings import get_embedding_mode, set_embedding_mode
        from core.bootstrap_search_models import (
            format_search_preset_download_failure,
            is_likely_embedding_load_failure,
        )
        from core.embedding_modes import normalize_mode_id

        revert = self._reindex_revert_embedding_mode
        failed_target = self._reindex_target_mode
        self._reindex_revert_embedding_mode = None
        self._reindex_target_mode = None
        if revert is not None:
            set_embedding_mode(revert)
            sv = self.window._settings_view
            if sv is not None:
                if hasattr(sv, "_sync_embedding_mode_selector"):
                    sv._sync_embedding_mode_selector()
                if hasattr(sv, "_sync_active_embedding_label"):
                    sv._sync_active_embedding_label()
                if hasattr(sv, "_sync_bootstrap_download_visibility"):
                    sv._sync_bootstrap_download_visibility()

        body = (
            "Reprocessing failed. Your library may be incomplete — try again.\n\n"
            f"{message}"
        )
        if is_likely_embedding_load_failure(message):
            mode_for_hint = normalize_mode_id(
                failed_target or get_embedding_mode()
            )
            body += f"\n\n{format_search_preset_download_failure(mode_for_hint)}"

        self._finish_embedding_job_ui()
        self.window.ensure_library_view().show_error(
            body,
            title="Reprocessing Failed",
        )
        self.window._activity_reducer.set_background_busy(False)
        self.window._sync_tray_presence()
        self.window.update_status("Idle", force=True)

    def _on_embedding_mode_change_requested(self, mode_id: str, previous_mode: str) -> None:
        self._reindex_revert_embedding_mode = normalize_mode_id(previous_mode)
        self._start_reindex_for_mode(mode_id)

    def _reload_stt_from_settings(self) -> None:
        if hasattr(self.stt_worker, "reload_from_settings"):
            self.stt_worker.reload_from_settings()
        w = self.window
        sv = w._settings_view
        if sv is not None:
            if hasattr(sv, "_sync_active_stt_label"):
                sv._sync_active_stt_label()
            if hasattr(sv, "_refresh_stt_model_list"):
                sv._refresh_stt_model_list()

    def _reload_tts_from_settings(self) -> None:
        from core.tts_models import resolve_boot_tts_path

        ok = self.tts_worker.load_voice(resolve_boot_tts_path())
        if not ok:
            logger.warning("[TTS] Failed to reload model from settings")
        w = self.window
        sv = w._settings_view
        if sv is not None:
            if hasattr(sv, "_sync_active_tts_label"):
                sv._sync_active_tts_label()
            if hasattr(sv, "_refresh_tts_model_list"):
                sv._refresh_tts_model_list()

    def _wire_library_app_routes(self) -> None:
        lv = self.window.peek_library_view()
        if lv is None:
            return
        lv.ingest_requested.connect(self._start_ingestion)

    def _wire_settings_app_routes(self) -> None:
        sv = self.window._settings_view
        if sv is None:
            return
        if hasattr(sv, "rag_kb_toggle"):
            sv.rag_kb_toggle.connect(self.on_rag_toggle_changed)
        if hasattr(sv, "memory_enrichment_changed"):
            sv.memory_enrichment_changed.connect(self.enrichment_worker.set_enabled)
            sv.memory_enrichment_changed.connect(
                self.memory_reflection_worker.set_enabled
            )
        if hasattr(sv, "memory_promotion_changed"):
            sv.memory_promotion_changed.connect(
                self.memory_promotion_worker.set_enabled
            )
        if hasattr(sv, "memory_consolidation_changed"):
            sv.memory_consolidation_changed.connect(
                self.memory_consolidation_worker.set_enabled
            )
        if hasattr(sv, "engine_mode_changed"):
            sv.engine_mode_changed.connect(self._on_engine_mode_changed)
        if hasattr(sv, "external_settings_reloaded"):
            sv.external_settings_reloaded.connect(self._on_external_settings_reloaded)
        if hasattr(sv, "embedding_model_changed"):
            sv.embedding_model_changed.connect(self._reload_embedder_from_settings)
        if hasattr(sv, "embedding_mode_change_requested"):
            sv.embedding_mode_change_requested.connect(
                self._on_embedding_mode_change_requested
            )
        if hasattr(sv, "stt_model_changed"):
            sv.stt_model_changed.connect(self._reload_stt_from_settings)
        if hasattr(sv, "tts_model_changed"):
            sv.tts_model_changed.connect(self._reload_tts_from_settings)
        if hasattr(self, "sidecar_worker") and hasattr(sv, "_sync_active_cognition_label"):
            self.sidecar_worker.model_reload_finished.connect(
                lambda _ok, _msg: sv._sync_active_cognition_label()
            )

    def _wire_telemetry_app_routes(self) -> None:
        w = self.window
        tv = w._telemetry_view
        if tv is None:
            return
        if hasattr(self.llm_worker, "router_telemetry_updated") and hasattr(
            tv, "update_router_telemetry"
        ):
            self.llm_worker.router_telemetry_updated.connect(tv.update_router_telemetry)
        if hasattr(self, "sidecar_worker") and hasattr(
            self.sidecar_worker, "sidecar_telemetry_updated"
        ) and hasattr(tv, "update_sidecar_telemetry"):
            self.sidecar_worker.sidecar_telemetry_updated.connect(
                tv.update_sidecar_telemetry
            )
            if hasattr(tv, "_refresh_sidecar_from_worker_snapshot"):
                self.sidecar_worker.model_reload_finished.connect(
                    lambda _ok, _msg: tv._refresh_sidecar_from_worker_snapshot()
                )
        if hasattr(self.llm_worker, "sidecar_telemetry_updated") and hasattr(
            tv, "update_sidecar_telemetry"
        ):
            self.llm_worker.sidecar_telemetry_updated.connect(
                tv.update_sidecar_telemetry
            )

    def _connect_signals(self):
        w = self.window
        
        # Global Shell Routing
        self.audio_worker.status_update.connect(self._on_audio_status)
        self.stt_worker.status_update.connect(w.update_status)
        self.llm_worker.status_update.connect(w.update_status)
        self.native_llama_engine.status_update.connect(w.update_status)
        self.tts_worker.status_update.connect(w.update_status)
        self.llm_worker.context_retrieved.connect(w.update_rag_indicator)
        self.llm_worker.web_search_active.connect(w.set_web_indicator_active)
        self.llm_worker.web_search_outcome_hint.connect(w.set_web_search_outcome_hint)
        self.llm_worker.ddg_backoff_started.connect(w.on_ddg_backoff_started)
        self.llm_worker.discovery_tier_b_suggested.connect(w.on_discovery_tier_b_suggested)
        self.llm_worker.response_finished.connect(
            lambda _sid, _text: w.update_rag_indicator(False)
        )
        self.llm_worker.response_finished.connect(
            lambda _sid, _text: w.set_web_indicator_active(False)
        )
        self.tts_worker.playback_finished.connect(self._handle_tts_finished)
        self.tts_worker.playback_started.connect(w.conversations_view.on_tts_playback_started)
        self.tts_worker.playback_finished.connect(w.conversations_view.on_tts_playback_finished)
        self.tts_worker.turn_settled.connect(w.conversations_view.on_tts_turn_settled)
        self.tts_worker.turn_settled.connect(self._handle_tts_turn_settled)

        self.deep_research_worker.progress.connect(self._on_deep_research_progress)
        self.deep_research_worker.finished.connect(self._on_deep_research_finished)

        # Lazy main stages: defer signal wiring until first visit (see MainWindow lifecycle comment).
        self.tts_worker.model_loaded.connect(self.window.update_tts_voice_dropdowns)
        w.register_main_stage_app_wirer(MAIN_STAGE_LIBRARY, self._wire_library_app_routes)
        w.register_main_stage_app_wirer(MAIN_STAGE_SETTINGS, self._wire_settings_app_routes)
        w.register_main_stage_app_wirer(MAIN_STAGE_TELEMETRY, self._wire_telemetry_app_routes)
        self.native_llama_engine.load_finished.connect(self._on_native_model_load_finished)

        # Conversations View Routing
        self.llm_worker.token_streamed.connect(w.conversations_view.on_llm_token_streamed)
        self.llm_worker.stream_replaced.connect(w.conversations_view.on_llm_stream_replaced)
        self.llm_worker.sources_found.connect(w.conversations_view.on_sources_found)
        self.llm_worker.evidence_transparency_found.connect(
            w.conversations_view.on_evidence_transparency_found
        )
        # 🔑 THE FIXES: Send the live status to the text box, and unlock it when finished!
        self.llm_worker.response_finished.connect(self._on_llm_response_finished)
        self.llm_worker.turn_notice.connect(self._on_llm_turn_notice)
        # Phase B memory enrichment: per-turn rich context (rag chunk ids +
        # message ids) is emitted just before response_finished. Capture it
        # on self and hand it to the enrichment worker in
        # _on_llm_response_finished so provenance is exact.
        self.llm_worker.enrichment_context_ready.connect(self._on_enrichment_context_ready)
        w.conversations_view.set_stop_requested_callback(self.stop_active_response)
        w.conversations_view.set_before_send_callback(self._before_chat_send)
        w.conversations_view.set_manual_voice_callback(self._start_manual_voice_capture)

        # Background Data Pipeline
        self.audio_worker.audio_captured.connect(self._on_audio_captured)
        self.audio_worker.wakeword_detected.connect(
            self._on_wakeword_detected,
            QtCore.Qt.ConnectionType.BlockingQueuedConnection,
        )
        self.stt_worker.transcription_ready.connect(self._handle_voice_prompt)
        self.stt_worker.transcription_failed.connect(self._on_stt_transcription_failed)
        
        # 🔑 UI BRIDGE: Ensure the session_id is passed from the LLM to the TTS
        self.llm_worker.sentence_ready.connect(self.tts_worker.add_to_queue)
        self.llm_worker.tts_turn_superseded.connect(
            lambda _sid: self.tts_worker.stop_playback()
        )

        # Library ingest wiring is deferred until the Library page is opened.

        # Telemetry View Routing (latency hooks use MainWindow shims; page wiring is lazy)
        if hasattr(self.stt_worker, 'stt_latency'):
            self.stt_worker.stt_latency.connect(w.update_stt_latency)
            if hasattr(w, "conversations_view") and hasattr(
                w.conversations_view, "update_stt_latency"
            ):
                self.stt_worker.stt_latency.connect(
                    w.conversations_view.update_stt_latency
                )
        if hasattr(self.llm_worker, 'ttft_latency'):
            self.llm_worker.ttft_latency.connect(w.update_ttft_latency)
            if hasattr(w, "conversations_view") and hasattr(
                w.conversations_view, "update_ttft_latency"
            ):
                self.llm_worker.ttft_latency.connect(
                    w.conversations_view.update_ttft_latency
                )
        if hasattr(self.llm_worker, "tps_metric") and hasattr(
            w, "conversations_view"
        ) and hasattr(w.conversations_view, "update_tps"):
            self.llm_worker.tps_metric.connect(w.conversations_view.update_tps)
        if hasattr(self.tts_worker, 'tts_latency'):
            self.tts_worker.tts_latency.connect(w.update_tts_latency)
            if hasattr(w, "conversations_view") and hasattr(
                w.conversations_view, "update_tts_latency"
            ):
                self.tts_worker.tts_latency.connect(
                    w.conversations_view.update_tts_latency
                )
        if hasattr(self.tts_worker, 'playback_level'):
            self.tts_worker.playback_level.connect(w.on_tts_playback_level)
        if hasattr(self.audio_worker, 'volume_update'):
            self.audio_worker.volume_update.connect(w.on_audio_volume_update)
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

    def _on_deep_research_progress(self, payload: dict) -> None:
        logger.info(
            "[DeepResearch] progress request_id=%s phase=%s %s",
            payload.get("request_id"),
            payload.get("phase"),
            payload.get("message"),
        )
        self._set_deep_research_presence(active=True, payload=payload)
        conv = getattr(getattr(self, "window", None), "conversations_view", None)
        if conv is not None and hasattr(conv, "on_deep_research_progress"):
            conv.on_deep_research_progress(payload)

    def _set_deep_research_presence(self, *, active: bool, payload: dict | None = None) -> None:
        w = getattr(self, "window", None)
        if w is None:
            return
        reducer = getattr(w, "_activity_reducer", None)
        if active:
            if getattr(self, "_deep_research_presence_active", False):
                phase = str((payload or {}).get("phase") or "")
                detail = str((payload or {}).get("message") or "").strip()
                if phase == "synthesizing":
                    w.update_status("Deep research: synthesizing…")
                elif detail:
                    w.update_status(f"Deep research: {detail.rstrip('…')}")
                return
            self._deep_research_presence_active = True
            if reducer is not None:
                reducer.set_background_busy(True)
            if hasattr(w, "_sync_tray_presence"):
                w._sync_tray_presence()
            w.update_status("Deep research…")
            return
        if not getattr(self, "_deep_research_presence_active", False):
            return
        self._deep_research_presence_active = False
        if reducer is not None:
            reducer.set_background_busy(False)
        if hasattr(w, "_sync_tray_presence"):
            w._sync_tray_presence()
        w.update_status("Idle", force=True)

    def _on_deep_research_finished(self, payload: dict) -> None:
        logger.info(
            "[DeepResearch] finished request_id=%s status=%s sources=%s latency_ms=%s synthesis=%s",
            payload.get("request_id"),
            payload.get("status"),
            payload.get("source_count"),
            payload.get("latency_ms"),
            payload.get("synthesis_applied"),
        )
        conv = getattr(getattr(self, "window", None), "conversations_view", None)
        if conv is not None and hasattr(conv, "on_deep_research_finished"):
            conv.on_deep_research_finished(payload)

        status = str(payload.get("status") or "")
        session_id = str(payload.get("session_id") or "")
        active = str(getattr(conv, "active_session_id", "") or "") if conv else ""
        if status == "ok" and session_id and session_id != active:
            if hasattr(self, "window") and hasattr(self.window, "emit_notification"):
                self.window.emit_notification(
                    deep_research_complete_event(
                        session_id=session_id,
                        query=str(payload.get("query") or ""),
                        source_count=int(payload.get("source_count") or 0),
                        synthesis_applied=bool(payload.get("synthesis_applied")),
                    )
                )
        self._set_deep_research_presence(active=False)

    def _on_llm_turn_notice(self, session_id: str, payload: dict) -> None:
        kind = str((payload or {}).get("kind") or "")
        sid = str(session_id or "")
        if kind == "max_tokens":
            self.window.emit_notification(
                output_truncated_max_tokens_event(session_id=sid)
            )
        elif kind == "format_retry":
            issues = list((payload or {}).get("issues") or [])
            self.window.emit_notification(
                format_retry_in_progress_event(session_id=sid, issues=issues)
            )

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
            self.window.update_status("Idle", force=True)
        if hasattr(self, 'tts_worker'):
            self.tts_worker.enqueue_turn_complete(session_id)

    def _before_chat_send(self) -> bool:
        from ui.bootstrap_feature_prompts import ensure_main_llm_for_chat

        return ensure_main_llm_for_chat(
            self.window,
            is_dark=getattr(self.window, "_is_dark_theme", True),
        )

    def _on_audio_captured(self, audio_bytes: bytes) -> None:
        """Assign a monotonic job epoch and hand captured audio to STT."""
        from core.voice_stt_pipeline import bump_voice_stt_epoch

        self._voice_stt_epoch = bump_voice_stt_epoch(self._voice_stt_epoch)
        epoch = self._voice_stt_epoch
        logger.info(
            "[Main] Voice audio captured; dispatching STT (job_epoch=%s, bytes=%d).",
            epoch,
            len(audio_bytes or b""),
        )
        self.stt_worker.process_audio(audio_bytes, epoch)

    def _on_stt_transcription_failed(self, reason: str, job_epoch: int = 0) -> None:
        """Release voice-turn UI when STT cannot run (model missing, load error, etc.)."""
        from core.voice_stt_pipeline import is_voice_stt_job_current

        if not is_voice_stt_job_current(int(job_epoch), int(self._voice_stt_epoch)):
            logger.info(
                "[Main] Ignoring stale STT failure "
                "(job_epoch=%s, current=%s, reason=%r).",
                job_epoch,
                self._voice_stt_epoch,
                reason,
            )
            return

        logger.warning(
            "[Main] Voice STT failed (job_epoch=%s, reason=%r).",
            job_epoch,
            reason,
        )
        conv = getattr(self.window, "conversations_view", None)
        if conv is not None:
            conv._voice_turn_active = False
            conv._voice_capture_active = False
            if not getattr(conv, "_llm_in_progress", False):
                conv.set_input_enabled(True)
            conv._refresh_send_stop_button()

        reason_lower = (reason or "").strip().lower()
        if "model not loaded" in reason_lower or "model load failed" in reason_lower:
            from core.bootstrap_missing_models import missing_stt_notification

            self.window.emit_notification(missing_stt_notification())
        else:
            self.window.emit_notification(stt_failed_event(reason=reason))
        self.window.update_status("Idle", force=True)

    def _cancel_pending_voice_stt(self) -> None:
        """Invalidate in-flight voice STT and request thread interruption."""
        from core.voice_stt_pipeline import bump_voice_stt_epoch

        self._voice_stt_epoch = bump_voice_stt_epoch(self._voice_stt_epoch)
        logger.info(
            "[Main] Voice STT cancelled (job_epoch=%s).",
            self._voice_stt_epoch,
        )
        stt_worker = getattr(self, "stt_worker", None)
        if stt_worker is not None:
            stt_worker.cancel_transcription()

    def _handle_voice_prompt(self, text: str, job_epoch: int = 0):
        from core.input_source import INPUT_SOURCE_VOICE
        from core.voice_stt_pipeline import is_voice_stt_job_current
        from ui.bootstrap_feature_prompts import ensure_main_llm_for_chat

        if not is_voice_stt_job_current(int(job_epoch), int(self._voice_stt_epoch)):
            logger.info(
                "[Main] Ignoring stale voice transcription "
                "(job_epoch=%s, current=%s, preview=%r).",
                job_epoch,
                self._voice_stt_epoch,
                (text or "")[:80],
            )
            return

        cleaned = (text or "").strip()
        if not cleaned:
            conv = self.window.conversations_view
            conv._voice_turn_active = False
            conv._voice_capture_active = False
            self.window.emit_notification(stt_failed_event())
            self.window.update_status("Idle", force=True)
            return

        if not ensure_main_llm_for_chat(
            self.window,
            is_dark=getattr(self.window, "_is_dark_theme", True),
        ):
            conv = self.window.conversations_view
            conv._voice_turn_active = False
            conv._voice_capture_active = False
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

        conv = self.window.conversations_view
        conv._llm_in_progress = True
        conv._voice_turn_active = True
        conv._awaiting_tts_end = False
        conv._tts_playing = False
        conv.set_input_enabled(False)
        conv._refresh_send_stop_button()
        conv.apply_presence_label(
            self.window._presence_service.snapshot().presence_label
        )

        from core.composer_skills import parse_composer_input

        logger.info(
            "[Main] Voice prompt accepted (job_epoch=%s, session_id=%s, preview=%r).",
            job_epoch,
            session_id,
            cleaned[:80],
        )
        conv.log_user_message(text, pending_assistant=True)
        clean, attachments, enforced_skills = parse_composer_input(cleaned)
        prompt = clean if clean else cleaned
        self.llm_worker.generate_response(
            prompt,
            session_id,
            attachments=attachments,
            enforced_skills=enforced_skills,
            persist_content=cleaned.strip(),
            input_source=INPUT_SOURCE_VOICE,
        )

    def _on_audio_status(self, message: str) -> None:
        """Route audio-worker status; force-clear capture when the mic gate closes."""
        force = (message or "").strip().casefold() == "voice capture idle"
        self.window.update_status(message, force=force)

    def _on_wakeword_detected(self) -> None:
        """
        Runs on the UI thread before the audio worker opens the capture buffer.

        Only cancels an in-flight LLM/TTS turn (barge-in). Capture status is owned
        by ``AudioListenerWorker`` (Listening → Voice capture idle / Working).
        """
        logger = logging.getLogger("Qube.Main")
        interrupted = False

        session_id = getattr(self, "_pending_turn_session_id", None)
        if session_id:
            self.window.notification_service.cancel_turn_complete(session_id)

        if hasattr(self, "llm_worker") and self.llm_worker.isRunning():
            self.llm_worker.cancel_generation()
            interrupted = True
        if hasattr(self, "tts_worker") and getattr(self.tts_worker, "is_playing", False):
            self.tts_worker.stop_playback()
            interrupted = True

        if interrupted:
            logger.info("Wakeword barge-in: cancelled active generation or TTS.")
            conv = getattr(self.window, "conversations_view", None)
            if conv is not None and hasattr(conv, "on_generation_stopped"):
                conv.on_generation_stopped()
        else:
            logger.info("Wakeword detected; voice capture will start on the audio thread.")

    def _start_manual_voice_capture(self) -> None:
        """Push-to-talk from the chat composer mic button."""
        logger = logging.getLogger("Qube.Main")
        audio_worker = getattr(self, "audio_worker", None)
        if audio_worker is None or audio_worker.is_capturing_voice:
            return

        from core.bootstrap_manifest import BootstrapModelId
        from ui.bootstrap_feature_prompts import ensure_bootstrap_model_downloaded

        if not ensure_bootstrap_model_downloaded(
            self.window,
            BootstrapModelId.WHISPER_SMALL,
            feature_label="Voice input",
            is_dark=getattr(self.window, "_is_dark_theme", True),
        ):
            conv = getattr(self.window, "conversations_view", None)
            if conv is not None and hasattr(conv, "composer_voice_btn"):
                conv.composer_voice_btn.setEnabled(conv.text_input.isEnabled())
            return

        self._on_wakeword_detected()
        audio_worker.request_manual_capture()
        logger.info("Manual voice capture requested from chat composer.")

    def stop_active_response(self):
        """Manual UI stop: cancel voice capture, LLM, and/or TTS; unlock text input."""
        logger.info("[Main] Manual Stop requested from chat UI.")
        audio_worker = getattr(self, "audio_worker", None)
        llm_running = hasattr(self, "llm_worker") and self.llm_worker.isRunning()
        tts_playing = hasattr(self, "tts_worker") and getattr(
            self.tts_worker, "is_playing", False
        )
        voice_capturing = bool(
            audio_worker is not None and getattr(audio_worker, "is_capturing_voice", False)
        )
        stt_running = bool(
            hasattr(self, "stt_worker") and self.stt_worker.isRunning()
        )
        conv = getattr(self.window, "conversations_view", None)
        voice_turn_active = bool(
            conv is not None and getattr(conv, "_voice_turn_active", False)
        )
        deep_research_active = bool(
            conv is not None and getattr(conv, "_deep_research_in_progress", False)
        )

        if deep_research_active and conv is not None:
            request_id = getattr(conv, "_active_deep_research_request_id", None)
            worker = getattr(self, "deep_research_worker", None)
            if worker is not None and request_id:
                worker.cancel_request(str(request_id))
            conv._deep_research_in_progress = False
            conv._active_deep_research_request_id = None
            conv._deep_research_session_id = None
            if hasattr(conv, "_hide_deep_research_progress"):
                conv._hide_deep_research_progress()
            conv._refresh_send_stop_button()
            self._set_deep_research_presence(active=False)
            return

        if voice_capturing and audio_worker is not None:
            audio_worker.cancel_voice_capture()

        if voice_capturing or stt_running or voice_turn_active:
            self._cancel_pending_voice_stt()

        session_id = getattr(self, "_pending_turn_session_id", None)
        if session_id:
            self.window.notification_service.cancel_turn_complete(session_id)
        if llm_running:
            self.llm_worker.cancel_generation()
        if tts_playing:
            self.tts_worker.stop_playback()

        if llm_running or tts_playing:
            if conv is not None:
                conv.on_generation_stopped()
            self.window.update_status("Idle", force=True)
        elif voice_capturing and conv is not None:
            conv.on_voice_capture_stopped()
        elif conv is not None and voice_turn_active:
            conv.on_generation_stopped()
            self.window.update_status("Idle", force=True)
    
    def _handle_tts_finished(self):
        """Safely resets the UI state based on the current microphone status."""
        session_id = getattr(self, "_pending_turn_session_id", None)
        if session_id:
            self.window.notification_service.flush_turn_complete(session_id)
            self._pending_turn_session_id = None
        if hasattr(self, 'window'):
            # 1. Determine the correct safe state
            safe_status = "Idle"
                
            # 2. Update the internal window state
            self.window.update_status(safe_status, force=True)
            
            # 3. Forcefully broadcast the safe status through the worker's
            # signal pipeline so the Top Bar and Input Box catch the update!
            if hasattr(self, 'tts_worker'):
                self.tts_worker.status_update.emit(safe_status)

    def _handle_tts_turn_settled(self) -> None:
        """Backstop: return to Idle if playback_finished did not run (should be rare)."""
        bubble = getattr(self.window._presence_service, "bubble_state", "idle")
        if bubble in ("thinking", "writing", "speaking"):
            self.window.update_status("Idle", force=True)

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

        removed = self.db_manager.dedupe_library_document_metadata()
        if removed:
            logger.info(
                "Removed %d duplicate library metadata row(s) during sync.",
                removed,
            )

        # Get what actually exists in the vector store
        lancedb_sources = self.store.get_all_indexed_sources()

        # Compare against every registered filename, not a recent-page subset.
        sqlite_docs = self.db_manager.get_all_library_document_filenames()

        # Calculate what is missing from the UI
        missing_from_ui = set(lancedb_sources) - sqlite_docs

        if missing_from_ui:
            logger.warning(
                "Found %d LanceDB source(s) missing from SQLite library registry; healing UI.",
                len(missing_from_ui),
            )
            from core.library_folder_policy import is_qube_managed_document_filename

            for source in sorted(missing_from_ui):
                if is_qube_managed_document_filename(source):
                    folder_id = self.db_manager.get_qube_library_folder_id()
                else:
                    folder_id = self.db_manager.get_main_library_folder_id()
                # Add a placeholder row so the UI can see orphaned LanceDB sources.
                self.db_manager.add_document_metadata(
                    source,
                    file_size_kb=0,
                    chunk_count=0,
                    folder_id=folder_id,
                )

            logger.info("Database synchronization complete.")

    def _maybe_seed_help_corpus(self) -> None:
        if self.embedder is None or is_reindex_in_progress():
            return
        try:
            from core.help_corpus_seed import seed_help_corpus_if_needed

            summary = seed_help_corpus_if_needed(
                self.store,
                self.embedder,
                self.db_manager,
            )
            if summary.get("skipped"):
                logger.debug("Help corpus seed: %s", summary.get("reason", "skipped"))
            else:
                logger.info(
                    "Help corpus seeded (indexed=%s, chunks=%s, corpus_version=%s)",
                    summary.get("indexed"),
                    summary.get("chunks"),
                    summary.get("corpus_version"),
                )
        except Exception as exc:
            logger.warning("Help corpus seed failed: %s", exc, exc_info=True)

    def _start_ingestion(self, file_paths: list, folder_id: str, ingest_mode: str = "standard"):
        """Spawns a background thread to safely embed documents without freezing the UI."""
        if is_reindex_in_progress():
            self.window.ensure_library_view().show_error(
                "Reprocessing is still running. Please wait until it finishes."
            )
            return
        self.window.update_status("Ingesting Documents...")
        self.window._activity_reducer.set_background_busy(True)
        self.window._sync_tray_presence()
        self.window.begin_background_progress("Preparing documents for indexing…")

        self.ingestion_worker = IngestionWorker(
            file_paths,
            self.embedder,
            self.store,
            self.db_manager,
            folder_id=folder_id,
            sidecar_worker=self.sidecar_worker,
            ingest_mode=ingest_mode,
        )

        # Wire the worker's progress signals back to the Library UI
        self.ingestion_worker.progress_update.connect(self._update_embedding_job_progress)
        self.ingestion_worker.file_done.connect(self._set_embedding_job_detail)
        self.ingestion_worker.ingestion_complete.connect(
            self.window.ensure_library_view().complete_ingestion
        )
        self.ingestion_worker.ingestion_complete.connect(self._on_ingestion_complete)

        # Route backend errors directly to the UI popup
        self.ingestion_worker.error_occurred.connect(
            self.window.ensure_library_view().show_error
        )
        self.ingestion_worker.error_occurred.connect(lambda _err: self._finish_embedding_job_ui())

        # Keep the terminal log as a backup
        self.ingestion_worker.error_occurred.connect(lambda err: logger.error(f"Ingestion Error: {err}"))

        # Fire it up!
        self.ingestion_worker.start()

    def _on_ingest_blurb_ready(self, filename: str, blurb: str) -> None:
        if self.db_manager.update_document_blurb(filename, blurb):
            if self.window._library_view is not None:
                self.window._library_view.refresh_library_list()

    def _on_ingestion_complete(self, chunk_count: int) -> None:
        file_count = len(getattr(self.ingestion_worker, "file_paths", []) or [])
        if file_count <= 0 and chunk_count > 0:
            file_count = 1
        if file_count > 0:
            self.window.emit_notification(ingestion_complete_event(file_count=file_count))
        self._finish_embedding_job_ui()
        # Clear background-busy before forcing Idle — otherwise BACKGROUND_BUSY wins
        # over an idle bubble (see AssistantActivityReducer.reduce).
        self.window._activity_reducer.set_background_busy(False)
        self.window.update_status("Idle", force=True)
        if (
            hasattr(self.window, "_companion_controller")
            and self.window._companion_controller is not None
        ):
            self.window._companion_controller.on_ingestion_complete(file_count)

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
            sv = self.window._settings_view
            if sv is not None and hasattr(sv, "_sync_wakeword_catalog"):
                sv._sync_wakeword_catalog(trigger="external settings")
        if KEY_MCP_INTERNET_HYBRID in changed:
            enabled = get_mcp_internet_hybrid_enabled()
            if hasattr(self, "llm_worker"):
                self.llm_worker.set_mcp_internet_hybrid(enabled)
            win = self.window
            toolbar_toggle = getattr(win, "tool_internet_hybrid_toggle", None)
            if toolbar_toggle is not None and toolbar_toggle.isChecked() != enabled:
                toolbar_toggle.blockSignals(True)
                toolbar_toggle.setChecked(enabled)
                toolbar_toggle.blockSignals(False)
            if hasattr(win, "_web_indicator_hybrid"):
                win._web_indicator_hybrid = bool(enabled)
            if hasattr(win, "_apply_web_indicator"):
                win._apply_web_indicator()
        sv = getattr(self.window, "_settings_view", None)
        if sv is not None and hasattr(sv, "_apply_external_privacy_settings_changed"):
            sv._apply_external_privacy_settings_changed(changed)

    def _on_native_model_load_finished(self, ok: bool, message: str) -> None:
        """Update Think toggle when internal GGUF load completes."""
        self._refresh_conversations_think_toggle()
        if ok and hasattr(self.window, "_companion_controller"):
            ctrl = self.window._companion_controller
            if ctrl is not None:
                import os

                path = getattr(self.native_llama_engine, "_model_path", "") or ""
                basename = os.path.basename(path) if path else ""
                if basename:
                    ctrl.on_model_loaded(basename)

    def _refresh_conversations_think_toggle(self) -> None:
        cv = getattr(getattr(self, "window", None), "conversations_view", None)
        if cv is not None and hasattr(cv, "refresh_think_toggle"):
            cv.refresh_think_toggle()

    def on_rag_toggle_changed(self, is_enabled: bool):
        """Updates the LLM worker when the user flips the RAG switch."""
        if hasattr(self, 'llm_worker'):
            self.llm_worker.set_mcp_rag(is_enabled)
            logger.debug(f"RAG Engine manually set to: {is_enabled}")

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def show(self) -> None:
        self.window.show()
        from PyQt6.QtCore import QTimer

        QTimer.singleShot(0, self.window._finalize_startup_geometry)
        QTimer.singleShot(120, self.window._finalize_startup_geometry)

    def _wait_worker_shutdown(
        self,
        worker,
        timeout_ms: int,
        *,
        label: str,
        warn_on_timeout: bool = True,
    ) -> bool | None:
        """Wait for a worker thread to exit.

        Returns True when the worker exited, False on timeout, None if interrupted.
        """
        try:
            exited = worker.wait(timeout_ms)
        except KeyboardInterrupt:
            logger.warning(
                "[Shutdown] Interrupted while waiting for %s; continuing exit.",
                label,
            )
            return None
        if warn_on_timeout and not exited:
            logger.warning(
                "[Shutdown] %s did not exit within %ds.",
                label,
                timeout_ms // 1000,
            )
        return exited

    def _graceful_shutdown(self):
        """Called automatically when the application is closing."""
        logger.info("Initiating graceful shutdown...")

        # 0. Model Manager — Hub search/README/list/download QThreads can block exit if still running
        mm = self.window._model_manager_view
        if mm is not None:
            mm.shutdown_hf_workers()

        # 0b. Memory Manager — QThread is not stopped via closeEvent when the page is embedded in the stack
        mmv = self.window._memory_manager_view
        if mmv is not None:
            mmw = getattr(mmv, "worker", None)
            if mmw is not None and hasattr(mmw, "isRunning") and mmw.isRunning():
                if hasattr(mmw, "shutdown"):
                    mmw.shutdown()
                self._wait_worker_shutdown(
                    mmw, 5000, label="Memory manager worker"
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
            self._wait_worker_shutdown(
                self.active_internet_worker, 2000, label="Internet worker"
            )

        if hasattr(self, 'ingestion_worker') and self.ingestion_worker.isRunning():
            self.ingestion_worker.stop()
            self._wait_worker_shutdown(
                self.ingestion_worker, 2000, label="Ingestion worker"
            )

        # 2. Stop the core background loop (Enrichment/Memory)
        if hasattr(self, 'enrichment_worker') and self.enrichment_worker.isRunning():
            self.enrichment_worker.stop()
            self._wait_worker_shutdown(
                self.enrichment_worker, 2000, label="Enrichment worker"
            )

        # Phase C: stop the periodic memory self-reflection worker.
        if hasattr(self, 'memory_reflection_worker') and self.memory_reflection_worker.isRunning():
            self.memory_reflection_worker.shutdown()
            self._wait_worker_shutdown(
                self.memory_reflection_worker, 2000, label="Memory reflection worker"
            )

        if hasattr(self, 'memory_promotion_worker') and self.memory_promotion_worker.isRunning():
            self.memory_promotion_worker.shutdown()
            self._wait_worker_shutdown(
                self.memory_promotion_worker, 2000, label="Memory promotion worker"
            )

        if hasattr(self, 'memory_consolidation_worker') and self.memory_consolidation_worker.isRunning():
            self.memory_consolidation_worker.shutdown()
            self._wait_worker_shutdown(
                self.memory_consolidation_worker,
                2000,
                label="Memory consolidation worker",
            )

        if hasattr(self, 'tts_worker'):
            # Cut any in-flight audio first, then request cooperative thread exit.
            self.tts_worker.stop_playback()
            self.tts_worker.request_graceful_stop()
            tts_exited = self._wait_worker_shutdown(
                self.tts_worker, 2000, label="TTS worker"
            )
            if tts_exited is False:
                # One more cooperative nudge before giving up on native handle teardown.
                self.tts_worker.request_graceful_stop()
                tts_exited = self._wait_worker_shutdown(
                    self.tts_worker, 3000, label="TTS worker"
                )
                if tts_exited is False:
                    logger.error(
                        "[Shutdown] TTS worker still active; skipping audio handle close to avoid crash."
                    )
            elif tts_exited:
                logger.info("[Shutdown] TTS worker exited cleanly.")
            if tts_exited and hasattr(self.tts_worker, "close_audio_resources"):
                self.tts_worker.close_audio_resources()

        if hasattr(self, "native_llama_engine"):
            self.native_llama_engine.stop_engine()

        if hasattr(self, "sidecar_worker") and self.sidecar_worker.isRunning():
            self.sidecar_worker.stop_engine()
            sidecar_exited = self._wait_worker_shutdown(
                self.sidecar_worker, 2000, label="Sidecar worker"
            )
            if sidecar_exited is False:
                sidecar_exited = self._wait_worker_shutdown(
                    self.sidecar_worker, 3000, label="Sidecar worker"
                )
                if sidecar_exited is False:
                    logger.error(
                        "[Shutdown] Sidecar worker still active after extended wait."
                    )
            elif sidecar_exited:
                logger.info("[Shutdown] Sidecar worker exited cleanly.")

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
                self._wait_worker_shutdown(
                    worker, 2000, label=f"{name} worker"
                )

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
            self._wait_worker_shutdown(llm, 10_000, label="LLM worker")

        stt = getattr(self, "stt_worker", None)
        if stt is not None and stt.isRunning():
            stt.cancel_transcription()
            if self._wait_worker_shutdown(stt, 10_000, label="STT worker") is False:
                logger.warning("[Shutdown] STT worker still running; terminating thread.")
                stt.terminate()
                self._wait_worker_shutdown(stt, 3000, label="STT worker")

        audio = getattr(self, "audio_worker", None)
        if audio is not None and audio.isRunning():
            if hasattr(audio, "stop"):
                audio.stop()
            if self._wait_worker_shutdown(audio, 8000, label="Audio worker") is False:
                logger.warning(
                    "[Shutdown] Audio worker still running; blocking until exit."
                )
                try:
                    audio.wait()
                except KeyboardInterrupt:
                    logger.warning(
                        "[Shutdown] Interrupted while waiting for audio worker; continuing exit."
                    )


if __name__ == "__main__":
    args = parse_boot_args()
    configure_user_model_paths()
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
    apply_linux_desktop_integration(app)
    repo_root = install_root()
    app_icon = qube_window_icon()
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)
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

    from core.reading_fonts import ensure_reading_fonts_loaded

    ensure_reading_fonts_loaded()

    from core.theme.applicator import ThemeApplicator
    from core.theme.feature_flags import is_generated_theme_enabled
    from core.theme.manager import ThemeManager
    from core.theme.storage import theme_storage_from_app_settings
    from core.surface_fill.storage import surface_fill_storage_from_app_settings

    theme_manager = ThemeManager(
        storage=theme_storage_from_app_settings(),
        surface_storage=surface_fill_storage_from_app_settings(),
        applicator=ThemeApplicator(
            use_generated_stylesheet=is_generated_theme_enabled(),
        ),
    )

    # 3. Boot the Qube Assistant (first launch defaults to Internal Engine)
    ensure_engine_mode_initialized()

    from core.bootstrap_selection import (
        effective_bootstrap_selection,
        get_voice_input_default,
        get_voice_output_default,
        is_bootstrap_completed,
        should_show_bootstrap_consent,
    )

    selected_models = effective_bootstrap_selection()
    needs_consent = should_show_bootstrap_consent()
    if args.mock_bootstrap_download:
        os.environ["QUBE_BOOTSTRAP_MOCK_DOWNLOAD"] = "1"
        logger.info("Bootstrap mock downloads enabled (--mock-bootstrap-download).")

    def _build_qube(
        *,
        embedder: EmbeddingModel | None,
        on_phase,
        on_complete,
        on_failed=None,
    ):
        return start_phased_qube_build(
            embedder=embedder,
            enable_routing_debug_tool=bool(args.routing_debug),
            enable_trace_diff_debug_tool=bool(
                args.trace_diff_debug or args.run_scenario or args.compare_sessions
            ),
            on_phase=on_phase,
            on_complete=on_complete,
            on_failed=on_failed,
            theme_manager=theme_manager,
        )

    def _on_qube_ready(qube: Qube) -> None:
        qube.window._qube = qube
        if is_bootstrap_completed():
            if hasattr(qube.window, "voice_input_toggle"):
                qube.window.voice_input_toggle.setChecked(get_voice_input_default())
            if hasattr(qube.window, "voice_bypass_toggle"):
                qube.window.voice_bypass_toggle.setChecked(get_voice_output_default())
        qube_tooltip_set_theme(getattr(qube.window, "_is_dark_theme", True))
        app.aboutToQuit.connect(qube._graceful_shutdown)
        if args.run_scenario:
            scenario_path = args.run_scenario
            if not os.path.isabs(scenario_path):
                scenario_path = os.path.join(str(repo_root), scenario_path)
            qube.window._run_scenario_path = scenario_path
            qube.window._scenario_backend = str(args.scenario_backend or "qube")
            qube.window._scenario_single_phase = bool(
                getattr(args, "scenario_single_phase", False)
            )
            if not qube.window.canonical_trace_diff_view:
                qube.window._setup_trace_diff_debug_window()
            qube.window.schedule_scenario_replay()
        if getattr(args, "compare_sessions", None):
            path_a, path_b = args.compare_sessions
            if not os.path.isabs(path_a):
                path_a = os.path.join(str(repo_root), path_a)
            if not os.path.isabs(path_b):
                path_b = os.path.join(str(repo_root), path_b)
            qube.window._compare_sessions = (path_a, path_b)
            if not qube.window.canonical_trace_diff_view:
                qube.window._setup_trace_diff_debug_window()
            qube.window.schedule_scenario_replay()
        qube.show()
        from PyQt6.QtCore import QTimer

        QTimer.singleShot(250, qube.window.focus_chat_composer_if_ready)

    # Keep a strong reference; otherwise StartupSplashController is GC'd and startup timers never fire.
    app._startup_splash_controller = bootstrap_with_splash(
        repo_root=repo_root,
        build_app_fn=_build_qube,
        on_ready=_on_qube_ready,
        selected_models=selected_models,
        needs_consent=needs_consent,
        mock_downloads=bool(args.mock_bootstrap_download),
    )
    logger.info("Entering Qt event loop.")
    sys.exit(app.exec())


    