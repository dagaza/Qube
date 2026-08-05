"""Main window. Prefer starting the app with `python main.py` from the repo root."""

import sys
from pathlib import Path

# Running `python ui/main_window.py` does not set a package; absolute `ui.*` imports need repo root on sys.path.
if __package__ in (None, ""):
    _repo_root = Path(__file__).resolve().parent.parent
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

import psutil
import math
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QToolButton, QLabel, QFrame,
    QSizeGrip, QMenu, QSystemTrayIcon, QStackedWidget, QSizePolicy,
    QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox, QProgressBar, QWIDGETSIZE_MAX,
    QGraphicsDropShadowEffect,
)
from PyQt6.QtCore import Qt, QSize, QTimer, QEasingCurve, QPropertyAnimation, QRect
from PyQt6.QtGui import (
    QAction, QPainter, QColor, QLinearGradient, QPixmap, QIcon, QFontMetrics, QScreen, QPen,
)
import qtawesome as qta
from core.paths import install_root, resource_path
from ui.app_icon import apply_window_branding, finalize_window_branding
from ui.views.conversations_view import ConversationsView
from ui.views.settings_view import SettingsView
from ui.views.library_view import LibraryView
from ui.views.memory_manager_view import MemoryManagerView
from ui.views.telemetry_view import TelemetryView
from ui.views.model_manager_view import ModelManagerView
from ui.components.toggle import PrestigeToggle
from ui.components.selector_button import SelectorButton
from ui.components.prestige_dialog import PrestigeDialog
from ui.components.app_notifications import AppNotificationCenter
from ui.components.ingest_progress_row import IngestProgressRow
from ui.components.modal_backdrop import ModalBackdrop
from ui.shell_theme import (
    accent_icon_color,
    apply_prestige_menu_theme,
    chevron_colors,
    muted_icon_color,
    nav_icon_colors,
    resolve_shell_theme,
    retrieval_indicator_colors,
    telemetry_metric_colors,
    theme_toggle_icon_colors,
    vu_meter_palette,
)
from core.theme.svg_icons import tinted_svg_icon, themed_fa_icon, themed_fa_pixmap
from core.theme.color_utils import with_alpha
from core.theme.constants import UNRESOLVED_TOKEN_COLOR
from core.theme.widget_styles import SUCCESS_STATUS
from core.app_notification_types import AppNotificationRequest
from core.app_restart import relaunch_and_quit, manual_restart_instructions
from core.assistant_presence import AssistantPresenceService
from core.companion_policy import companion_attention_mode
from core.notification_service import NotificationService
from core.notification_types import NotificationEvent
from ui.os_notification_adapter import OsNotificationAdapter
from ui.tray_controller import TrayController
from ui.companion.companion_controller import CompanionController
from core.app_settings import (
    get_auto_load_last_model_on_startup,
    get_audio_input_device_index,
    get_engine_mode,
    get_internal_model_path,
    get_llm_chat_history_messages,
    get_llm_context_limit,
    get_llm_models_dir,
    get_llm_output_token_limit,
    get_llm_output_token_limit_enabled,
    get_llm_temperature,
    get_mcp_internet_hybrid_enabled,
    get_mcp_rag_auto_activator_enabled,
    get_mcp_rag_enabled,
    get_mcp_rag_strict_enabled,
    get_onboarding_local_llm_tour_completed,
    is_secondary_gguf_shard,
    resolve_internal_model_path,
    set_auto_load_last_model_on_startup,
    set_audio_input_device_index,
    set_internal_model_path,
)
from core.audio_utils import get_input_devices
from core.local_gguf_display import format_local_gguf_display, local_gguf_sort_key
from core.local_gguf_library import list_local_gguf_menu_entries
from core.qube_tooltip import qube_tooltip_set_theme
from core.ui_language import tr
from core.platform.work_area import workspace_bounds_for_screen
from ui.onboarding.local_llm_setup_tour import build_local_llm_setup_tour
import logging

logger = logging.getLogger("Qube.UI")

# ---------------------------------------------------------------------------
# Main stage router (QStackedWidget indices == nav button indices)
#
#   0 Conversations  — built at startup (always present)
#   1 Library        — lazy (placeholder until first nav click)
#   2 Memory         — lazy
#   3 Telemetry      — lazy
#   4 Model Manager  — lazy
#   5 Settings       — lazy
#
# Lifecycle:
#   • Startup: only Conversations + empty placeholders exist in the stack.
#   • Navigation: ``_route_view`` → ``_ensure_main_stage_view`` replaces a
#     placeholder, wires page-local + app-level routes, applies theme.
#   • Theme toggle: refreshes built stages only; unbuilt pages cost ~0.
#   • App wiring: ``main.py`` registers ``register_main_stage_app_wirer``;
#     do not connect signals via ``window.settings_view`` (eager-builds).
#
# Access conventions (avoid accidental eager construction):
#   • Peek:  ``self._settings_view`` or ``peek_settings_view()`` → None if unbuilt
#   • Build: ``ensure_settings_view()`` or navigate via ``_route_view``
#   • Never: ``getattr(w, "settings_view")`` / ``hasattr(w, "settings_view")``
# ---------------------------------------------------------------------------
MAIN_STAGE_CONVERSATIONS = 0
MAIN_STAGE_LIBRARY = 1
MAIN_STAGE_MEMORY = 2
MAIN_STAGE_TELEMETRY = 3
MAIN_STAGE_MODEL_MANAGER = 4
MAIN_STAGE_SETTINGS = 5
_LAZY_MAIN_STAGE_INDICES = frozenset(
    {
        MAIN_STAGE_LIBRARY,
        MAIN_STAGE_MEMORY,
        MAIN_STAGE_TELEMETRY,
        MAIN_STAGE_MODEL_MANAGER,
        MAIN_STAGE_SETTINGS,
    }
)

# Restore/maximize tuning for portrait and narrow monitors (width <= default layout minimum).
_RESTORE_MARGIN_PX = 24
_RESTORE_WIDTH_RATIO = 0.90
_RESTORE_HEIGHT_RATIO = 0.82
_ABSOLUTE_MIN_WIDTH = 640
_ABSOLUTE_MIN_HEIGHT = 480

class VUMeter(QWidget):
    """A sleek, custom-painted VU meter with a Green-Yellow-Red gradient."""
    _ATTENTION_TICK_MS = 50
    _ATTENTION_DEFAULT_MS = 8000
    _ATTENTION_LEVEL_STOP = 0.08
    _ATTENTION_LEVEL_STOP_DELAY_MS = 2000

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(100, 6) # Thin, modern horizontal bar
        self._level = 0.0 # Range: 0.0 to 1.0
        self._attention_pulse = False
        self._pulse_phase = 0.0
        self._attention_timer = QTimer(self)
        self._attention_timer.setInterval(self._ATTENTION_TICK_MS)
        self._attention_timer.timeout.connect(self._on_attention_tick)
        self._attention_remaining_ms = 0
        self._attention_elapsed_ms = 0
        self._glow_effect = QGraphicsDropShadowEffect(self)
        self._glow_effect.setBlurRadius(12)
        self._glow_effect.setOffset(0, 0)
        self._glow_effect.setColor(QColor(0, 0, 0, 0))
        self.setGraphicsEffect(self._glow_effect)
        self._palette: dict[str, str] = {}

    def apply_theme(self, theme) -> None:
        self._palette = vu_meter_palette(theme)
        self.update()

    def _color(self, key: str, fallback_key: str = "accent") -> QColor:
        return QColor(
            self._palette.get(key, self._palette.get(fallback_key, UNRESOLVED_TOKEN_COLOR))
        )

    def start_attention_pulse(self, duration_ms: int | None = None) -> None:
        """Pulse a glow around the meter so users notice the live input level bar."""
        ms = self._ATTENTION_DEFAULT_MS if duration_ms is None else max(500, int(duration_ms))
        self._attention_pulse = True
        self._attention_remaining_ms = ms
        self._attention_elapsed_ms = 0
        if not self._attention_timer.isActive():
            self._attention_timer.start()
        self._sync_attention_glow()
        self.raise_()

    def _stop_attention_pulse(self) -> None:
        self._attention_pulse = False
        self._attention_remaining_ms = 0
        self._attention_elapsed_ms = 0
        self._attention_timer.stop()
        self._glow_effect.setColor(QColor(0, 0, 0, 0))
        self.update()
        parent = self.parentWidget()
        if parent is not None:
            parent.update()

    def _sync_attention_glow(self) -> None:
        glow_alpha = int(120 + 135 * (0.5 + 0.5 * math.sin(self._pulse_phase)))
        glow = self._color("accent")
        glow.setAlpha(glow_alpha)
        self._glow_effect.setColor(glow)
        self.update()
        parent = self.parentWidget()
        if parent is not None:
            parent.update()

    def _on_attention_tick(self) -> None:
        self._pulse_phase += 0.22
        self._attention_remaining_ms -= self._ATTENTION_TICK_MS
        self._attention_elapsed_ms += self._ATTENTION_TICK_MS
        if self._attention_remaining_ms <= 0:
            self._stop_attention_pulse()
            return
        self._sync_attention_glow()

    def set_level(self, level: float):
        """Updates the visual level and triggers a repaint."""
        level = max(0.0, min(1.0, float(level)))
        # Fast attack, slower release so speech reads smoothly between chunks.
        if level >= self._level:
            self._level = level
        else:
            self._level = max(level, self._level * 0.82)
        if (
            self._attention_pulse
            and level >= self._ATTENTION_LEVEL_STOP
            and self._attention_elapsed_ms >= self._ATTENTION_LEVEL_STOP_DELAY_MS
        ):
            self._stop_attention_pulse()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 1. Draw the dark background track
        track_color = self._color(
            "track_pulse" if self._attention_pulse else "track_idle"
        )
        painter.setBrush(track_color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 3, 3)

        if self._attention_pulse:
            overlay_alpha = int(40 + 35 * (0.5 + 0.5 * math.sin(self._pulse_phase * 0.9)))
            overlay = self._color("accent")
            overlay.setAlpha(overlay_alpha)
            painter.setBrush(overlay)
            painter.drawRoundedRect(self.rect(), 3, 3)

            rim_alpha = int(90 + 80 * (0.5 + 0.5 * math.sin(self._pulse_phase * 1.2)))
            rim = self._color("accent_hover")
            rim.setAlpha(rim_alpha)
            painter.setPen(QPen(rim, 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 3, 3)
            painter.setPen(Qt.PenStyle.NoPen)

        if self._level > 0:
            # 2. Calculate how far the bar should fill
            active_width = int(self.width() * self._level)
            active_rect = QRect(0, 0, active_width, self.height())

            # 3. Create the Green -> Yellow -> Red gradient
            gradient = QLinearGradient(0, 0, self.width(), 0)
            gradient.setColorAt(0.0, self._color("gradient_start"))
            gradient.setColorAt(0.7, self._color("gradient_mid"))
            gradient.setColorAt(1.0, self._color("gradient_end"))

            # 4. Paint the active level
            painter.setBrush(gradient)
            painter.drawRoundedRect(active_rect, 3, 3)
        elif self._attention_pulse:
            # Subtle idle shimmer so the bar is noticeable before the user speaks.
            shimmer_alpha = int(80 + 70 * (0.5 + 0.5 * math.sin(self._pulse_phase * 1.4)))
            shimmer = self._color("accent_muted")
            shimmer.setAlpha(shimmer_alpha)
            painter.setBrush(shimmer)
            painter.drawRoundedRect(QRect(0, 0, max(16, self.width() // 4), self.height()), 3, 3)

class NoScrollSpinBox(QSpinBox):
    def wheelEvent(self, event):
        event.ignore() # Blocks the scroll from changing the value

class NoScrollDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):
        event.ignore()

class MainWindow(QMainWindow):
    """
    MASTER GLOBAL SHELL
    Responsible for the frameless lifecycle, global navigation, and routing.
    All distinct screens are hosted within the QStackedWidget (Main Stage).
    """

    def __init__(
        self,
        workers: dict,
        gpu_monitor,
        native_engine=None,
        enable_routing_debug_tool: bool = False,
        enable_trace_diff_debug_tool: bool = False,
        run_scenario_path: str = "",
        scenario_backend: str = "qube",
        compare_sessions: tuple[str, str] | None = None,
        theme_manager=None,
    ):
        super().__init__()
        self._project_root = install_root()
        self.setWindowTitle("Qube - Workspace")
        self._default_minimum_size = QSize(1200, 950)

        # Custom maximize: Qt's showMaximized() is unreliable for frameless windows on
        # secondary/portrait monitors (sets maximized state without resizing, which also
        # hides QSizeGrip). Track geometry ourselves against the screen under the window.
        self._workspace_maximized = False
        self._pre_maximize_geometry: QRect | None = None
        self._pending_restore_geometry: QRect | None = None
        self._geometry_update_depth = 0

        self.workers = workers
        self.db = workers.get("db") # Ensure your DB manager is in the workers dict

        # 1. Frameless Window Setup
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # setWindowFlags() recreates the native window and drops prior icon/size state.
        apply_window_branding(self)
        self.setMinimumSize(self._default_minimum_size)
        self._old_pos = None

        # 2. Worker References
        self._audio_worker = workers.get("audio")
        self._tts_worker   = workers.get("tts")
        self._llm_worker   = workers.get("llm")
        self._gpu_monitor  = gpu_monitor
        self._native_engine = native_engine
        self._enable_routing_debug_tool = bool(enable_routing_debug_tool)
        self._enable_trace_diff_debug_tool = bool(enable_trace_diff_debug_tool)
        self._run_scenario_path = str(run_scenario_path or "").strip()
        self._scenario_backend = str(scenario_backend or "qube").strip().lower()
        self._scenario_single_phase = False
        self._compare_sessions: tuple[str, str] | None = compare_sessions
        self._scenario_replay_started = False
        self._scenario_qube_phase_done = False
        self._scenario_workflow_dialog = None
        self._scenario_workflow_dialog_open = False
        self.routing_debug_tool_view = None
        self.canonical_trace_diff_view = None
        self._force_app_exit = False
        self._last_mic_notification_detail: str | None = None
        self._pending_native_model_path: str | None = None
        self._native_model_loading: bool = False
        self._native_model_unloading: bool = False
        self._native_model_loaded_success: bool = False
        self._presence_service = AssistantPresenceService(self)
        self._activity_reducer = self._presence_service
        self._notification_service = NotificationService(self)
        self._os_notification_adapter = OsNotificationAdapter()
        self.tray_controller: TrayController | None = None
        self.tray_icon = None  # legacy alias set by TrayController
        self._companion_controller: CompanionController | None = None

        self._sidecar_client = workers.get("sidecar")
        self._sidecar_worker = workers.get("sidecar_worker")

        from core.theme.applicator import ThemeApplicator
        from core.theme.feature_flags import is_generated_theme_enabled
        from core.theme.manager import ThemeManager
        from core.theme.storage import theme_storage_from_app_settings
        from core.surface_fill.storage import surface_fill_storage_from_app_settings

        self._owns_theme_manager = theme_manager is None
        self._theme_manager = theme_manager or ThemeManager(
            storage=theme_storage_from_app_settings(),
            surface_storage=surface_fill_storage_from_app_settings(),
            applicator=ThemeApplicator(
                use_generated_stylesheet=is_generated_theme_enabled(),
            ),
        )
        self._theme_manager.subscribe(self._on_theme_applied)
        self._active_theme_profiler = None
        self._pending_hidden_stages: list[int] | None = None
        self._theme_stage_dirty: set[int] = set()
        self._setup_system_appearance_listener()

        self._cached_tts_model_name: str = ""
        self._cached_tts_voices: list[str] = []

        self._setup_ui()
        if self._native_engine is not None:
            self._native_engine.load_finished.connect(self._on_native_model_load_finished_ui)
            self._native_engine.status_update.connect(self._on_native_engine_status_update)
        self._setup_tray()
        self._setup_companion()
        self._start_timers()

        # 🔑 4. Wire the AI Titling Logic
        # We wait until the UI is setup so we can access conversations_view
        self._setup_titling_connections()

        self._local_llm_tour = build_local_llm_setup_tour(self)
        self._active_tour = None
        self._theme_profile_startup_logged = False

        if self._owns_theme_manager:
            self._theme_manager.apply(persist=False)

    @property
    def theme_manager(self):
        return self._theme_manager

    @property
    def _is_dark_theme(self) -> bool:
        return self._theme_manager.is_dark

    def _setup_system_appearance_listener(self) -> None:
        """Follow OS light/dark changes when appearance preference is follow-system."""
        from PyQt6.QtGui import QGuiApplication

        app = QGuiApplication.instance()
        if app is None:
            return
        hints = app.styleHints()
        hints.colorSchemeChanged.connect(self._on_system_color_scheme_changed)

    def _on_system_color_scheme_changed(self, _scheme) -> None:
        from core.theme.follow_system import ThemeAppearancePreference

        if self._theme_manager.appearance_preference is not ThemeAppearancePreference.FOLLOW_SYSTEM:
            return
        self._theme_manager.sync_with_system_appearance(persist=True)

    def _on_theme_applied(self, resolved) -> None:
        """Refresh widget-level theme chrome after ``ThemeManager.apply()``."""
        from PyQt6.QtWidgets import QApplication

        from core.qube_tooltip import qube_tooltip_set_theme
        from core.richtext_styles import apply_app_link_palette
        from core.theme_toggle_profile import ThemeToggleProfiler

        is_dark = resolved.is_dark
        profiler = self._active_theme_profiler or ThemeToggleProfiler.maybe_enabled()

        with profiler.step("nav_theme_chrome"):
            self._update_nav_theme_toggle(is_dark)
            qube_tooltip_set_theme(is_dark)

        if self._pending_hidden_stages is not None:
            if is_dark:
                logger.info("Theme switched to Dark Mode.")
            else:
                logger.info("Theme switched to Light Mode.")

        app = QApplication.instance()
        active_stage = self._active_main_stage_index()
        hidden_stages = self._pending_hidden_stages
        if hidden_stages is None:
            hidden_stages = [
                idx
                for idx in range(self.main_stage.count())
                if idx != active_stage and self._is_main_stage_built(idx)
            ]

        self.setUpdatesEnabled(False)
        try:
            with profiler.step("apply_app_link_palette"):
                apply_app_link_palette(app, theme=resolved)

            self._refresh_global_theme_chrome(is_dark, profiler)

            with profiler.step(f"stage_{active_stage}.refresh"):
                self._refresh_stage_theme(active_stage, is_dark)
        finally:
            self.setUpdatesEnabled(True)
            self.update()

        if self._active_theme_profiler is None:
            self._schedule_deferred_theme_refreshes(hidden_stages)

    def _nav_theme_tooltip(self, is_dark: bool) -> str:
        from core.theme.catalog import ThemeCatalog
        from core.theme.tokens import ThemeMode

        catalog = ThemeCatalog(self._theme_manager.list_schemes())
        target_mode = ThemeMode.LIGHT if is_dark else ThemeMode.DARK
        sibling = catalog.sibling_for_polarity(self._theme_manager.scheme_id, target_mode)
        if sibling:
            return f"Switch to {catalog.display_name(sibling)}"
        if is_dark:
            return "Switch to light theme"
        return "Switch to dark theme"

    def _update_nav_theme_toggle(self, is_dark: bool) -> None:
        import qtawesome as qta

        theme = resolve_shell_theme(self, is_dark=is_dark)
        moon, sun = theme_toggle_icon_colors(theme)
        if is_dark:
            self.nav_theme.setIcon(qta.icon("fa5s.moon", color=moon))
            self.nav_theme.setToolTip(self._nav_theme_tooltip(is_dark=True))
        else:
            self.nav_theme.setIcon(qta.icon("fa5s.sun", color=sun))
            self.nav_theme.setToolTip(self._nav_theme_tooltip(is_dark=False))

    def _sync_retrieval_indicator_palette(self) -> None:
        colors = retrieval_indicator_colors(self._theme_manager.current)
        self._RETRIEVAL_COLOR_OFF = colors["off"]
        self._RETRIEVAL_COLOR_ACTIVE = colors["active"]
        self._RAG_COLOR_STANDBY = colors["rag_standby"]
        self._WEB_COLOR_STANDBY = colors["web_standby"]
        self._DDG_BACKOFF_COLOR = colors["ddg_backoff"]
        if hasattr(self, "ddg_backoff_label"):
            self.ddg_backoff_label.setStyleSheet(
                self._retrieval_indicator_stylesheet(self._DDG_BACKOFF_COLOR)
            )
        self._apply_web_indicator()
        if hasattr(self, "rag_status_dot"):
            current = getattr(self, "_rag_indicator_state", "off")
            self.set_rag_state(current)

    def showEvent(self, event):
        super().showEvent(event)
        if not getattr(self, "_window_branding_finalized", False):
            self._window_branding_finalized = True
            finalize_window_branding(self)
        if not getattr(self, "_theme_profile_startup_logged", False):
            self._theme_profile_startup_logged = True
            self._log_theme_profile_startup_snapshot()
        if not getattr(self, "_startup_geometry_finalized", False):
            self._schedule_startup_geometry()
        if not getattr(self, "_onboarding_start_scheduled", False):
            self._onboarding_start_scheduled = True
            QTimer.singleShot(900, self._maybe_start_local_llm_onboarding)
        QTimer.singleShot(1500, self.schedule_scenario_replay)

    def _log_theme_profile_startup_snapshot(self) -> None:
        from PyQt6.QtWidgets import QApplication

        from core.theme_toggle_profile import log_startup_widget_snapshot

        conversation_rows: int | None = None
        cv = getattr(self, "conversations_view", None)
        if cv is not None:
            history = getattr(cv, "history_list", None)
            if history is not None:
                conversation_rows = int(history.count())

        log_startup_widget_snapshot(
            QApplication.instance(),
            built_main_stages=len(getattr(self, "_main_stage_built", ())),
            conversation_rows=conversation_rows,
        )

    def finish_active_tour(self) -> None:
        from ui.onboarding.tour_helpers import dismiss_page_tour_transients

        tour = getattr(self, "_active_tour", None)
        if tour is not None and getattr(tour, "is_active", False):
            tour.finish()
        dismiss_page_tour_transients(self)
        self._active_tour = None

    def refresh_active_tour_layout(self) -> None:
        tour = getattr(self, "_active_tour", None)
        if tour is not None and getattr(tour, "is_active", False):
            tour.refresh_layout()

    def _start_tour(self, tour) -> None:
        if tour is None:
            return
        self.finish_active_tour()
        self._active_tour = tour
        tour.start()

    def request_page_tour(
        self,
        tour_id: str,
        *,
        area_display_name: str | None = None,
    ) -> None:
        """Build and start a registered page tour."""
        from ui.onboarding.tour_registry import build_tour

        tour = build_tour(tour_id, self)
        if tour is None:
            return
        self._start_tour(tour)

    def _maybe_start_local_llm_onboarding(self) -> None:
        if get_onboarding_local_llm_tour_completed():
            return
        if not hasattr(self, "_local_llm_tour"):
            return
        active = getattr(self, "_active_tour", None)
        if active is not None and getattr(active, "is_active", False):
            return
        self._start_tour(self._local_llm_tour)

    def start_local_llm_onboarding_tour(self) -> None:
        """Public entry to replay the local LLM setup tour."""
        if hasattr(self, "_local_llm_tour"):
            self._start_tour(self._local_llm_tour)

    def show_composer_mention_guide(self) -> None:
        """Open the scrollable @ composer guide (Settings → Help uses the same dialog)."""
        from ui.components.composer_mention_guide_dialog import show_composer_mention_guide

        show_composer_mention_guide(self, is_dark=getattr(self, "_is_dark_theme", True))

    def focus_chat_composer_if_ready(self) -> None:
        if hasattr(self, "conversations_view"):
            self.conversations_view.focus_composer_if_ready()

    def _resolve_logo_asset(self, name: str) -> Path | None:
        """Resolve logo paths across new and legacy asset directories."""
        for parts in (
            ("assets", "logos", name),
            ("assets", "icons", name),
            ("assets", name),
        ):
            candidate = resource_path(*parts)
            if candidate.is_file():
                return candidate
        return None

    def _setup_titling_connections(self):
        """Wires the background AI to the Chat UI."""
        
        # 1. When the main LLM finishes a message, check if we need a title
        if self._llm_worker:
            self._llm_worker.response_finished.connect(self._check_for_titling)

        # 2. When the sidecar finishes titling, refresh the history sidebar
        if self._sidecar_worker is not None:
            self._sidecar_worker.title_generated.connect(
                lambda s_id, title: self.conversations_view._refresh_history_list()
            )

    def _check_for_titling(self, session_id, full_response):
        """Internal logic to only title 'New Conversations'."""
        # Check history length: 1 User + 1 Assistant = 2 total messages
        history = self.db.get_session_history(session_id)
        
        if len(history) == 2:
            user_prompt = history[0].get("content") or ""
            assistant_reply = history[1].get("content") or full_response or ""
            
            if self._sidecar_client is not None:
                self._sidecar_client.enqueue_title(
                    user_prompt,
                    session_id,
                    assistant_reply=assistant_reply,
                )

    # ------------------------------------------------------------------ #
    #  UI CONSTRUCTION                                                   #
    # ------------------------------------------------------------------ #

    def _setup_ui(self) -> None:
        # Base container matching the active theme
        self.main_container = QFrame()
        self.main_container.setObjectName("MainContainer")
        self.setCentralWidget(self.main_container)
        
        root_layout = QVBoxLayout(self.main_container)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Build the Multi-Pane Layout
        self.top_bar = self._build_top_bar()
        root_layout.addWidget(self.top_bar)

        self.background_progress_banner = QWidget()
        self.background_progress_banner.setObjectName("BackgroundProgressBanner")
        banner_layout = QVBoxLayout(self.background_progress_banner)
        banner_layout.setContentsMargins(16, 6, 16, 6)
        banner_layout.setSpacing(0)
        self.background_progress_row = IngestProgressRow(self.background_progress_banner)
        banner_layout.addWidget(self.background_progress_row)
        self.background_progress_banner.hide()
        root_layout.addWidget(self.background_progress_banner)

        workspace_layout = QHBoxLayout()
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(0)

        self.nav_sidebar = self._build_nav_sidebar()
        workspace_layout.addWidget(self.nav_sidebar)

        # MAIN STAGE: The QStackedWidget Router
        self.main_stage = QStackedWidget()
        self.main_stage.setStyleSheet("background-color: transparent;")
        
        # Conversations is eager; other main stages are built on first navigation (Phase 2).
        self.conversations_view = ConversationsView(self.workers, self.workers.get("db"))
        self._library_view = None
        self._memory_manager_view = None
        self._telemetry_view = None
        self._model_manager_view = None
        self._settings_view = None
        self._settings_view_wired = False
        self._generation_toolbar_sync_wired: set[str] = set()
        self._model_manager_view_wired = False
        self._model_manager_settings_crosslink_wired = False
        self._model_manager_companion_crosslink_wired = False
        self._main_stage_built: set[int] = {MAIN_STAGE_CONVERSATIONS}
        self._main_stage_app_wirers: dict[int, list] = {
            idx: [] for idx in _LAZY_MAIN_STAGE_INDICES
        }
        self._main_stage_app_wired: set[int] = set()

        # 🔑 THE FIX: Prevent UI Stretching (Policy Ignored)
        from PyQt6.QtWidgets import QSizePolicy
        self.main_stage.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        self.main_stage.addWidget(self.conversations_view)
        for stage_index in sorted(_LAZY_MAIN_STAGE_INDICES):
            placeholder = QWidget()
            placeholder.setObjectName(f"MainStagePlaceholder_{stage_index}")
            self.main_stage.addWidget(placeholder)
            assert self.main_stage.indexOf(placeholder) == stage_index

        workspace_layout.addWidget(self.main_stage, stretch=1)
        
        # GLOBAL RIGHT TOOLBAR
        self.global_tools = self._build_tools_pane()
        workspace_layout.addWidget(self.global_tools)

        root_layout.addLayout(workspace_layout)

        self.notification_center = AppNotificationCenter(self.main_container)
        self.notification_center.action_triggered.connect(self._on_notification_action)
        self.notification_center.apply_theme(self._is_dark_theme)

        self._modal_backdrop = ModalBackdrop(self.main_container)
        self._modal_backdrop.apply_theme(self._is_dark_theme)
        self._modal_backdrop_depth = 0

        # Resize Grip
        self.grip = QSizeGrip(self.main_container) 
        self.grip.setFixedSize(16, 16)

        # --- THE SYNC WIRING (Updated for new names) ---
        # Settings ↔ toolbar sync is wired when Settings is first opened (_wire_settings_view).
        self.audio_extra_controls.setVisible(True)
        self.global_voice_selector.setVisible(True)

        # 4. Initialize Toolbar values from the worker
        if self._audio_worker:
            self.toolbar_timeout_spin.setValue(self._audio_worker.silence_timeout)
            self.toolbar_threshold_spin.setValue(int(self._audio_worker.speech_threshold))
            wakeword_threshold = float(getattr(self._audio_worker, "active_wakeword_threshold", 0.5))
            self.toolbar_wakeword_sensitivity_spin.setValue(
                max(10, min(95, int((1.0 - wakeword_threshold) * 100)))
            )
            
            # Wire Toolbar directly to worker methods
            self.toolbar_timeout_spin.valueChanged.connect(self._audio_worker.set_silence_timeout)
            self.toolbar_threshold_spin.valueChanged.connect(self._audio_worker.set_speech_threshold)
            self.toolbar_wakeword_sensitivity_spin.valueChanged.connect(
                lambda v: self._audio_worker.set_wakeword_threshold(
                    max(0.1, min(0.95, 1.0 - (float(v) / 100.0)))
                )
            )

        # 4b. Generation parameters: Settings ↔ Toolbar (wired when Settings opens)

        # 5–6. Settings / Model Manager wiring deferred until those stages are first opened.
        self.toolbar_auto_load_model_toggle.toggled.connect(
            self._on_toolbar_auto_load_model_toggle_changed
        )
        QTimer.singleShot(0, self.refresh_toolbar_native_model_dropdown)
        if self._enable_routing_debug_tool:
            self._setup_routing_debug_tool_window()
        if self._enable_trace_diff_debug_tool:
            self._setup_trace_diff_debug_window()

    def _setup_trace_diff_debug_window(self) -> None:
        from ui.canonical_trace_diff import open_canonical_trace_diff_window

        self.canonical_trace_diff_view = open_canonical_trace_diff_window(parent=self)
        self.canonical_trace_diff_view.set_scenario_hooks(
            scenario_runner=self._ui_run_scenario_serial,
            session_comparer=self._ui_compare_sessions,
            workflow_starter=self._ui_start_scenario_workflow,
        )

    def _ui_start_scenario_workflow(self, scenario_path: str, *, single_phase: bool = False) -> None:
        self._open_scenario_workflow(scenario_path, single_phase=single_phase)

    def _open_scenario_workflow(self, scenario_path: str, *, single_phase: bool | None = None) -> None:
        if single_phase is None:
            single_phase = bool(self._scenario_single_phase)

        existing = self._scenario_workflow_dialog
        if existing is not None and not existing.qube_phase_done():
            existing.show()
            existing.raise_()
            existing.activateWindow()
            self._scenario_workflow_dialog_open = True
            return

        from ui.canonical_trace_diff.scenario_workflow_dialog import (
            open_scenario_comparison_workflow,
        )
        from core.scenario_workflow import qube_pathway_ready, suggested_external_model_name

        dialog = open_scenario_comparison_workflow(
            self,
            scenario_path=scenario_path,
            repo_root=self._project_root,
            qube_ready=lambda: qube_pathway_ready(self),
            run_qube=lambda path: self._ui_run_scenario_serial(path, "qube"),
            compare_sessions=self._ui_compare_sessions,
            model_hint=lambda scenario: suggested_external_model_name(self, scenario),
            single_phase=single_phase,
        )
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dialog.qube_phase_completed.connect(self._on_scenario_qube_phase_completed)
        dialog.finished.connect(self._on_scenario_workflow_finished)
        self._scenario_workflow_dialog = dialog
        self._scenario_workflow_dialog_open = True

    def _on_scenario_qube_phase_completed(self) -> None:
        self._scenario_qube_phase_done = True

    def _on_scenario_workflow_finished(self, _result: int) -> None:
        self._scenario_workflow_dialog_open = False
        self._scenario_workflow_dialog = None

    def _ui_run_scenario_serial(self, scenario_path: str, backend: str) -> str:
        from core.conversation_replay import ConversationReplayEngine
        from core.scenario_loader import load_scenario, run_scenario_serial

        scenario = load_scenario(scenario_path)
        engine = ConversationReplayEngine(
            llm_worker=self._llm_worker if backend == "qube" else None,
            db_manager=self.db if backend == "qube" else None,
            backend=backend,  # type: ignore[arg-type]
            process_events=lambda: QApplication.processEvents(),
        )
        result = run_scenario_serial(scenario, backend, engine, log_traces=True)
        return str(result.output_path or "")

    def _ui_compare_sessions(self, path_a: str, path_b: str):
        from core.scenario_loader import compare_sessions

        return compare_sessions(path_a, path_b, save=True)

    def schedule_scenario_replay(self) -> None:
        """Queue guided scenario workflow or offline session compare after startup."""
        if self._compare_sessions and not self._scenario_replay_started:
            from PyQt6.QtCore import QTimer

            QTimer.singleShot(250, self._execute_session_compare)
            return
        if not self._run_scenario_path or self._scenario_qube_phase_done:
            return
        if self._scenario_workflow_dialog is not None:
            if not self._scenario_workflow_dialog.isVisible():
                self._scenario_workflow_dialog.show()
                self._scenario_workflow_dialog.raise_()
            return
        if self._scenario_workflow_dialog_open:
            return
        from PyQt6.QtCore import QTimer

        QTimer.singleShot(2000, self._begin_scenario_workflow)

    def _begin_scenario_workflow(self) -> None:
        if not self._run_scenario_path or self._scenario_qube_phase_done:
            return
        if self._scenario_workflow_dialog is not None:
            if not self._scenario_workflow_dialog.isVisible():
                self._scenario_workflow_dialog.show()
                self._scenario_workflow_dialog.raise_()
            return
        if self._scenario_workflow_dialog_open:
            return
        if not self.canonical_trace_diff_view:
            self._setup_trace_diff_debug_window()
        self._open_scenario_workflow(self._run_scenario_path)

    def _execute_scenario_replay(self) -> None:
        """Legacy single-backend replay (prefer guided workflow)."""
        if not self._run_scenario_path or self._scenario_replay_started:
            return
        self._scenario_replay_started = True
        path = self._run_scenario_path
        backend = self._scenario_backend if self._scenario_backend in ("qube", "external") else "qube"
        try:
            from core.conversation_replay import ConversationReplayEngine
            from core.scenario_loader import load_scenario, run_scenario_serial, session_file_path

            scenario = load_scenario(path)
            engine = ConversationReplayEngine(
                llm_worker=self._llm_worker if backend == "qube" else None,
                db_manager=self.db if backend == "qube" else None,
                backend=backend,  # type: ignore[arg-type]
                process_events=lambda: QApplication.processEvents(),
            )
            result = run_scenario_serial(scenario, backend, engine, log_traces=True)
            logger.info(
                "[ScenarioReplay] serial run %r backend=%s (%s turn(s)); log=%s",
                scenario.name,
                backend,
                len(result.session.traces),
                result.output_path,
            )
            expected = session_file_path(result.session.scenario_id, backend)
            if backend == "qube":
                logger.info(
                    "[ScenarioReplay] Next: run LM Studio with "
                    "'python3 -m tools.run_scenario_replay --scenario %s --backend external' "
                    "then compare sessions or use Compare in the diff UI.",
                    path,
                )
            self._notify_scenario_session_saved(str(result.output_path or expected))
        except Exception:
            logger.exception("[ScenarioReplay] failed for %s", path)

    def _execute_session_compare(self) -> None:
        if not self._compare_sessions or self._scenario_replay_started:
            return
        self._scenario_replay_started = True
        path_a, path_b = self._compare_sessions
        try:
            from core.scenario_loader import compare_sessions
            from ui.canonical_trace_diff import load_scenario_run_pair_view

            pair = compare_sessions(path_a, path_b, save=True)
            view = self.canonical_trace_diff_view
            if view is None:
                view = load_scenario_run_pair_view(pair, parent=self, show=True)
                self.canonical_trace_diff_view = view
            else:
                view.load_scenario_run_pair(pair)
                view.show()
                view.raise_()
            logger.info("[ScenarioReplay] compared sessions; diff ready in UI")
        except Exception:
            logger.exception("[ScenarioReplay] compare failed")

    def _notify_scenario_session_saved(self, path: str) -> None:
        if self.canonical_trace_diff_view is None:
            return
        try:
            self.canonical_trace_diff_view.set_status_message(
                f"Session saved: {path}. Run the other backend, then Compare sessions."
            )
        except Exception:
            pass

    def _setup_routing_debug_tool_window(self) -> None:
        from ui.views.routing_debug_view import RoutingDebugView

        self.routing_debug_tool_view = RoutingDebugView(
            self.workers,
            self._gpu_monitor,
            native_engine=self._native_engine,
            parent=self,
        )
        self.routing_debug_tool_view.setWindowFlag(Qt.WindowType.Window, True)
        self.routing_debug_tool_view.setWindowTitle("Qube - Routing Debug")
        self.routing_debug_tool_view.resize(1200, 800)
        self.routing_debug_tool_view.show()

    def _sync_toolbar_auto_load_model_toggle(self, checked: bool) -> None:
        t = self.toolbar_auto_load_model_toggle
        t.blockSignals(True)
        t.setChecked(checked)
        t.blockSignals(False)

    def _on_toolbar_auto_load_model_toggle_changed(self, checked: bool) -> None:
        set_auto_load_last_model_on_startup(checked)
        sv = self._settings_view
        if sv is None:
            return
        cb = sv.auto_load_last_model_cb
        cb.blockSignals(True)
        cb.setChecked(checked)
        cb.blockSignals(False)

    def moveEvent(self, event):
        super().moveEvent(event)
        if self._geometry_update_depth:
            return
        self._apply_workspace_minimum_size()

    def minimumSizeHint(self) -> QSize:
        """Match enforced minimum so Qt's first show does not shrink below the layout floor."""
        screen = self._screen_for_window() or QApplication.primaryScreen()
        return self._layout_minimum_size(screen)

    def sizeHint(self) -> QSize:
        """Prefer the design layout size on first paint (frameless windows ignore pre-show resize)."""
        screen = QApplication.primaryScreen() or self._screen_for_window()
        return self._default_window_geometry(screen).size()

    def resizeEvent(self, event):
        """Ensures the floating resize grip stays in the bottom-right corner."""
        if not self._geometry_update_depth and not self._workspace_maximized:
            clamped = event.size().expandedTo(self.minimumSize())
            if clamped != event.size():
                self._geometry_update_depth += 1
                try:
                    self.resize(clamped)
                finally:
                    self._geometry_update_depth -= 1
                return
        super().resizeEvent(event)
        if hasattr(self, 'grip'):
            # Position it at the absolute bottom-right of the container
            self.grip.move(
                self.main_container.width() - self.grip.width(),
                self.main_container.height() - self.grip.height()
            )
            # Ensure it stays on top of the sidebars
            self.grip.raise_()
        if hasattr(self, "notification_center"):
            self.notification_center.relayout()
        self.refresh_active_tour_layout()

    def _build_top_bar(self) -> QFrame:
        bar = QFrame()
        bar.setFixedHeight(45)
        bar.setObjectName("TopBar")
        
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(15, 0, 15, 0)

        # --- 1. FAR LEFT: LOGO & VU METER ---
        left_container = QWidget()
        left_container.setFixedWidth(196) # Logo + mic + VU + chevron
        left_layout = QHBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12) # 12px padding between elements

        # --- Logo Setup ---
        self.app_logo = QLabel()
        from PyQt6.QtGui import QPixmap 

        logo_path = self._resolve_logo_asset("qube_logo_256.png")
        logo_img = QPixmap(str(logo_path)) if logo_path is not None else QPixmap()
        if not logo_img.isNull():
            self.app_logo.setPixmap(logo_img.scaled(24, 24, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            self.app_logo.setText("🧊") 
            self.app_logo.setStyleSheet("font-size: 18px;")

        # Mic icon, VU meter, and chevron mic selector
        self.topbar_mic_icon = QLabel()
        mic_theme = self._theme_manager.current
        self.topbar_mic_icon.setPixmap(
            themed_fa_pixmap("fa5s.microphone", accent_icon_color(mic_theme), 14)
        )
        self.vu_meter = VUMeter()
        self.vu_meter.apply_theme(mic_theme)
        self._topbar_mic_attention_phase = 0.0
        self._topbar_mic_attention_ms = 0
        self._topbar_mic_attention_timer = QTimer(self)
        self._topbar_mic_attention_timer.setInterval(VUMeter._ATTENTION_TICK_MS)
        self._topbar_mic_attention_timer.timeout.connect(self._on_topbar_mic_attention_tick)

        self.mic_selector_btn = QToolButton()
        self.mic_selector_btn.setObjectName("TopBarMicSelector")
        self.mic_selector_btn.setAutoRaise(True)
        self.mic_selector_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.mic_selector_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.mic_selector_btn.setToolTip("Select microphone input")
        self.mic_selector_btn.setFixedSize(18, 18)
        self.mic_selector_btn.setIconSize(QSize(10, 10))

        self._setup_topbar_mic_picker_menu()
        self._apply_topbar_mic_chevron_style()

        self.topbar_mic_cluster = QWidget()
        self.topbar_mic_cluster.setObjectName("TopBarMicCluster")
        topbar_mic_cluster_layout = QHBoxLayout(self.topbar_mic_cluster)
        topbar_mic_cluster_layout.setContentsMargins(0, 0, 0, 0)
        topbar_mic_cluster_layout.setSpacing(12)
        topbar_mic_cluster_layout.addWidget(self.topbar_mic_icon)
        topbar_mic_cluster_layout.addWidget(self.vu_meter)
        topbar_mic_cluster_layout.addWidget(self.mic_selector_btn)

        left_layout.addWidget(self.app_logo)
        left_layout.addWidget(self.topbar_mic_cluster)
        left_layout.addStretch()
        
        layout.addWidget(left_container)

        layout.addStretch(1)

        # --- 2. DEAD CENTER: STATUS & RAG INDICATOR ---
        center_container = QWidget()
        center_layout = QHBoxLayout(center_container)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(10)

        # Left counterbalance to keep the bubble perfectly centered
        dummy_spacer = QWidget()
        dummy_spacer.setFixedWidth(200)
        center_layout.addWidget(dummy_spacer)

        # Status Bubble
        self.status_bubble = QLabel(" IDLE")
        self.status_bubble.setFixedSize(200, 26)
        self.status_bubble.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_bubble.setObjectName("StatusBubble")
        center_layout.addWidget(self.status_bubble)

        # 🔑 The missing RAG Indicator!
        self.rag_status_dot = QLabel("● RAG")
        self.rag_status_dot.setFixedWidth(60) 
        self.rag_status_dot.setObjectName("RagStatusDot")
        self.rag_status_dot.setToolTip(
            tr("Knowledge base status: grey = off, blue = ready, green = retrieving")
        )
        self._rag_indicator_state = "off"
        center_layout.addWidget(self.rag_status_dot)

        self.web_status_dot = QLabel("● WEB")
        self.web_status_dot.setFixedWidth(60)
        self.web_status_dot.setObjectName("WebStatusDot")
        self.web_status_dot.setToolTip("Web search is off")
        center_layout.addWidget(self.web_status_dot)

        self.hybrid_status_dot = QLabel("● HYBRID")
        self.hybrid_status_dot.setFixedWidth(85)
        self.hybrid_status_dot.setObjectName("HybridStatusDot")
        self.hybrid_status_dot.setToolTip("Hybrid Internet Mode is off")
        center_layout.addWidget(self.hybrid_status_dot)

        self.ddg_backoff_label = QLabel("⏸ DDG")
        self.ddg_backoff_label.setFixedWidth(88)
        self.ddg_backoff_label.setObjectName("DdgBackoffLabel")
        self.ddg_backoff_label.setToolTip("DuckDuckGo discovery pause")
        self.ddg_backoff_label.hide()
        center_layout.addWidget(self.ddg_backoff_label)

        self._ddg_backoff_timer = QTimer(self)
        self._ddg_backoff_timer.setInterval(1000)
        self._ddg_backoff_timer.timeout.connect(self._tick_ddg_backoff_indicator)
        self._ddg_backoff_was_active = False

        self._web_indicator_force = False
        self._web_indicator_hybrid = get_mcp_internet_hybrid_enabled()
        self._web_indicator_active = False
        self._web_indicator_active_direct = False
        self._web_indicator_active_hybrid = False
        self._web_indicator_outcome_hint: str | None = None
        self._sync_retrieval_indicator_palette()
        self.refresh_ddg_backoff_indicator()

        layout.addWidget(center_container)

        layout.addStretch(1)

        # --- 3. FAR RIGHT: WINDOW CONTROLS ---
        win_controls = QWidget()
        win_controls.setFixedWidth(180) # 🔑 Matches left_container to keep the center balanced
        win_layout = QHBoxLayout(win_controls)
        win_layout.setContentsMargins(0, 0, 0, 0)
        win_layout.setSpacing(8)
        
        win_icon_color = muted_icon_color(self._theme_manager.current)

        self._min_btn = QPushButton()
        self._min_btn.setIcon(themed_fa_icon("fa5s.minus", win_icon_color, 14))
        self._min_btn.setProperty("class", "WindowControlButton")
        self._min_btn.setToolTip(tr("Minimise window"))
        self._min_btn.clicked.connect(self.showMinimized)

        self.max_btn = QPushButton()
        self.max_btn.setIcon(
            themed_fa_icon("fa5s.expand-arrows-alt", win_icon_color, 14)
        )
        self.max_btn.setProperty("class", "WindowControlButton")
        self.max_btn.setToolTip(tr("Maximise window"))
        self.max_btn.clicked.connect(self._toggle_maximize)

        self._close_btn = QPushButton()
        self._close_btn.setIcon(themed_fa_icon("fa5s.times", win_icon_color, 14))
        self._close_btn.setProperty("class", "WindowControlButton")
        self._close_btn.setToolTip(tr("Minimise to system tray"))
        self._close_btn.clicked.connect(self.hide)

        win_layout.addStretch()
        win_layout.addWidget(self._min_btn)
        win_layout.addWidget(self.max_btn)
        win_layout.addWidget(self._close_btn)

        layout.addWidget(win_controls)
        
        return bar

    def _apply_topbar_mic_chevron_style(self) -> None:
        theme = self._theme_manager.current
        chevron_color = accent_icon_color(theme)
        hover = with_alpha(theme.text_muted, 0.18 if not theme.is_dark else 0.08)
        self.mic_selector_btn.setIcon(
            themed_fa_icon("fa5s.chevron-down", chevron_color, 10)
        )
        self.mic_selector_btn.setStyleSheet(
            f"""
            QToolButton#TopBarMicSelector {{
                background: transparent;
                border: none;
                padding: 0px;
            }}
            QToolButton#TopBarMicSelector:hover {{
                background: {hover};
                border-radius: 4px;
            }}
            QToolButton#TopBarMicSelector::menu-indicator {{
                image: none;
                width: 0px;
            }}
            """
        )

    def _short_mic_device_label(self, display_name: str) -> str:
        prefix = "Input "
        if display_name.startswith(prefix) and ": " in display_name:
            return display_name.split(": ", 1)[1]
        return display_name

    def _resolve_active_mic_device_index(self) -> int | None:
        saved = get_audio_input_device_index()
        if saved is not None:
            return saved
        worker = getattr(self, "_audio_worker", None)
        worker_idx = getattr(worker, "input_device_index", None) if worker else None
        if worker_idx is not None:
            return worker_idx
        try:
            import pyaudio

            pa = pyaudio.PyAudio()
            try:
                info = pa.get_default_input_device_info()
                return int(info.get("index"))
            finally:
                pa.terminate()
        except Exception:
            return None

    def _sync_settings_mic_selector_from_index(self, device_index: int) -> None:
        settings = self._settings_view
        if settings is None or not hasattr(settings, "mic_selector"):
            return
        for idx, name in get_input_devices():
            if idx == device_index:
                settings.mic_selector.setText(name)
                break

    def _on_topbar_mic_device_selected(self, device_index: int) -> None:
        set_audio_input_device_index(device_index)
        if self._audio_worker:
            self._audio_worker.set_input_device(device_index)
        self._sync_settings_mic_selector_from_index(device_index)

    def _setup_topbar_mic_picker_menu(self) -> None:
        from PyQt6.QtWidgets import QWidgetAction, QListWidget, QListWidgetItem

        menu = QMenu(self.mic_selector_btn)
        menu.setObjectName("PrestigeMenu")
        menu.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._topbar_mic_menu = menu
        self._apply_menu_theme(menu, getattr(self, "_is_dark_theme", True))

        list_widget = QListWidget()
        list_widget.setObjectName("PrestigeMenuList")
        list_widget.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._topbar_mic_list = list_widget

        def refresh_mic_menu() -> None:
            mics = get_input_devices()
            active_idx = self._resolve_active_mic_device_index()
            list_widget.clear()
            for idx, name in mics:
                short = self._short_mic_device_label(name)
                prefix = "✓  " if idx == active_idx else "   "
                row = QListWidgetItem(f"{prefix}{short}")
                row.setData(Qt.ItemDataRole.UserRole, idx)
                list_widget.addItem(row)

            if not mics:
                row = QListWidgetItem("No microphones found")
                row.setFlags(Qt.ItemFlag.NoItemFlags)
                list_widget.addItem(row)

            required_height = max(1, list_widget.count()) * 32 + 10
            main_win = self.window()
            max_height = int(main_win.height() * 0.5) if main_win else 400
            list_widget.setFixedHeight(min(required_height, max_height))

            content_w = list_widget.sizeHintForColumn(0) + 40
            cap = min(480, int(main_win.width() * 0.45)) if main_win else 480
            list_widget.setFixedWidth(min(cap, max(content_w, 260)))

        menu.aboutToShow.connect(refresh_mic_menu)

        def on_item_clicked(item) -> None:
            idx = item.data(Qt.ItemDataRole.UserRole)
            if idx is None:
                return
            self._on_topbar_mic_device_selected(int(idx))
            menu.hide()

        list_widget.itemClicked.connect(on_item_clicked)

        action = QWidgetAction(menu)
        action.setDefaultWidget(list_widget)
        menu.addAction(action)
        self.mic_selector_btn.setMenu(menu)
    
    def update_mic_level(self, level: float) -> None:
        """
        Updates the top bar VU meter. 
        Expects a normalized float between 0.0 (silence) and 1.0 (clipping).
        """
        if hasattr(self, 'vu_meter'):
            # Linear peak often reads low on quiet mics; curve for visibility.
            display = min(1.0, float(level) ** 0.55)
            self.vu_meter.set_level(display)

    def pulse_mic_vu_meter_attention(self, duration_ms: int = 8000) -> None:
        """Highlight the top-bar mic level meter (Settings → Audio Input hint)."""
        if hasattr(self, "vu_meter"):
            self.vu_meter.start_attention_pulse(duration_ms)
        if hasattr(self, "topbar_mic_icon"):
            self._start_topbar_mic_attention(duration_ms)
        worker = self._audio_worker
        voice_off = (
            hasattr(self, "voice_input_toggle")
            and not self.voice_input_toggle.isChecked()
        )
        if worker is not None and voice_off and hasattr(worker, "request_level_monitor"):
            worker.request_level_monitor(duration_ms / 1000.0)

    def _start_topbar_mic_attention(self, duration_ms: int) -> None:
        self._topbar_mic_attention_ms = max(500, int(duration_ms))
        if not self._topbar_mic_attention_timer.isActive():
            self._topbar_mic_attention_timer.start()
        self._on_topbar_mic_attention_tick()

    def _on_topbar_mic_attention_tick(self) -> None:
        self._topbar_mic_attention_phase += 0.25
        self._topbar_mic_attention_ms -= VUMeter._ATTENTION_TICK_MS
        if self._topbar_mic_attention_ms <= 0:
            self._topbar_mic_attention_timer.stop()
            color = accent_icon_color(self._theme_manager.current)
        else:
            pulse = 0.5 + 0.5 * math.sin(self._topbar_mic_attention_phase)
            theme = self._theme_manager.current
            color = theme.accent if pulse >= 0.5 else theme.warning
        self.topbar_mic_icon.setPixmap(
            themed_fa_pixmap("fa5s.microphone", color, 14)
        )
    
    def set_rag_state(self, state: str) -> None:
        """Manages the Traffic Light colors of the RAG indicator."""
        self._rag_indicator_state = state
        if state == 'off':
            color = self._RETRIEVAL_COLOR_OFF
        elif state == 'standby':
            color = self._RAG_COLOR_STANDBY
        elif state == 'active':
            color = self._RETRIEVAL_COLOR_ACTIVE
        else:
            return

        self.rag_status_dot.setStyleSheet(
            self._retrieval_indicator_stylesheet(color)
        )

    _RETRIEVAL_COLOR_OFF = ""
    _RETRIEVAL_COLOR_ACTIVE = ""
    _RAG_COLOR_STANDBY = ""
    _WEB_COLOR_STANDBY = ""
    _DDG_BACKOFF_COLOR = ""

    def _retrieval_indicator_stylesheet(self, color: str) -> str:
        return f"color: {color}; font-weight: bold; font-size: 11px;"

    def refresh_web_indicator(self) -> None:
        """Sync the top-bar WEB/HYBRID indicators from toolbar toggles and chat Web button."""
        self._web_indicator_force = self._resolve_web_force_enabled()
        self._web_indicator_hybrid = self._resolve_web_hybrid_enabled()
        self._apply_web_indicator()

    def set_web_indicator_active(
        self,
        active: bool,
        via_direct: bool = False,
        via_hybrid: bool = False,
    ) -> None:
        """Highlight WEB/HYBRID while an in-flight web search contributes to the turn."""
        if active:
            self._web_indicator_outcome_hint = None
        self._web_indicator_active = bool(active)
        self._web_indicator_active_direct = bool(active and via_direct)
        self._web_indicator_active_hybrid = bool(active and via_hybrid)
        self._apply_web_indicator()

    def set_web_search_outcome_hint(self, hint: str) -> None:
        """Brief tooltip after a failed web search (cleared on next active search)."""
        text = (hint or "").strip()
        self._web_indicator_outcome_hint = text or None
        self._apply_web_indicator()

    def _resolve_web_force_enabled(self) -> bool:
        cv = getattr(self, "conversations_view", None)
        if cv is not None and hasattr(cv, "web_btn"):
            return cv.web_btn.isChecked()
        worker = getattr(self, "_llm_worker", None)
        if worker is not None:
            return bool(getattr(worker, "_force_web_enabled", False))
        return False

    def _resolve_web_hybrid_enabled(self) -> bool:
        if hasattr(self, "tool_internet_hybrid_toggle"):
            return self.tool_internet_hybrid_toggle.isChecked()
        return get_mcp_internet_hybrid_enabled()

    def _apply_web_indicator(self) -> None:
        if not hasattr(self, "web_status_dot"):
            return

        force_on = bool(self._web_indicator_force)
        hybrid_on = bool(self._web_indicator_hybrid)
        active_direct = bool(getattr(self, "_web_indicator_active_direct", False))
        active_hybrid = bool(getattr(self, "_web_indicator_active_hybrid", False))
        style = self._retrieval_indicator_stylesheet

        if active_direct:
            web_color = self._RETRIEVAL_COLOR_ACTIVE
            web_tooltip = "Web search is active for this turn"
        elif active_hybrid:
            web_color = self._RETRIEVAL_COLOR_ACTIVE
            web_tooltip = "Web search is active via Hybrid mode for this turn"
        elif getattr(self, "_web_indicator_outcome_hint", None):
            web_color = self._WEB_COLOR_STANDBY
            web_tooltip = self._web_indicator_outcome_hint
        elif force_on:
            web_color = self._WEB_COLOR_STANDBY
            web_tooltip = "Web search enabled for every message in this chat"
        else:
            web_color = self._RETRIEVAL_COLOR_OFF
            web_tooltip = "Web search is off"

        self.web_status_dot.setStyleSheet(style(web_color))
        self.web_status_dot.setToolTip(web_tooltip)

        if hasattr(self, "hybrid_status_dot"):
            if active_hybrid:
                hybrid_color = self._RETRIEVAL_COLOR_ACTIVE
                hybrid_tooltip = "Hybrid Internet Mode is searching the web for this turn"
            elif hybrid_on:
                hybrid_color = self._WEB_COLOR_STANDBY
                hybrid_tooltip = (
                    "Hybrid Internet Mode: Qube automatically decides when to search the web"
                )
            else:
                hybrid_color = self._RETRIEVAL_COLOR_OFF
                hybrid_tooltip = "Hybrid Internet Mode is off"
            self.hybrid_status_dot.setStyleSheet(style(hybrid_color))
            self.hybrid_status_dot.setToolTip(hybrid_tooltip)

    def begin_ddg_backoff_tutorial_preview(self) -> None:
        """Show the DDG cooldown label with a demo countdown during the Conversations tour."""
        if not hasattr(self, "ddg_backoff_label"):
            return
        from core.knowledge.discovery.backoff import ddg_bot_backoff_seconds

        self._tour_ddg_preview_active = True
        remaining = ddg_bot_backoff_seconds()
        minutes, seconds = divmod(remaining, 60)
        total_minutes = max(1, (remaining + 59) // 60)
        self.ddg_backoff_label.setText(f"⏸ DDG {minutes}:{seconds:02d}")
        self.ddg_backoff_label.setToolTip(
            "DuckDuckGo is paused after a bot challenge. "
            f"DDG searches resume in {minutes}:{seconds:02d} "
            f"(~{total_minutes} min pause). "
            "Web searches use Brave or Wikipedia fallbacks meanwhile."
        )
        self.ddg_backoff_label.show()
        if hasattr(self, "_ddg_backoff_timer") and self._ddg_backoff_timer.isActive():
            self._ddg_backoff_timer.stop()

    def end_ddg_backoff_tutorial_preview(self) -> None:
        """Hide tour-only DDG preview and restore the real backoff indicator state."""
        if not getattr(self, "_tour_ddg_preview_active", False):
            return
        self._tour_ddg_preview_active = False
        self.refresh_ddg_backoff_indicator()

    def begin_library_chat_fab_tutorial_preview(self) -> None:
        """Show the chat-with-document FAB during the Library guided tour."""
        lv = self.ensure_library_view()
        lv._tour_chat_fab_preview_active = True
        btn = getattr(lv, "_chat_with_doc_btn", None)
        if btn is not None:
            btn.show()
            lv._reposition_chat_with_doc_fab()

    def end_library_chat_fab_tutorial_preview(self) -> None:
        """Hide tour-only chat FAB preview and restore normal visibility rules."""
        lv = getattr(self, "_library_view", None)
        if lv is None:
            return
        if not getattr(lv, "_tour_chat_fab_preview_active", False):
            return
        lv._tour_chat_fab_preview_active = False
        lv._sync_chat_with_doc_fab_visibility()

    def begin_memory_themes_tutorial_preview(self) -> None:
        """Show the recurring-themes card during the Memory Manager guided tour."""
        mv = self.ensure_memory_manager_view()
        mv._tour_themes_preview_active = True
        mv.themes_body.setText(
            "Recurring subjects across your memories appear here when enough "
            "patterns are detected."
        )
        mv.themes_card.setVisible(True)

    def end_memory_themes_tutorial_preview(self) -> None:
        """Hide tour-only themes preview and restore normal visibility rules."""
        mv = getattr(self, "_memory_manager_view", None)
        if mv is None:
            return
        if not getattr(mv, "_tour_themes_preview_active", False):
            return
        mv._tour_themes_preview_active = False
        if hasattr(mv, "_render_rows"):
            mv._render_rows()

    def refresh_ddg_backoff_indicator(self) -> None:
        """Show or hide the DDG backoff countdown in the top bar."""
        if not hasattr(self, "ddg_backoff_label"):
            return
        if getattr(self, "_tour_ddg_preview_active", False):
            return
        from core.knowledge.discovery.backoff import (
            ddg_bot_backoff_seconds,
            get_provider_backoff,
        )
        from core.knowledge.discovery.policy import PRIMARY_DISCOVERY_PROVIDER_ID

        entry = get_provider_backoff(PRIMARY_DISCOVERY_PROVIDER_ID)
        if entry is None:
            self.ddg_backoff_label.hide()
            if hasattr(self, "_ddg_backoff_timer") and self._ddg_backoff_timer.isActive():
                self._ddg_backoff_timer.stop()
            return

        remaining = entry.remaining_seconds
        minutes, seconds = divmod(remaining, 60)
        self.ddg_backoff_label.setText(f"⏸ DDG {minutes}:{seconds:02d}")
        total_minutes = max(1, (ddg_bot_backoff_seconds() + 59) // 60)
        self.ddg_backoff_label.setToolTip(
            "DuckDuckGo is paused after a bot challenge. "
            f"DDG searches resume in {minutes}:{seconds:02d} "
            f"(~{total_minutes} min pause). "
            "Web searches use Brave or Wikipedia fallbacks meanwhile."
        )
        self.ddg_backoff_label.show()
        if hasattr(self, "_ddg_backoff_timer") and not self._ddg_backoff_timer.isActive():
            self._ddg_backoff_timer.start()

    def _tick_ddg_backoff_indicator(self) -> None:
        was_active = getattr(self, "_ddg_backoff_was_active", False)
        self.refresh_ddg_backoff_indicator()
        is_active = (
            hasattr(self, "ddg_backoff_label") and self.ddg_backoff_label.isVisible()
        )
        if was_active and not is_active:
            self._sync_web_discovery_policy_section()
        self._ddg_backoff_was_active = is_active

    def on_ddg_backoff_started(self, remaining_seconds: int) -> None:
        """Toast + top-bar countdown when DDG enters a new backoff window."""
        self.refresh_ddg_backoff_indicator()
        self._ddg_backoff_was_active = True
        from core.notification_types import ddg_backoff_event

        self.emit_notification(
            ddg_backoff_event(remaining_seconds=remaining_seconds)
        )

    def on_discovery_tier_b_suggested(self) -> None:
        """Suggest optional API fallback tier after repeated DDG challenges."""
        from core.notification_types import discovery_tier_b_suggestion_event

        self.emit_notification(discovery_tier_b_suggestion_event())

    def _sync_web_discovery_policy_section(self) -> None:
        settings_view = self._settings_view
        if settings_view is None:
            return
        try:
            from ui.views.settings.sections.knowledge_web_discovery import (
                sync_web_discovery_policy_section,
            )

            sync_web_discovery_policy_section(settings_view)
        except Exception:
            pass
    
    def _screen_for_window(self) -> QScreen | None:
        """Return the monitor that contains most of the window (not always primary)."""
        center = self.frameGeometry().center()
        return QApplication.screenAt(center) or QApplication.primaryScreen()

    def _screen_available_geometry(self, screen: QScreen | None = None) -> QRect | None:
        screen = screen or self._screen_for_window()
        if screen is None:
            return None
        return workspace_bounds_for_screen(screen)

    def _layout_minimum_size(self, screen: QScreen | None = None) -> QSize:
        """Ideal layout minimum, capped when the active monitor is smaller than the design target."""
        screen = screen or self._screen_for_window()
        if screen is None:
            return self._default_minimum_size
        geo = workspace_bounds_for_screen(screen)
        return QSize(
            min(self._default_minimum_size.width(), geo.width()),
            min(self._default_minimum_size.height(), geo.height()),
        )

    def _default_window_geometry(self, screen: QScreen | None = None) -> QRect:
        """Centered startup/restored size: design target capped to the monitor."""
        screen = screen or QApplication.primaryScreen() or self._screen_for_window()
        if screen is None:
            return QRect(
                0,
                0,
                self._default_minimum_size.width(),
                self._default_minimum_size.height(),
            )
        geo = workspace_bounds_for_screen(screen)
        target = self._layout_minimum_size(screen)
        x = geo.left() + max(0, (geo.width() - target.width()) // 2)
        y = geo.top() + max(0, (geo.height() - target.height()) // 2)
        return QRect(x, y, target.width(), target.height())

    def _schedule_startup_geometry(self) -> None:
        """Frameless windows need geometry after the first show/layout pass."""
        if getattr(self, "_startup_geometry_scheduled", False):
            return
        self._startup_geometry_scheduled = True
        QTimer.singleShot(0, self._finalize_startup_geometry)
        QTimer.singleShot(120, self._finalize_startup_geometry)

    def _finalize_startup_geometry(self) -> None:
        if getattr(self, "_startup_geometry_finalized", False):
            return
        if self._workspace_maximized or self._geometry_update_depth:
            return
        screen = QApplication.primaryScreen() or self._screen_for_window()
        if screen is None:
            return
        target_rect = self._default_window_geometry(screen)
        target_size = target_rect.size()
        self._geometry_update_depth += 1
        try:
            self.setMinimumSize(target_size)
            self.setGeometry(target_rect)
            self._startup_geometry_finalized = True
        finally:
            self._geometry_update_depth -= 1

    def _restorable_minimum_size(self, screen: QScreen | None = None) -> QSize:
        """Minimum allowed while restored; must permit a visibly smaller-than-fullscreen window."""
        screen = screen or self._screen_for_window()
        geo = self._screen_available_geometry(screen)
        if geo is None:
            return QSize(_ABSOLUTE_MIN_WIDTH, _ABSOLUTE_MIN_HEIGHT)

        layout_min = self._layout_minimum_size(screen)
        if geo.width() > self._default_minimum_size.width():
            return layout_min

        return QSize(
            max(
                _ABSOLUTE_MIN_WIDTH,
                min(int(geo.width() * 0.55), geo.width() - _RESTORE_MARGIN_PX * 2),
            ),
            max(
                _ABSOLUTE_MIN_HEIGHT,
                min(int(geo.height() * 0.30), geo.height() - _RESTORE_MARGIN_PX * 2),
            ),
        )

    def _geometry_is_monitor_filling(self, rect: QRect, screen: QScreen | None = None) -> bool:
        geo = self._screen_available_geometry(screen)
        if geo is None:
            return False
        return (
            rect.width() >= geo.width() - _RESTORE_MARGIN_PX
            and rect.height() >= geo.height() - _RESTORE_MARGIN_PX
        )

    def _restore_target_geometry(self, screen: QScreen | None = None) -> QRect:
        """Compute a centered restored size that is clearly smaller than the full monitor."""
        screen = screen or self._screen_for_window()
        geo = self._screen_available_geometry(screen)
        if geo is None:
            return QRect(0, 0, self._default_minimum_size.width(), self._default_minimum_size.height())

        rest_min = self._restorable_minimum_size(screen)
        width = min(
            geo.width() - _RESTORE_MARGIN_PX * 2,
            max(rest_min.width(), int(geo.width() * _RESTORE_WIDTH_RATIO)),
        )
        height = min(
            geo.height() - _RESTORE_MARGIN_PX * 2,
            max(rest_min.height(), int(geo.height() * _RESTORE_HEIGHT_RATIO)),
        )
        x = geo.left() + (geo.width() - width) // 2
        y = geo.top() + (geo.height() - height) // 2
        return QRect(x, y, width, height)

    def _clamp_geometry_to_screen(
        self,
        rect: QRect,
        screen: QScreen | None = None,
        *,
        min_size: QSize | None = None,
    ) -> QRect:
        screen = screen or self._screen_for_window()
        geo = self._screen_available_geometry(screen)
        if geo is None:
            return rect
        floor = min_size or self._restorable_minimum_size(screen)
        width = max(floor.width(), min(rect.width(), geo.width()))
        height = max(floor.height(), min(rect.height(), geo.height()))
        x = max(geo.left(), min(rect.x(), geo.right() - width + 1))
        y = max(geo.top(), min(rect.y(), geo.bottom() - height + 1))
        return QRect(x, y, width, height)

    def _clear_platform_maximized_state(self) -> None:
        """Drop WM/Qt maximized flags so custom geometry is not immediately reverted."""
        state = self.windowState()
        if state & Qt.WindowState.WindowMaximized:
            self.showNormal()

    def _unlock_window_size(self) -> None:
        self.setMaximumSize(QSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX))

    def _apply_ui_language(self) -> None:
        """Refresh window chrome strings after the UI language setting changes."""
        self._min_btn.setToolTip(tr("Minimise window"))
        self._close_btn.setToolTip(tr("Minimise to system tray"))
        self.rag_status_dot.setToolTip(
            tr("Knowledge base status: grey = off, blue = ready, green = retrieving")
        )
        maximized = bool(self.windowState() & Qt.WindowState.WindowMaximized)
        self.max_btn.setToolTip(tr("Restore window") if maximized else tr("Maximise window"))

    def _apply_maximize_chrome(self, maximized: bool) -> None:
        win_icon_color = muted_icon_color(self._theme_manager.current)
        if maximized:
            self.max_btn.setIcon(
                themed_fa_icon("fa5s.compress-arrows-alt", win_icon_color, 14)
            )
            self.max_btn.setToolTip(tr("Restore window"))
            self.main_container.setStyleSheet(
                self.main_container.styleSheet().replace("border-radius: 12px;", "border-radius: 0px;")
            )
            if hasattr(self, "grip"):
                self.grip.setVisible(False)
        else:
            self.max_btn.setIcon(
                themed_fa_icon("fa5s.expand-arrows-alt", win_icon_color, 14)
            )
            self.max_btn.setToolTip(tr("Maximise window"))
            self.main_container.setStyleSheet(
                self.main_container.styleSheet().replace("border-radius: 0px;", "border-radius: 12px;")
            )
            if hasattr(self, "grip"):
                self.grip.setVisible(True)

    def _save_pre_maximize_geometry(self, screen: QScreen | None = None) -> None:
        screen = screen or self._screen_for_window()
        current = self.geometry()
        if screen is not None and not self._geometry_is_monitor_filling(current, screen):
            self._pre_maximize_geometry = current
        else:
            self._pre_maximize_geometry = self._restore_target_geometry(screen)

    def _maximize_to_current_screen(self) -> None:
        screen = self._screen_for_window()
        bounds = self._screen_available_geometry(screen)
        if bounds is None:
            return
        if not self._workspace_maximized:
            self._save_pre_maximize_geometry(screen)
        self._pending_restore_geometry = None
        self._geometry_update_depth += 1
        try:
            self._clear_platform_maximized_state()
            self._unlock_window_size()
            self.setMinimumSize(self._layout_minimum_size(screen))
            self.setGeometry(bounds)
            self.setFixedSize(bounds.size())
        finally:
            self._geometry_update_depth -= 1
        self._workspace_maximized = True
        self._apply_maximize_chrome(True)
        QTimer.singleShot(0, self._sync_active_view_sidebar_surfaces)

    def _sync_active_view_sidebar_surfaces(self) -> None:
        """Re-paint hub sidebars after workspace geometry changes (custom maximize)."""
        if not hasattr(self, "main_stage"):
            return
        is_dark = self._is_dark_theme
        idx = self.main_stage.currentIndex()
        if idx == MAIN_STAGE_SETTINGS:
            sv = self._settings_view
            if sv is not None and hasattr(sv, "_apply_settings_sidebar_surface"):
                sv._apply_settings_sidebar_surface(is_dark)

    def _restore_workspace_geometry(self) -> None:
        screen = self._screen_for_window()
        geo = self._screen_available_geometry(screen)
        if geo is None:
            self._workspace_maximized = False
            self._apply_maximize_chrome(False)
            return

        target = self._restore_target_geometry(screen)
        if self._pre_maximize_geometry is not None:
            candidate = self._clamp_geometry_to_screen(
                self._pre_maximize_geometry,
                screen,
                min_size=self._restorable_minimum_size(screen),
            )
            if not self._geometry_is_monitor_filling(candidate, screen):
                target = candidate

        if self._geometry_is_monitor_filling(target, screen):
            target = self._restore_target_geometry(screen)

        rest_min = self._restorable_minimum_size(screen)
        self._pending_restore_geometry = target
        self._geometry_update_depth += 1
        try:
            self._clear_platform_maximized_state()
            self._unlock_window_size()
            self.setMinimumSize(rest_min)
            self.setFixedSize(target.size())
            self.move(target.topLeft())
            self._unlock_window_size()
            self.setMinimumSize(rest_min)
            self.resize(target.size())
            self.move(target.topLeft())
        finally:
            self._geometry_update_depth -= 1

        self._workspace_maximized = False
        self._apply_maximize_chrome(False)
        QTimer.singleShot(0, self._sync_active_view_sidebar_surfaces)
        QTimer.singleShot(0, self._ensure_restored_geometry)

    def _ensure_restored_geometry(self) -> None:
        """Re-apply restore if the WM or layout ignored the first resize."""
        if self._workspace_maximized or self._pending_restore_geometry is None:
            return
        screen = self._screen_for_window()
        if screen is None:
            return
        current = self.geometry()
        target = self._pending_restore_geometry
        if (
            self._geometry_is_monitor_filling(current, screen)
            or abs(current.width() - target.width()) > 4
            or abs(current.height() - target.height()) > 4
        ):
            rest_min = self._restorable_minimum_size(screen)
            self._geometry_update_depth += 1
            try:
                self._clear_platform_maximized_state()
                self._unlock_window_size()
                self.setMinimumSize(rest_min)
                self.setFixedSize(target.size())
                self.move(target.topLeft())
                self._unlock_window_size()
                self.setMinimumSize(rest_min)
                self.resize(target.size())
                self.move(target.topLeft())
            finally:
                self._geometry_update_depth -= 1

    def _apply_workspace_minimum_size(self, screen: QScreen | None = None) -> None:
        """Keep resize limits sensible for the active monitor (restored window only)."""
        if self._workspace_maximized or self._geometry_update_depth:
            return
        screen = screen or self._screen_for_window() or QApplication.primaryScreen()
        if screen is None:
            self.setMinimumSize(self._default_minimum_size)
            return
        self.setMinimumSize(self._layout_minimum_size(screen))

    def _toggle_maximize(self):
        """Toggle fit-to-monitor vs the last normal window geometry."""
        if self._workspace_maximized:
            self._restore_workspace_geometry()
        else:
            self._maximize_to_current_screen()

    def _build_nav_sidebar(self) -> QFrame:
        """Global Left Navigation: Switches views and shows mini-telemetry."""
        sidebar = QFrame()
        sidebar.setFixedWidth(70)
        sidebar.setObjectName("NavSidebar")

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 20, 0, 20)
        layout.setSpacing(25)

        # Helper to create consistent Nav Buttons
        def create_nav_btn(icon_name=None, index=None, size=24, tooltip=None, svg_icon=None):
            btn = QPushButton()
            if svg_icon is not None:
                btn._nav_svg_icon = svg_icon
            else:
                btn._nav_fa_icon = icon_name
            btn._nav_icon_size = size
            btn.setFixedSize(44, 44)
            btn.setCheckable(True)
            btn.setProperty("class", "NavButton")
            if tooltip:
                btn.setToolTip(tooltip)
            if index is not None:
                btn.clicked.connect(lambda: self._route_view(index, btn))
            return btn

        # Top Icons
        self.nav_chat = create_nav_btn('fa5s.comment-alt', 0, tooltip="Conversations")
        self.nav_chat.setObjectName("NavChat")
        self.nav_chat.setChecked(True)

        self.nav_library = create_nav_btn('fa5s.book', 1, tooltip="Library")
        self.nav_library.setObjectName("NavLibrary")
        self.nav_memory = create_nav_btn('fa5s.memory', 2, size=22, tooltip="Memory Manager")
        self.nav_memory.setObjectName("NavMemory")
        self.nav_telemetry = create_nav_btn('fa5s.tachometer-alt', 3, tooltip="Telemetry")
        self.nav_telemetry.setObjectName("NavTelemetry")
        self.nav_models = create_nav_btn(
            index=4,
            size=24,
            tooltip="Model Manager",
            svg_icon=resource_path("assets", "icons", "ai.svg"),
        )
        self.nav_models.setObjectName("NavModels")

        layout.addWidget(self.nav_chat, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.nav_library, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.nav_memory, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.nav_models, alignment=Qt.AlignmentFlag.AlignHCenter)

        layout.addStretch()

        # Bottom Controls
        self.nav_theme = QPushButton()
        self.nav_theme.setObjectName("NavThemeToggle")
        self.nav_theme.setProperty("class", "NavButton")
        self._update_nav_theme_toggle(self._is_dark_theme)
        self.nav_theme.setIconSize(QSize(20, 20))
        self.nav_theme.setFixedSize(44, 44)
        self.nav_theme.setToolTip(self._nav_theme_tooltip(self._is_dark_theme))
        self.nav_theme.clicked.connect(self._toggle_theme)
        layout.addWidget(self.nav_theme, alignment=Qt.AlignmentFlag.AlignHCenter)

        layout.addWidget(self.nav_telemetry, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.nav_settings = create_nav_btn('fa5s.cog', 5, size=20, tooltip="Settings")
        self.nav_settings.setObjectName("NavSettings")
        layout.addWidget(self.nav_settings, alignment=Qt.AlignmentFlag.AlignHCenter)

        # --- 🔑 THE PRESTIGE MINI-TELEMETRY BLOCK ---
        tele_container = QWidget()
        tele_layout = QVBoxLayout(tele_container)
        tele_layout.setContentsMargins(0, 0, 0, 0)
        tele_layout.setSpacing(4) # Tight, elegant spacing

        # Create individual labels for specific coloring
        self.side_cpu_lbl = QLabel("CPU --")
        self.side_ram_lbl = QLabel("RAM --")
        self.side_gpu_lbl = QLabel("GPU --")

        # Style mapping: legend colors match TelemetryView
        theme = self._theme_manager.current
        metrics = [
            (self.side_cpu_lbl, telemetry_metric_colors(theme)[0]),
            (self.side_ram_lbl, telemetry_metric_colors(theme)[1]),
            (self.side_gpu_lbl, telemetry_metric_colors(theme)[2]),
        ]

        for lbl, color in metrics:
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            # 🔑 Stylized: Bold, Inter font (global), and specific legend colors
            lbl.setStyleSheet(f"""
                color: {color}; 
                font-weight: bold; 
                font-size: 10px; 
                letter-spacing: 0.5px;
            """)
            tele_layout.addWidget(lbl)

        layout.addWidget(tele_container, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.nav_buttons = [
            self.nav_chat,
            self.nav_library,
            self.nav_memory,
            self.nav_models,
            self.nav_telemetry,
            self.nav_settings,
        ]
        self._nav_active_btn = self.nav_chat
        for btn in self.nav_buttons:
            self._refresh_nav_btn_icon(btn)

        return sidebar
    
    def _build_tools_pane(self) -> QFrame:
        """Global Right Sidebar: Restored 'Card' look with animated content."""
        _TOOLS_MAIN_V_SPACING = 23
        _TOOLS_INNER_V_SPACING = 8

        # 1. THE MAIN BAR (The container with the background/border)
        self.tools_frame = QFrame()
        self.tools_frame.setObjectName("ToolsPane") 
        self.tools_frame.setFixedWidth(300) 
        self.tools_frame.setMinimumWidth(40) 
        self.tools_frame.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        
        outer_layout = QHBoxLayout(self.tools_frame)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # 2. THE HANDLE LANE (Persistent Button)
        handle_container = QWidget()
        handle_container.setFixedWidth(40)
        handle_layout = QVBoxLayout(handle_container)
        handle_layout.setContentsMargins(5, 20, 5, 0)
        
        self.toggle_tools_btn = QPushButton()
        self.toggle_tools_btn.setFixedSize(30, 30)
        theme = self._theme_manager.current
        self.toggle_tools_btn.setIcon(
            qta.icon("fa5s.chevron-right", color=theme.link)
        )
        self.toggle_tools_btn.setStyleSheet("background: transparent; border: none;")
        self.toggle_tools_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_tools_btn.setToolTip("Hide tools panel")
        self.toggle_tools_btn.clicked.connect(self._toggle_tools_pane)
        
        handle_layout.addWidget(self.toggle_tools_btn)
        handle_layout.addStretch()
        outer_layout.addWidget(handle_container)

        # 3. THE CONTENT AREA (The part that slides)
        # 🔑 Standardized Name: self.tools_content
        self.tools_content = QWidget()
        self.tools_content.setFixedWidth(260)
        self.tools_content.setMinimumWidth(0)
        
        # 🔑 FIX: Named this 'main_layout' so your section code works
        main_layout = QVBoxLayout(self.tools_content)
        main_layout.setContentsMargins(10, 18, 20, 18)
        main_layout.setSpacing(_TOOLS_MAIN_V_SPACING)

        # --- 0. LOCAL LLM (internal engine model picker) ---
        native_llm_layout = QVBoxLayout()
        native_llm_layout.setSpacing(_TOOLS_INNER_V_SPACING)
        llm_title = QLabel("LOCAL LLM")
        llm_title.setProperty("class", "ToolsPaneHeader")
        native_llm_layout.addWidget(llm_title)

        self.toolbar_native_model_selector = QPushButton()
        self.toolbar_native_model_selector.setObjectName("SettingsMenuButton")
        self.toolbar_native_model_selector.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.toolbar_native_model_selector.setIcon(
            themed_fa_icon(
                "fa5s.chevron-down",
                accent_icon_color(self._theme_manager.current),
                12,
            )
        )
        self.toolbar_native_model_selector.setMenu(QMenu(self.toolbar_native_model_selector))
        self.toolbar_native_model_selector.setText("Select AI Model")
        self.toolbar_native_model_selector.setToolTip(
            "Choose and load a local AI model (.gguf)"
        )
        self.toolbar_native_model_selector.clicked.connect(
            self._on_toolbar_native_model_selector_clicked
        )
        self._apply_native_model_selector_text_state(False)
        self.toolbar_native_model_progress = QProgressBar()
        self.toolbar_native_model_progress.setObjectName("NativeModelLoadProgress")
        self.toolbar_native_model_progress.setRange(0, 100)
        self.toolbar_native_model_progress.setValue(0)
        self.toolbar_native_model_progress.setTextVisible(False)
        self.toolbar_native_model_progress.setFixedHeight(4)
        self._set_native_model_progress_loading(False)
        native_llm_layout.addWidget(self.toolbar_native_model_progress)
        native_model_row = QHBoxLayout()
        native_model_row.setSpacing(6)
        native_model_row.addWidget(self.toolbar_native_model_selector, 1)
        self.toolbar_native_model_eject_btn = QPushButton()
        self.toolbar_native_model_eject_btn.setObjectName("NativeModelEjectButton")
        self.toolbar_native_model_eject_btn.setFixedSize(32, 32)
        self.toolbar_native_model_eject_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toolbar_native_model_eject_btn.setToolTip("Eject loaded model (free VRAM)")
        self._apply_native_model_eject_button_style()
        self.toolbar_native_model_eject_btn.clicked.connect(self._on_native_model_eject_clicked)
        native_model_row.addWidget(self.toolbar_native_model_eject_btn)
        native_llm_layout.addLayout(native_model_row)

        _auto_load_model_tip = (
            "Automatically loads the last used model at startup. This may significantly increase "
            "application startup time depending on the model size and your hardware."
        )
        _silence_cutoff_tip = (
            "How many seconds the assistant waits before deciding you have finished "
            "speaking. Lower values make the app respond faster, but it might interrupt you if "
            "you pause to think."
        )
        auto_load_row = QHBoxLayout()
        self.toolbar_auto_load_model_toggle = PrestigeToggle()
        self.toolbar_auto_load_model_toggle.setChecked(
            get_auto_load_last_model_on_startup()
        )
        self.toolbar_auto_load_model_toggle.setToolTip(_auto_load_model_tip)
        auto_load_lbl = QLabel("Load model on startup")
        auto_load_lbl.setProperty("class", "ToolsPaneControl")
        auto_load_lbl.setToolTip("")
        auto_load_info = QLabel()
        auto_load_info.setPixmap(
            qta.icon("fa5s.info-circle", color=muted_icon_color(self._theme_manager.current)).pixmap(
                QSize(12, 12)
            )
        )
        auto_load_info.setToolTip(_auto_load_model_tip)
        auto_load_info.setCursor(Qt.CursorShape.PointingHandCursor)
        auto_load_row.addWidget(self.toolbar_auto_load_model_toggle)
        auto_load_row.addWidget(auto_load_lbl)
        auto_load_row.addWidget(auto_load_info)
        auto_load_row.addStretch()
        native_llm_layout.addLayout(auto_load_row)
        main_layout.addLayout(native_llm_layout)

        # --- 1. AUDIO & TTS VOICE ---
        audio_tts_layout = QVBoxLayout()
        audio_tts_layout.setSpacing(_TOOLS_INNER_V_SPACING)
        at_title = QLabel("Audio & TTS Voice")
        at_title.setProperty("class", "ToolsPaneHeader")
        audio_tts_layout.addWidget(at_title)

        mic_row = QHBoxLayout()
        self.voice_input_toggle = PrestigeToggle()
        self.voice_input_toggle.setChecked(True)
        mic_lbl = QLabel("Enable Voice Input")
        mic_lbl.setProperty("class", "ToolsPaneControl")
        _voice_input_tip = (
            "Listen for speech and wakeword. Turn off to pause microphone capture entirely."
        )
        self.voice_input_toggle.setToolTip(_voice_input_tip)
        mic_lbl.setToolTip(_voice_input_tip)
        mic_row.addWidget(self.voice_input_toggle)
        mic_row.addWidget(mic_lbl)
        mic_row.addStretch()
        audio_tts_layout.addLayout(mic_row)

        self.audio_extra_controls = QWidget()
        extra_layout = QVBoxLayout(self.audio_extra_controls)
        extra_layout.setContentsMargins(0, 4, 0, 4)
        extra_layout.setSpacing(_TOOLS_INNER_V_SPACING)

        def create_mirrored_row(label_text, spinner, tooltip_text=None):
            row = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setProperty("class", "ToolsPaneControl")
            lbl.setMinimumWidth(0)
            if tooltip_text:
                lbl.setToolTip("")
                info_icon = QLabel()
                info_icon.setPixmap(
                    qta.icon("fa5s.info-circle", color=muted_icon_color(self._theme_manager.current)).pixmap(
                        QSize(12, 12)
                    )
                )
                info_icon.setToolTip(tooltip_text)
                info_icon.setCursor(Qt.CursorShape.PointingHandCursor)
            spinner.setFixedWidth(90)
            spinner.setProperty("class", "ToolsPaneInput")
            # Stretch on label only: extra width goes to the text so labels truncate less.
            # Icon + spinner live in a tight inner row so outer layout spacing does not sit between them.
            row.addWidget(lbl, 1)
            if tooltip_text:
                icon_input = QHBoxLayout()
                icon_input.setContentsMargins(0, 0, 0, 0)
                icon_input.setSpacing(2)
                icon_input.addWidget(info_icon, 0)
                icon_input.addWidget(spinner, 0)
                row.addLayout(icon_input, 0)
            else:
                row.addWidget(spinner, 0)
            return row

        self.toolbar_timeout_spin = NoScrollDoubleSpinBox()
        self.toolbar_timeout_spin.setRange(0.5, 5.0)
        self.toolbar_timeout_spin.setSingleStep(0.1)
        self.toolbar_timeout_spin.setSuffix(" sec")

        self.toolbar_threshold_spin = NoScrollSpinBox()
        self.toolbar_threshold_spin.setRange(1, 100)
        self.toolbar_threshold_spin.setSuffix("%")
        _vad_threshold_tip = (
            "Acts as a background noise filter which controls when normal speech is considered loud enough to keep "
            "recording/transcription active. Lower values protect against false positives."
        )
        self.toolbar_threshold_spin.setToolTip(_vad_threshold_tip)
        self.toolbar_wakeword_sensitivity_spin = NoScrollSpinBox()
        self.toolbar_wakeword_sensitivity_spin.setRange(10, 95)
        self.toolbar_wakeword_sensitivity_spin.setSuffix("%")
        _wakeword_sensitivity_tip = (
            "Controls how easily the assistant responds to your calling its name. "
            "Lower values make the assistant more responsive to calling its name, but may increase false positives. "
            "Best kept at around 50% for a balance of responsiveness and accuracy."
        )
        self.toolbar_wakeword_sensitivity_spin.setToolTip(_wakeword_sensitivity_tip)

        self.toolbar_timeout_spin.setToolTip(_silence_cutoff_tip)
        extra_layout.addLayout(
            create_mirrored_row(
                "Silence Cutoff",
                self.toolbar_timeout_spin,
                tooltip_text=_silence_cutoff_tip,
            )
        )
        extra_layout.addLayout(
            create_mirrored_row(
                "Noise Suppression",
                self.toolbar_threshold_spin,
                tooltip_text=_vad_threshold_tip,
            )
        )
        extra_layout.addLayout(
            create_mirrored_row(
                "Trigger Threshold",
                self.toolbar_wakeword_sensitivity_spin,
                tooltip_text=_wakeword_sensitivity_tip,
            )
        )

        audio_tts_layout.addWidget(self.audio_extra_controls)

        tts_row = QHBoxLayout()
        self.voice_bypass_toggle = PrestigeToggle()
        self.voice_bypass_toggle.setChecked(True)
        tts_label = QLabel("Enable TTS Voice")
        tts_label.setProperty("class", "ToolsPaneControl")
        _tts_tip = "Speak assistant responses aloud. Turn off to mute text-to-speech output."
        self.voice_bypass_toggle.setToolTip(_tts_tip)
        tts_label.setToolTip(_tts_tip)
        tts_row.addWidget(self.voice_bypass_toggle)
        tts_row.addWidget(tts_label)
        tts_row.addStretch()
        audio_tts_layout.addLayout(tts_row)

        self.global_voice_selector = QPushButton("Select Voice...")
        self.global_voice_selector.setObjectName("SettingsMenuButton")
        self.global_voice_selector.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.global_voice_selector.setIcon(
            themed_fa_icon(
                "fa5s.chevron-down",
                accent_icon_color(self._theme_manager.current),
                12,
            )
        )
        self.global_voice_selector.setMenu(QMenu(self.global_voice_selector))
        self.global_voice_selector.setToolTip("Choose text-to-speech voice")
        audio_tts_layout.addWidget(self.global_voice_selector)
        main_layout.addLayout(audio_tts_layout)

        def create_spinbox_row(label_text, tooltip_text, spinner):
            row = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setProperty("class", "ToolsPaneControl")
            lbl.setMinimumWidth(0)
            info_icon = QLabel()
            info_icon.setPixmap(
                qta.icon("fa5s.info-circle", color=muted_icon_color(self._theme_manager.current)).pixmap(
                    QSize(12, 12)
                )
            )
            info_icon.setToolTip(tooltip_text)
            info_icon.setCursor(Qt.CursorShape.PointingHandCursor)
            spinner.setToolTip(tooltip_text)
            spinner.setFixedWidth(90)
            icon_input = QHBoxLayout()
            icon_input.setContentsMargins(0, 0, 0, 0)
            icon_input.setSpacing(2)
            icon_input.addWidget(info_icon, 0)
            icon_input.addWidget(spinner, 0)
            row.addWidget(lbl, 1)
            row.addLayout(icon_input, 0)
            return row

        # --- 3. GENERATION PARAMETERS ---
        param_layout = QVBoxLayout()
        param_layout.setSpacing(_TOOLS_INNER_V_SPACING)
        p_title = QLabel("GENERATION PARAMETERS")
        p_title.setProperty("class", "ToolsPaneHeader")
        param_layout.addWidget(p_title)

        desc_temp = (
            "Creativity Slider: Lower values (0.1-0.3) produce strict, factual answers,  "
            "but will make the answers sound more robotic and less natural. "
            "Higher values (0.7-1.0) make Qube more creative and will make the answers sound more natural. "
            "It is recommended to keep the temperature around 0.7 - 0.8 for balanced performance."
        )
        desc_ctx = (
            "Total token budget per turn: instructions, chat history, your message, "
            "and the reply share one window. On the local engine this sets n_ctx "
            "(reloads the model); higher values use more RAM/VRAM. Max reply tokens "
            "are in Settings → AI — both draw from the same pool."
        )
        desc_history = (
            "How many recent user/assistant messages to include in each prompt. "
            "More history improves continuity but uses more context window space "
            "for the prompt, leaving less room for long replies. Also uses more "
            "RAM/VRAM during inference. Long-term memory still covers facts dropped "
            "from this window."
        )
        desc_max_reply = (
            "Upper bound on new tokens per assistant reply when "
            "Limit maximum reply length is on in Settings. Prompt tokens "
            "(chat history, RAG, system text) are counted first inside the "
            "context window."
        )

        self.temp_spin = NoScrollDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setProperty("class", "ToolsPaneInput")
        param_layout.addLayout(create_spinbox_row("Temperature:", desc_temp, self.temp_spin))

        self.ctx_spin = NoScrollSpinBox()
        self.ctx_spin.setRange(1024, 128000)
        self.ctx_spin.setSingleStep(256)
        self.ctx_spin.setProperty("class", "ToolsPaneInput")
        param_layout.addLayout(create_spinbox_row("Context Limit:", desc_ctx, self.ctx_spin))

        self.history_spin = NoScrollSpinBox()
        self.history_spin.setRange(2, 100)
        self.history_spin.setSingleStep(2)
        self.history_spin.setProperty("class", "ToolsPaneInput")
        param_layout.addLayout(create_spinbox_row("Chat History:", desc_history, self.history_spin))

        self.max_reply_spin = NoScrollSpinBox()
        self.max_reply_spin.setRange(256, 32768)
        self.max_reply_spin.setSingleStep(256)
        self.max_reply_spin.setProperty("class", "ToolsPaneInput")
        param_layout.addLayout(
            create_spinbox_row("Max Reply Tokens:", desc_max_reply, self.max_reply_spin)
        )
        self._apply_toolbar_generation_spin_values()

        main_layout.addLayout(param_layout)

        # --- 4. RAG ENGINE (Consolidated) ---
        rag_layout = QVBoxLayout()
        rag_layout.setSpacing(_TOOLS_INNER_V_SPACING)
        r_title = QLabel("RAG ENGINE")
        r_title.setProperty("class", "ToolsPaneHeader")
        rag_layout.addWidget(r_title)

        # 🔑 THE REFINED TOOLTIP-AWARE ROW BUILDER
        def create_toggle_row(label_text, tooltip_text, checked=False):
            row = QHBoxLayout()
            
            toggle = PrestigeToggle()
            toggle.setChecked(checked)
            toggle.setToolTip(tooltip_text)
            
            lbl = QLabel(label_text)
            lbl.setProperty("class", "ToolsPaneControl")
            lbl.setToolTip(tooltip_text)
            
            row.addWidget(toggle)
            row.addWidget(lbl)
            
            # The visual indicator icon (The ONLY thing with a tooltip now)
            info_icon = QLabel()
            info_icon.setPixmap(
                qta.icon("fa5s.info-circle", color=muted_icon_color(self._theme_manager.current)).pixmap(
                    QSize(12, 12)
                )
            )
            info_icon.setToolTip(tooltip_text)
            info_icon.setCursor(Qt.CursorShape.PointingHandCursor)
            row.addWidget(info_icon)
            
            row.addStretch()
            return row, toggle

        # 🔑 THE NEW, PUNCHIER DESCRIPTIONS
        desc_kb = "Master Switch: Grants Qube permission to read and cite your local library."
        
        # Highlighting the "Magic" and pointing them to Settings
        desc_auto = "Smart Override: Say a custom trigger to magically wake the Knowledge Base for a single turn, even if the master switch is OFF. (You can add custom 'magic words' in Settings)."
        
        desc_strict = "Lawyer Mode: Forces Qube to ONLY use your files. It will refuse to guess or use its general knowledge if the answer isn't in the documents."
        
        local_row, self.tool_rag_toggle = create_toggle_row(
            "Local Knowledge Base", desc_kb, checked=get_mcp_rag_enabled()
        )
        auto_row, self.rag_auto_toggle = create_toggle_row(
            "NLP Auto-Activator", desc_auto, checked=get_mcp_rag_auto_activator_enabled()
        )
        strict_row, self.rag_strict_toggle = create_toggle_row(
            "Strict Isolation Mode", desc_strict, checked=get_mcp_rag_strict_enabled()
        )
        
        rag_layout.addLayout(local_row)
        rag_layout.addLayout(auto_row) 
        rag_layout.addLayout(strict_row)
        main_layout.addLayout(rag_layout)

        # --- 5. INTERNET TOOLS ---
        tools_layout = QVBoxLayout()
        tools_layout.setSpacing(_TOOLS_INNER_V_SPACING)
        t_title = QLabel("INTERNET TOOLS")
        t_title.setProperty("class", "ToolsPaneHeader")
        tools_layout.addWidget(t_title)

        # 🔑 NEW: Cognitive/Hybrid Internet Mode
        desc_hybrid = "Hybrid Mode: Let Qube automatically decide when to search the internet based on context and cognitive routing."
        hybrid_row, self.tool_internet_hybrid_toggle = create_toggle_row(
            "Hybrid Internet Mode", desc_hybrid, checked=get_mcp_internet_hybrid_enabled()
        )
        tools_layout.addLayout(hybrid_row)
        main_layout.addLayout(tools_layout)

        privacy_layout = QVBoxLayout()
        privacy_layout.setSpacing(_TOOLS_INNER_V_SPACING)
        privacy_title = QLabel("Privacy")
        privacy_title.setProperty("class", "ToolsPaneHeader")
        privacy_layout.addWidget(privacy_title)

        _privacy_tier_tip = (
            "Web discovery privacy tier for @internet and Hybrid Internet Mode. "
            "Private keeps searches on DuckDuckGo and Wikipedia; higher tiers may "
            "use optional API fallbacks or a self-hosted SearXNG instance."
        )
        self.toolbar_privacy_tier_selector = SelectorButton(
            "Private",
            is_dark=getattr(self, "_is_dark_theme", True),
        )
        self.toolbar_privacy_tier_selector.setMenu(
            QMenu(self.toolbar_privacy_tier_selector)
        )
        self.toolbar_privacy_tier_selector.setToolTip(_privacy_tier_tip)
        privacy_layout.addWidget(self.toolbar_privacy_tier_selector)

        main_layout.addLayout(privacy_layout)
        self._build_toolbar_privacy_tier_menu()
        outer_layout.addWidget(self.tools_content)
        # --------------------------------------------------------- #
        #  WIRING TO WORKERS                                        #
        # --------------------------------------------------------- #
        if self._audio_worker:
            self.voice_input_toggle.toggled.connect(self._on_voice_input_toggle)
            # 🔑 Catch the volume signal and route it to the VU meter
            self._audio_worker.volume_update.connect(self.update_mic_level)

        if self._tts_worker:
            self.voice_bypass_toggle.toggled.connect(self._on_voice_bypass_toggle)
        if self._llm_worker:
            self.temp_spin.valueChanged.connect(self._llm_worker.set_temperature)
            self.ctx_spin.valueChanged.connect(self._llm_worker.set_context_window)
            self.history_spin.valueChanged.connect(self._llm_worker.set_max_history_messages)
            self.max_reply_spin.valueChanged.connect(self._llm_worker.set_output_token_limit)

            # 🔑 THE NEW RAG WIRING
            def on_rag_toggled(checked):
                if not self._guard_embedding_feature_toggle(self.tool_rag_toggle, checked):
                    return
                self.set_rag_state('standby' if checked else 'off')
                self._llm_worker.set_mcp_rag(checked)

            self.tool_rag_toggle.toggled.connect(on_rag_toggled)
            
            # Force initial state check on boot
            self.set_rag_state('standby' if self.tool_rag_toggle.isChecked() else 'off')

            # 🔑 THE NEW STRICT WIRE
            def on_strict_toggled(checked):
                if not self._guard_embedding_feature_toggle(self.rag_strict_toggle, checked):
                    return
                self._llm_worker.set_mcp_strict(checked)

            self.rag_strict_toggle.toggled.connect(on_strict_toggled)
            # 🔑 THE NEW AUTO-ACTIVATOR WIRE
            def on_auto_toggled(checked):
                if not self._guard_embedding_feature_toggle(self.rag_auto_toggle, checked):
                    return
                self._llm_worker.set_mcp_auto(checked)

            self.rag_auto_toggle.toggled.connect(on_auto_toggled)

            # Hybrid toggle controls web search + cognitive auto-web routing.
            def on_hybrid_toggled(checked: bool):
                self._llm_worker.set_mcp_internet_hybrid(checked)
                self._web_indicator_hybrid = bool(checked)
                self._apply_web_indicator()
                sv = getattr(self, "_settings_view", None)
                if sv is not None and hasattr(sv, "_sync_privacy_data_internet_hybrid_toggle"):
                    sv._sync_privacy_data_internet_hybrid_toggle()

            self.tool_internet_hybrid_toggle.toggled.connect(on_hybrid_toggled)
            # Seed worker state from the current toggle value.
            on_hybrid_toggled(self.tool_internet_hybrid_toggle.isChecked())
            self.refresh_web_indicator()

        main_layout.addStretch()
        
        # 🔑 FIX: This now matches the definition above
        outer_layout.addWidget(self.tools_content)

        return self.tools_frame

    def _generation_spin_values(self) -> tuple[float, int, int]:
        """Resolved temperature / context / history for toolbar display."""
        if self._llm_worker is not None:
            return (
                float(self._llm_worker.temperature),
                int(self._llm_worker.context_window),
                int(self._llm_worker.max_history_messages),
            )
        return (
            get_llm_temperature(),
            get_llm_context_limit(),
            get_llm_chat_history_messages(),
        )

    def _apply_toolbar_generation_spin_values(self) -> None:
        """Seed toolbar generation controls from LLMWorker / QSettings without write-back."""
        if not hasattr(self, "temp_spin"):
            return
        temp, ctx, history = self._generation_spin_values()
        for spin, value in (
            (self.temp_spin, temp),
            (self.ctx_spin, ctx),
            (self.history_spin, history),
        ):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        if hasattr(self, "max_reply_spin"):
            if self._llm_worker is not None:
                raw_limit = getattr(self._llm_worker, "output_token_limit", None)
                if isinstance(raw_limit, (int, float)) and not isinstance(raw_limit, bool):
                    max_reply = int(raw_limit)
                else:
                    max_reply = get_llm_output_token_limit()
            else:
                max_reply = get_llm_output_token_limit()
            max_reply = max(256, min(32768, max_reply))
            self.max_reply_spin.blockSignals(True)
            self.max_reply_spin.setValue(max_reply)
            self.max_reply_spin.blockSignals(False)
            self._sync_toolbar_output_limit_enabled()

    def _sync_toolbar_output_limit_enabled(self, enabled: bool | None = None) -> None:
        if not hasattr(self, "max_reply_spin"):
            return
        if enabled is None:
            sv = self._settings_view
            limit_cb = getattr(sv, "llm_output_limit_cb", None) if sv is not None else None
            if limit_cb is not None:
                enabled = limit_cb.isChecked()
            elif self._llm_worker is not None:
                enabled = bool(getattr(self._llm_worker, "output_token_limit_enabled", True))
            else:
                enabled = get_llm_output_token_limit_enabled()
        self.max_reply_spin.setEnabled(bool(enabled))

    def _wire_generation_settings_toolbar_sync(self) -> None:
        """Keep Settings and toolbar generation spinboxes aligned (audio-style sync)."""
        if not self._llm_worker:
            return
        sv = self._settings_view
        if sv is None:
            return
        pairs = (
            ("temp", self.temp_spin, getattr(sv, "llm_temp_spin", None)),
            ("ctx", self.ctx_spin, getattr(sv, "llm_ctx_spin", None)),
            ("history", self.history_spin, getattr(sv, "llm_history_spin", None)),
            ("max_reply", self.max_reply_spin, getattr(sv, "llm_output_limit_spin", None)),
        )
        for key, toolbar_spin, settings_spin in pairs:
            if toolbar_spin is None or settings_spin is None:
                continue
            if key in self._generation_toolbar_sync_wired:
                continue
            self._generation_toolbar_sync_wired.add(key)
            settings_spin.valueChanged.connect(toolbar_spin.setValue)
            toolbar_spin.valueChanged.connect(settings_spin.setValue)
            toolbar_spin.blockSignals(True)
            toolbar_spin.setValue(settings_spin.value())
            toolbar_spin.blockSignals(False)
        if (
            hasattr(sv, "llm_output_limit_cb")
            and "max_reply_enabled" not in self._generation_toolbar_sync_wired
        ):
            self._generation_toolbar_sync_wired.add("max_reply_enabled")
            sv.llm_output_limit_cb.toggled.connect(self._sync_toolbar_output_limit_enabled)
        self._sync_toolbar_output_limit_enabled()

    def _build_toolbar_privacy_tier_menu(self) -> None:
        if not hasattr(self, "toolbar_privacy_tier_selector"):
            return
        from core.knowledge.discovery.privacy_policy import (
            TIER_BALANCED,
            TIER_ENHANCED,
            TIER_PRIVATE,
            TIER_SEARXNG,
            privacy_tier_description,
            privacy_tier_label,
        )

        items = [
            (
                f"{privacy_tier_label(tier)} — {privacy_tier_description(tier)}",
                tier,
            )
            for tier in (TIER_PRIVATE, TIER_BALANCED, TIER_ENHANCED, TIER_SEARXNG)
        ]
        self._build_prestige_menu(
            self.toolbar_privacy_tier_selector,
            items,
            self._on_toolbar_privacy_tier_selected,
        )
        self._sync_toolbar_privacy_tier_selector()

    def _on_toolbar_privacy_tier_selected(self, tier: str) -> None:
        from core.app_settings import set_discovery_privacy_tier

        if not tier:
            return
        set_discovery_privacy_tier(str(tier))
        self._sync_toolbar_privacy_tier_selector()
        sv = self._settings_view
        if sv is not None:
            from ui.views.settings.sections.knowledge_web_discovery import (
                sync_web_discovery_policy_section,
            )

            sync_web_discovery_policy_section(sv)

    def _sync_toolbar_privacy_tier_selector(self) -> None:
        if not hasattr(self, "toolbar_privacy_tier_selector"):
            return
        from core.app_settings import get_discovery_privacy_tier
        from core.knowledge.discovery.privacy_policy import privacy_tier_label

        self.toolbar_privacy_tier_selector.setText(
            privacy_tier_label(get_discovery_privacy_tier())
        )
        self.toolbar_privacy_tier_selector.update()

    def _wire_toolbar_internet_settings_sync(self, sv) -> None:
        if hasattr(sv, "_on_discovery_privacy_tier_selected"):
            original_privacy = sv._on_discovery_privacy_tier_selected

            def wrapped_privacy(tier: str) -> None:
                original_privacy(tier)
                self._sync_toolbar_privacy_tier_selector()

            sv._on_discovery_privacy_tier_selected = wrapped_privacy
            if hasattr(sv, "_build_privacy_tier_menus"):
                sv._build_privacy_tier_menus()
            elif hasattr(sv, "_build_discovery_privacy_tier_menu"):
                sv._build_discovery_privacy_tier_menu()

    def acquire_modal_backdrop(self) -> None:
        """Dim the main window while a modal dialog is open (reference-counted)."""
        self._modal_backdrop_depth += 1
        if self._modal_backdrop_depth == 1:
            self._modal_backdrop.apply_theme(self._is_dark_theme)
            self._modal_backdrop.show_animated()

    def release_modal_backdrop(self) -> None:
        """Remove one modal dim layer; hides when the last dialog closes."""
        if self._modal_backdrop_depth <= 0:
            return
        self._modal_backdrop_depth -= 1
        if self._modal_backdrop_depth == 0:
            self._modal_backdrop.hide_animated()

    def _is_tools_pane_collapsed(self) -> bool:
        return self.tools_content.maximumWidth() == 0

    def _set_tools_pane_expanded(self, expanded: bool, *, animate: bool = True) -> None:
        """Show or hide the global tools panel content area."""
        if self._is_tools_pane_collapsed() == (not expanded):
            return

        end_content = 260 if expanded else 0
        end_frame = 300 if expanded else 40

        if expanded:
            icon = qta.icon("fa5s.chevron-right", color=self._theme_manager.current.link)
            tooltip = "Hide tools panel"
        else:
            icon = qta.icon("fa5s.chevron-left", color=self._theme_manager.current.link)
            tooltip = "Show tools panel"

        if animate:
            self.content_anim = QPropertyAnimation(self.tools_content, b"maximumWidth")
            self.content_anim.setDuration(350)
            self.content_anim.setEasingCurve(QEasingCurve.Type.InOutQuart)
            self.content_anim.setEndValue(end_content)

            self.frame_anim = QPropertyAnimation(self.tools_frame, b"maximumWidth")
            self.frame_anim.setDuration(350)
            self.frame_anim.setEasingCurve(QEasingCurve.Type.InOutQuart)
            self.frame_anim.setEndValue(end_frame)

            self.content_anim.start()
            self.frame_anim.start()
        else:
            self.tools_content.setMaximumWidth(end_content)
            self.tools_frame.setMaximumWidth(end_frame)

        self.toggle_tools_btn.setIcon(icon)
        self.toggle_tools_btn.setToolTip(tooltip)

    def _toggle_tools_pane(self):
        """Animates the collapse of the content while keeping the handle visible."""
        self._set_tools_pane_expanded(self._is_tools_pane_collapsed(), animate=True)

    def _open_model_manager_page(self) -> None:
        self._restore_workspace_from_tray()
        if hasattr(self, "nav_models"):
            self.nav_models.setChecked(True)
            self._route_view(4, self.nav_models)

    def _open_local_model_picker_from_toolbar(self) -> None:
        """Expand the tools pane and open the toolbar Select AI Model menu."""
        self._restore_workspace_from_tray()
        if get_engine_mode() != "internal" or not list_local_gguf_menu_entries():
            self._open_model_manager_page()
            return

        need_expand = self._is_tools_pane_collapsed()
        if need_expand:
            self._set_tools_pane_expanded(True, animate=True)

        def _popup_menu() -> None:
            btn = getattr(self, "toolbar_native_model_selector", None)
            if btn is None or not btn.isEnabled():
                self._open_model_manager_page()
                return
            if btn.menu() is None:
                self.refresh_toolbar_native_model_dropdown()
            menu = btn.menu()
            if menu is None or menu.isEmpty():
                self._open_model_manager_page()
                return
            btn.setFocus(Qt.FocusReason.OtherFocusReason)
            btn.showMenu()

        QTimer.singleShot(380 if need_expand else 0, _popup_menu)

    def _refresh_toolbar_native_model_from_settings_signal(self, mode: str) -> None:
        """Uses the value from Settings' Inference engine menu (authoritative for this UI tick)."""
        self.refresh_toolbar_native_model_dropdown(mode)

    def _on_toolbar_native_model_selector_clicked(self) -> None:
        """When the local library is empty, guide the user to Model Manager."""
        if get_engine_mode() != "internal":
            return
        if list_local_gguf_menu_entries():
            return
        self._show_no_local_models_dialog()

    def _show_no_local_models_dialog(self) -> None:
        is_dark = getattr(self, "_is_dark_theme", True)
        if PrestigeDialog(
            self,
            "No models found",
            "No local .gguf models were detected on this device.\n\n"
            "Open Model Manager to browse Qube Verified models, download one, "
            "then return here and pick it from Select AI Model.",
            is_dark=is_dark,
            confirm_text="OPEN MODEL MANAGER",
        ).exec():
            self._on_notification_action("open_models")

    def _apply_settings_menu_button_chevron_state(self, button: QPushButton) -> None:
        """QtAwesome icons ignore QSS; match chevron to menu button enabled/disabled look."""
        if isinstance(button, SelectorButton):
            button.apply_theme(getattr(self, "_is_dark_theme", True))
            return
        is_dark = getattr(self, "_is_dark_theme", True)
        theme = resolve_shell_theme(self, is_dark=is_dark)
        color = (
            accent_icon_color(theme)
            if button.isEnabled()
            else chevron_colors(theme, enabled=False)
        )
        button.setIcon(themed_fa_icon("fa5s.chevron-down", color, 12))

    def refresh_toolbar_native_model_dropdown(self, mode: str | None = None) -> None:
        """Toolbar picker for internal .gguf models: mirrors engine mode and downloads folder.

        When *mode* is omitted, reads persisted engine mode (e.g. after model downloads).
        When *mode* is passed (from ``engine_mode_changed``), use it so the toolbar matches
        the user's selection even if other slots have not persisted yet.
        """
        if not hasattr(self, "toolbar_native_model_selector"):
            return
        btn = self.toolbar_native_model_selector
        try:
            if mode is not None:
                m = str(mode).lower().strip()
                if m not in ("external", "internal"):
                    m = get_engine_mode()
            else:
                m = get_engine_mode()

            if m == "external":
                self._pending_native_model_path = None
                self._native_model_loading = False
                self._native_model_loaded_success = False
                self._set_native_model_progress_loading(False)
                btn.setEnabled(False)
                btn.setText("Inactive — External Server")
                btn.setToolTip(
                    "Local model selection is disabled while AI Engine is set to External Server. "
                    "Open Settings → AI Engine → Internal Engine (native) to use on-device .gguf models."
                )
                self._apply_native_model_selector_text_state(False, inactive=True)
                btn.setMenu(None)
                return

            btn.setEnabled(True)
            models_dir = Path(get_llm_models_dir())
            try:
                ggufs = sorted(
                    (p for p in models_dir.glob("*.gguf") if not is_secondary_gguf_shard(str(p))),
                    key=local_gguf_sort_key,
                )
            except OSError:
                ggufs = []

            fm = QFontMetrics(btn.font())
            cap_btn = max(100, btn.width() - 56)
            if btn.width() <= 1:
                cap_btn = max(100, self.tools_content.width() - 56)

            if not ggufs:
                self._pending_native_model_path = None
                self._native_model_loading = False
                self._native_model_loaded_success = False
                self._set_native_model_progress_loading(False)
                btn.setText("Select AI Model")
                btn.setToolTip(
                    "No local .gguf models found. Click to open Model Manager and download one."
                )
                self._apply_native_model_selector_text_state(False)
                btn.setMenu(None)
                return

            def _elide_button_label(path: str) -> str:
                display = format_local_gguf_display(path, models_dir=models_dir)
                return fm.elidedText(
                    display.button_label, Qt.TextElideMode.ElideMiddle, cap_btn
                )

            def on_pick(path: str) -> None:
                self.load_native_model_from_path(path)
                # Keep optimistic label; final state is resolved by load_finished.

            items = []
            for p in ggufs:
                abs_p = str(p.resolve())
                display = format_local_gguf_display(str(p), models_dir=models_dir)
                items.append((display.menu_label, abs_p))

            self._build_prestige_menu(
                btn,
                items,
                on_pick,
                menu_width="fit_content",
                min_menu_width=280,
            )

            if self._native_model_loading and self._pending_native_model_path:
                btn.setText(_elide_button_label(self._pending_native_model_path))
                pending_display = format_local_gguf_display(
                    self._pending_native_model_path, models_dir=models_dir
                )
                btn.setToolTip(pending_display.tooltip)
                self._apply_native_model_selector_text_state(False)
                return

            if self._native_model_unloading:
                snap = self._native_engine.get_model_reasoning_telemetry() if self._native_engine else None
                loaded_name = str((snap or {}).get("model_basename") or "").strip()
                if loaded_name:
                    matched = next((p for p in ggufs if p.name == loaded_name), None)
                    if matched is not None:
                        unloading_display = format_local_gguf_display(
                            str(matched), models_dir=models_dir
                        )
                        btn.setText(
                            fm.elidedText(
                                unloading_display.button_label,
                                Qt.TextElideMode.ElideMiddle,
                                cap_btn,
                            )
                        )
                        btn.setToolTip("Ejecting model from memory…")
                        self._apply_native_model_selector_text_state(False)
                        return

            snap = self._native_engine.get_model_reasoning_telemetry() if self._native_engine else None
            loaded = bool((snap or {}).get("loaded"))
            loaded_name = str((snap or {}).get("model_basename") or "").strip()
            matched: Path | None = None
            if loaded and loaded_name:
                matched = next((p for p in ggufs if p.name == loaded_name), None)

            if loaded and matched is not None:
                loaded_display = format_local_gguf_display(
                    str(matched), models_dir=models_dir
                )
                btn.setText(
                    fm.elidedText(
                        loaded_display.button_label,
                        Qt.TextElideMode.ElideMiddle,
                        cap_btn,
                    )
                )
                btn.setToolTip(loaded_display.tooltip)
                self._apply_native_model_selector_text_state(self._native_model_loaded_success)
            else:
                btn.setText(fm.elidedText("Select AI Model", Qt.TextElideMode.ElideMiddle, cap_btn))
                btn.setToolTip("")
                self._apply_native_model_selector_text_state(False)
        finally:
            self._apply_settings_menu_button_chevron_state(btn)
            self._sync_native_model_eject_button()
            if self._settings_view is not None and hasattr(
                self._settings_view, "sync_active_native_model_label"
            ):
                self._settings_view.sync_active_native_model_label()

    def _apply_native_model_eject_button_style(self) -> None:
        if not hasattr(self, "toolbar_native_model_eject_btn"):
            return
        btn = self.toolbar_native_model_eject_btn
        theme = self._theme_manager.current
        hover = with_alpha(theme.accent, 0.12)
        btn.setStyleSheet(
            f"""
            QPushButton#NativeModelEjectButton {{
                background: transparent;
                border: none;
                border-radius: 6px;
            }}
            QPushButton#NativeModelEjectButton:hover:enabled {{
                background: {hover};
            }}
            QPushButton#NativeModelEjectButton:disabled {{
                background: transparent;
            }}
            """
        )
        color = accent_icon_color(theme) if btn.isEnabled() else chevron_colors(theme, enabled=False)
        btn.setIcon(qta.icon("fa5s.eject", color=color))

    def _sync_native_model_eject_button(self) -> None:
        if not hasattr(self, "toolbar_native_model_eject_btn"):
            return
        btn = self.toolbar_native_model_eject_btn
        if get_engine_mode() == "external":
            btn.setEnabled(False)
            btn.setToolTip(
                "Local model eject is unavailable while AI Engine is set to External Server."
            )
            self._apply_native_model_eject_button_style()
            return
        if self._native_model_loading:
            btn.setEnabled(False)
            btn.setToolTip("Wait until the current model finishes loading.")
            self._apply_native_model_eject_button_style()
            return
        if self._native_model_unloading:
            btn.setEnabled(False)
            btn.setToolTip("Ejecting model from memory…")
            self._apply_native_model_eject_button_style()
            return
        snap = self._native_engine.get_model_reasoning_telemetry() if self._native_engine else None
        loaded = bool((snap or {}).get("loaded"))
        btn.setEnabled(loaded)
        btn.setToolTip(
            "Eject loaded model (free VRAM)"
            if loaded
            else "No model is loaded in memory."
        )
        self._apply_native_model_eject_button_style()

    def _on_native_model_eject_clicked(self) -> None:
        if not self._llm_worker:
            return
        cv = getattr(self, "conversations_view", None)
        if cv is not None and hasattr(cv, "interrupt_active_response"):
            cv.interrupt_active_response()
        self._native_model_unloading = True
        self._native_model_loaded_success = False
        self._set_native_model_progress_loading(True)
        self._sync_native_model_eject_button()
        self._llm_worker.eject_loaded_native_model()

    def _on_native_engine_status_update(self, message: str) -> None:
        msg = str(message or "").strip()
        if msg == "Loading native model…":
            if not self._native_model_unloading:
                self._native_model_loading = True
                self._set_native_model_progress_loading(True)
                self._sync_native_model_eject_button()
            return
        if msg == "Unloading native model…":
            self._native_model_unloading = True
            self._native_model_loading = False
            self._set_native_model_progress_loading(True)
            self._sync_native_model_eject_button()
            return
        if msg == "Native model unloaded":
            if self._native_model_loading:
                return
            self._on_native_model_ejected_ui()

    def _on_native_model_ejected_ui(self) -> None:
        self._pending_native_model_path = None
        self._native_model_loading = False
        self._native_model_unloading = False
        self._native_model_loaded_success = False
        self._set_native_model_progress_loading(False)
        self.refresh_toolbar_native_model_dropdown()
        cv = getattr(self, "conversations_view", None)
        if cv is not None and hasattr(cv, "refresh_think_toggle"):
            cv.refresh_think_toggle()

    def _set_native_model_progress_loading(self, loading: bool) -> None:
        if not hasattr(self, "toolbar_native_model_progress"):
            return
        bar = self.toolbar_native_model_progress
        theme = self._theme_manager.current
        if loading:
            bar.setRange(0, 0)
            track = vu_meter_palette(theme)["progress_track"]
            bar.setStyleSheet(
                f"""
                QProgressBar {{
                    background: {track};
                    border: none;
                    border-radius: 2px;
                }}
                QProgressBar::chunk {{
                    background-color: {theme.accent};
                    border-radius: 2px;
                }}
                """
            )
        else:
            bar.setRange(0, 100)
            bar.setValue(0)
            # Keep spacer height without visible fill.
            bar.setStyleSheet(
                """
                QProgressBar {
                    background: transparent;
                    border: none;
                    border-radius: 2px;
                }
                QProgressBar::chunk {
                    background: transparent;
                    border: none;
                }
                """
            )

    def _apply_native_model_selector_text_state(
        self, success: bool, *, inactive: bool = False
    ) -> None:
        if not hasattr(self, "toolbar_native_model_selector"):
            return
        btn = self.toolbar_native_model_selector
        theme = self._theme_manager.current
        if inactive:
            btn.setStyleSheet(f"color: {muted_icon_color(theme)}; font-style: italic;")
        elif success:
            btn.setStyleSheet(
                f"color: {theme.color(SUCCESS_STATUS)}; font-weight: 600;"
            )
        else:
            btn.setStyleSheet("")

    def _on_native_model_load_finished_ui(self, ok: bool, message: str) -> None:
        stale_ignored = False
        if self._native_model_loading and self._pending_native_model_path:
            pending_name = Path(self._pending_native_model_path).name
            # Ignore stale completion from an older rapid selection.
            if ok and str(message or "").strip() and str(message).strip() != pending_name:
                stale_ignored = True
        if stale_ignored:
            return
        self._native_model_loading = False
        self._native_model_unloading = False
        self._native_model_loaded_success = bool(ok)
        self._pending_native_model_path = None
        self._set_native_model_progress_loading(False)
        if not ok and "missing model shards" in str(message or "").lower():
            is_dark = getattr(self, "_is_dark_theme", True)
            PrestigeDialog(
                self,
                "Missing model shards",
                "This GGUF model is split into multiple shard files and some parts are missing.\n\n"
                f"{str(message or '').strip()}",
                is_dark=is_dark,
            ).exec()
        self.refresh_toolbar_native_model_dropdown()
        if ok and self._run_scenario_path and not self._scenario_qube_phase_done:
            self.schedule_scenario_replay()

    # --- PRESTIGE MENU LOGIC ---
    def _build_prestige_menu(
        self,
        button,
        items,
        callback,
        *,
        menu_width: str = "match_button",
        min_menu_width: int = 220,
    ):
        """Builds a palette-forced QMenu with a dynamic, scrollable list."""
        from PyQt6.QtWidgets import QMenu, QWidgetAction, QListWidget, QListWidgetItem
        from PyQt6.QtCore import Qt

        fit_content = menu_width == "fit_content"

        menu = QMenu(button)
        menu.setObjectName("PrestigeMenu")
        # The Magic Line:
        menu.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        # Apply the theme palette
        is_dark = self._is_dark_theme if hasattr(self, '_is_dark_theme') else getattr(self.window(), '_is_dark_theme', True)
        self._apply_menu_theme(menu, is_dark)

        # 1. Create the Scrollable List
        list_widget = QListWidget()
        list_widget.setObjectName("PrestigeMenuList")
        list_widget.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        
        # --- BUG 2 FIX: Kill the phantom horizontal scrollbar ---
        list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # 2. Populate the List (UserRole holds payload so elided labels stay unambiguous)
        for label, data in items:
            row = QListWidgetItem(label)
            row.setData(Qt.ItemDataRole.UserRole, data)
            list_widget.addItem(row)
            
        # 3. Dynamic Height Calculation
        required_height = len(items) * 32 + 10 
        main_win = self.window()
        max_height = int(main_win.height() * 0.5) if main_win else 400
        list_widget.setFixedHeight(min(required_height, max_height))

        # --- BUG 1 FIX: Just-In-Time Sizing ---
        # This recalculates the exact width a millisecond before the popup opens.
        def sync_dropdown_width():
            if fit_content:
                content_w = list_widget.sizeHintForColumn(0) + 40
                cap = 480
                if main_win:
                    cap = min(480, int(main_win.width() * 0.45))
                w = min(cap, max(button.width() - 8, content_w, min_menu_width))
                list_widget.setFixedWidth(w)
                return

            # button.width() gets the actual drawn size.
            # We subtract 8px to account for the 4px CSS padding on each side of the QMenu.
            w = button.width() - 8
            list_widget.setFixedWidth(w)
            # Re-elide file rows (e.g. .gguf paths) to match the live list width
            fm = list_widget.fontMetrics()
            elide_w = max(40, w - 40)
            for i in range(list_widget.count()):
                it = list_widget.item(i)
                data = it.data(Qt.ItemDataRole.UserRole)
                if isinstance(data, str) and data.lower().endswith(".gguf"):
                    it.setText(
                        fm.elidedText(Path(data).name, Qt.TextElideMode.ElideMiddle, elide_w)
                    )

        menu.aboutToShow.connect(sync_dropdown_width)

        # 4. Handle Selection
        def on_item_clicked(item):
            selected_label = item.text()
            matched_data = item.data(Qt.ItemDataRole.UserRole)
            if matched_data is None:
                matched_data = next((d for l, d in items if l == selected_label), selected_label)
            self._handle_selection(button, selected_label, matched_data, callback)
            menu.hide()

        list_widget.itemClicked.connect(on_item_clicked)

        # 5. Embed the List into the Menu
        action = QWidgetAction(menu)
        action.setDefaultWidget(list_widget)
        menu.addAction(action)

        button.setMenu(menu)

    def _apply_menu_theme(self, menu, is_dark: bool):
        apply_prestige_menu_theme(menu, resolve_shell_theme(self, is_dark=is_dark))

    def _handle_selection(self, button, label, data, callback):
        button.setText(label)
        callback(data)

    def _refresh_toolbar_control_icons(self, theme=None) -> None:
        if theme is None:
            theme = self._theme_manager.current
        accent = accent_icon_color(theme)
        muted = muted_icon_color(theme)
        if hasattr(self, "_min_btn"):
            self._min_btn.setIcon(themed_fa_icon("fa5s.minus", muted, 14))
        if hasattr(self, "_close_btn"):
            self._close_btn.setIcon(themed_fa_icon("fa5s.times", muted, 14))
        if hasattr(self, "max_btn"):
            maximized = bool(self.windowState() & Qt.WindowState.WindowMaximized)
            max_icon = "fa5s.compress-arrows-alt" if maximized else "fa5s.expand-arrows-alt"
            self.max_btn.setIcon(themed_fa_icon(max_icon, muted, 14))
        if hasattr(self, "toolbar_native_model_selector") and not isinstance(
            self.toolbar_native_model_selector, SelectorButton
        ):
            self.toolbar_native_model_selector.setIcon(
                themed_fa_icon("fa5s.chevron-down", accent, 12)
            )
        if hasattr(self, "global_voice_selector") and not isinstance(
            self.global_voice_selector, SelectorButton
        ):
            self.global_voice_selector.setIcon(
                themed_fa_icon("fa5s.chevron-down", accent, 12)
            )

    def _nav_icon_colors(self) -> tuple[str, str]:
        return nav_icon_colors(self._theme_manager.current)

    def _refresh_nav_btn_icon(self, btn: QPushButton) -> None:
        size = getattr(btn, "_nav_icon_size", 24)
        active_color, inactive_color = self._nav_icon_colors()
        color = active_color if btn.isChecked() else inactive_color
        svg_path = getattr(btn, "_nav_svg_icon", None)
        if svg_path is not None:
            btn.setIcon(tinted_svg_icon(str(svg_path), color, size))
            btn.setIconSize(QSize(size, size))
            return
        icon_name = getattr(btn, "_nav_fa_icon", None)
        if not icon_name:
            return
        btn.setIcon(themed_fa_icon(icon_name, color, size))
        btn.setIconSize(QSize(size, size))

    def _is_main_stage_built(self, index: int) -> bool:
        return index in getattr(self, "_main_stage_built", ())

    def peek_library_view(self) -> LibraryView | None:
        return self._library_view

    def peek_memory_manager_view(self) -> MemoryManagerView | None:
        return self._memory_manager_view

    def peek_telemetry_view(self) -> TelemetryView | None:
        return self._telemetry_view

    def peek_model_manager_view(self) -> ModelManagerView | None:
        return self._model_manager_view

    def peek_settings_view(self) -> SettingsView | None:
        return self._settings_view

    def ensure_library_view(self) -> LibraryView:
        return self._ensure_main_stage_view(MAIN_STAGE_LIBRARY)

    def ensure_memory_manager_view(self) -> MemoryManagerView:
        return self._ensure_main_stage_view(MAIN_STAGE_MEMORY)

    def ensure_telemetry_view(self) -> TelemetryView:
        return self._ensure_main_stage_view(MAIN_STAGE_TELEMETRY)

    def open_telemetry_focus(self, focus: str) -> None:
        """Navigate to Telemetry and scroll to a dashboard section."""
        from PyQt6.QtCore import QTimer

        tv = self.ensure_telemetry_view()
        cv = getattr(self, "conversations_view", None)
        if cv is not None:
            session_id = getattr(cv, "active_session_id", None)
            if session_id is not None:
                tv.set_active_session_id(str(session_id))
        self._route_view(MAIN_STAGE_TELEMETRY, self.nav_telemetry)

        target = None
        if focus == "web_discovery":
            target = getattr(tv, "discovery_card", None)
        elif focus == "session_integrations":
            target = getattr(tv, "session_egress_panel", None)
        if target is not None:
            QTimer.singleShot(120, lambda widget=target: tv.scroll_to_widget(widget))

    def ensure_model_manager_view(self) -> ModelManagerView:
        return self._ensure_main_stage_view(MAIN_STAGE_MODEL_MANAGER)

    def ensure_settings_view(self) -> SettingsView:
        return self._ensure_main_stage_view(MAIN_STAGE_SETTINGS)

    def register_main_stage_app_wirer(self, stage_index: int, wirer) -> None:
        """Register app-level signal wiring to run when a lazy stage is first built."""
        if stage_index not in _LAZY_MAIN_STAGE_INDICES:
            raise ValueError(f"Stage {stage_index} is not lazy-loaded")
        self._main_stage_app_wirers.setdefault(stage_index, []).append(wirer)
        if self._is_main_stage_built(stage_index):
            self._flush_main_stage_app_wirers(stage_index)

    def _flush_main_stage_app_wirers(self, stage_index: int) -> None:
        if stage_index in self._main_stage_app_wired:
            return
        self._main_stage_app_wired.add(stage_index)
        for wirer in self._main_stage_app_wirers.get(stage_index, ()):
            wirer()
        self._maybe_wire_lazy_stage_crosslinks()

    def _maybe_wire_lazy_stage_crosslinks(self) -> None:
        mm = self._model_manager_view
        sv = self._settings_view
        if (
            mm is not None
            and sv is not None
            and not self._model_manager_settings_crosslink_wired
            and hasattr(mm, "native_library_changed")
            and hasattr(sv, "refresh_native_local_library")
        ):
            mm.native_library_changed.connect(sv.refresh_native_local_library)
            self._model_manager_settings_crosslink_wired = True

        if (
            mm is not None
            and self._companion_controller is not None
            and not self._model_manager_companion_crosslink_wired
            and hasattr(mm, "download_succeeded")
        ):
            mm.download_succeeded.connect(
                self._companion_controller.on_model_download_complete
            )
            self._model_manager_companion_crosslink_wired = True

    def _create_main_stage_view(self, index: int) -> QWidget:
        if index == MAIN_STAGE_LIBRARY:
            self._library_view = LibraryView(self.workers, self.workers.get("db"))
            return self._library_view
        if index == MAIN_STAGE_MEMORY:
            self._memory_manager_view = MemoryManagerView(
                self.workers,
                self.workers.get("db"),
            )
            return self._memory_manager_view
        if index == MAIN_STAGE_TELEMETRY:
            self._telemetry_view = TelemetryView(
                self.workers,
                self._gpu_monitor,
                native_engine=self._native_engine,
            )
            return self._telemetry_view
        if index == MAIN_STAGE_MODEL_MANAGER:
            self._model_manager_view = ModelManagerView(
                self.workers,
                self.workers.get("db"),
            )
            self._wire_model_manager_view()
            return self._model_manager_view
        if index == MAIN_STAGE_SETTINGS:
            self._settings_view = SettingsView(
                self.workers,
                self.workers.get("db"),
                parent=self,
            )
            self._wire_settings_view()
            return self._settings_view
        raise ValueError(f"Unknown main stage index: {index}")

    def _wire_settings_view(self) -> None:
        if self._settings_view_wired:
            return
        sv = self._settings_view
        if sv is None:
            return
        self._settings_view_wired = True

        sv.audio_pin_toggle.connect(self.audio_extra_controls.setVisible)
        self.audio_extra_controls.setVisible(sv.pin_audio_cb.isChecked())

        sv.tts_voice_pin_toggle.connect(self.global_voice_selector.setVisible)
        self.global_voice_selector.setVisible(sv.pin_tts_voice_cb.isChecked())

        sv.mic_vu_hint_requested.connect(self.pulse_mic_vu_meter_attention)

        sv.timeout_spinner.valueChanged.connect(self.toolbar_timeout_spin.setValue)
        sv.threshold_spinner.valueChanged.connect(self.toolbar_threshold_spin.setValue)
        if hasattr(sv, "wakeword_sensitivity"):
            sv.wakeword_sensitivity.valueChanged.connect(
                self.toolbar_wakeword_sensitivity_spin.setValue
            )

        self.toolbar_timeout_spin.valueChanged.connect(sv.timeout_spinner.setValue)
        self.toolbar_threshold_spin.valueChanged.connect(sv.threshold_spinner.setValue)
        if hasattr(sv, "wakeword_sensitivity"):
            self.toolbar_wakeword_sensitivity_spin.valueChanged.connect(
                sv.wakeword_sensitivity.setValue
            )

        sv.rag_kb_toggle.connect(self.tool_rag_toggle.setChecked)
        if getattr(sv, "rag_kb_cb", None) is not None:
            self.tool_rag_toggle.toggled.connect(sv.rag_kb_cb.setChecked)
        sv.auto_activator_toggle.connect(self.rag_auto_toggle.setChecked)
        if getattr(sv, "auto_activator_cb", None) is not None:
            self.rag_auto_toggle.toggled.connect(sv.auto_activator_cb.setChecked)

        sv.auto_load_last_model_changed.connect(self._sync_toolbar_auto_load_model_toggle)

        sv.engine_mode_changed.connect(self._refresh_toolbar_native_model_from_settings_signal)
        sv.engine_mode_changed.connect(lambda _mode: self.refresh_active_tour_layout())

        self._wire_generation_settings_toolbar_sync()
        self._wire_toolbar_internet_settings_sync(sv)
        self._sync_settings_tts_voice_selector()
        self._sync_settings_tts_voice_enabled_toggle(
            self.voice_bypass_toggle.isChecked()
            if hasattr(self, "voice_bypass_toggle")
            else True
        )
        if hasattr(sv, "tts_voice_enabled_toggle"):
            sv.tts_voice_enabled_toggle.toggled.connect(
                self._on_settings_tts_voice_enabled_toggled
            )
        self._sync_settings_voice_input_enabled_toggle(
            self.voice_input_toggle.isChecked()
            if hasattr(self, "voice_input_toggle")
            else True
        )
        if hasattr(sv, "voice_input_enabled_toggle"):
            sv.voice_input_enabled_toggle.toggled.connect(
                self._on_settings_voice_input_enabled_toggled
            )

    def _wire_model_manager_view(self) -> None:
        if self._model_manager_view_wired:
            return
        mm = self._model_manager_view
        if mm is None:
            return
        self._model_manager_view_wired = True
        if hasattr(mm, "native_library_changed"):
            mm.native_library_changed.connect(self.refresh_toolbar_native_model_dropdown)

    def _ensure_main_stage_view(self, index: int) -> QWidget:
        if self._is_main_stage_built(index):
            widget = self.main_stage.widget(index)
            if widget is not None:
                return widget

        view = self._create_main_stage_view(index)
        placeholder = self.main_stage.widget(index)
        if placeholder is not None:
            self.main_stage.removeWidget(placeholder)
            placeholder.deleteLater()
        self.main_stage.insertWidget(index, view)
        self._main_stage_built.add(index)
        self._refresh_stage_theme(index, self._is_dark_theme)
        self._flush_main_stage_app_wirers(index)
        return view

    @property
    def library_view(self) -> LibraryView:
        return self.ensure_library_view()

    @library_view.setter
    def library_view(self, value: LibraryView | None) -> None:
        self._library_view = value

    @property
    def memory_manager_view(self) -> MemoryManagerView:
        return self.ensure_memory_manager_view()

    @memory_manager_view.setter
    def memory_manager_view(self, value: MemoryManagerView | None) -> None:
        self._memory_manager_view = value

    @property
    def telemetry_view(self) -> TelemetryView:
        return self.ensure_telemetry_view()

    @telemetry_view.setter
    def telemetry_view(self, value: TelemetryView | None) -> None:
        self._telemetry_view = value

    @property
    def model_manager_view(self) -> ModelManagerView:
        return self.ensure_model_manager_view()

    @model_manager_view.setter
    def model_manager_view(self, value: ModelManagerView | None) -> None:
        self._model_manager_view = value

    @property
    def settings_view(self) -> SettingsView:
        return self.ensure_settings_view()

    @settings_view.setter
    def settings_view(self, value: SettingsView | None) -> None:
        self._settings_view = value

    def _route_view(self, index: int, active_button: QPushButton):
        """Switches the QStackedWidget and manages button highlights.

        Updates icons only for the previous and newly active buttons to avoid
        rebuilding all nav pixmaps each click (noticeable flicker on Windows).
        """
        if index in _LAZY_MAIN_STAGE_INDICES:
            self._ensure_main_stage_view(index)

        prev_active = getattr(self, "_nav_active_btn", None)
        stage = self.main_stage
        stage.setUpdatesEnabled(False)
        try:
            stage.setCurrentIndex(index)
            for btn in self.nav_buttons:
                btn.setChecked(btn is active_button)
            active_button.setChecked(True)
            updated: set[QPushButton] = set()
            if isinstance(prev_active, QPushButton) and prev_active in self.nav_buttons:
                updated.add(prev_active)
            updated.add(active_button)
            for btn in updated:
                self._refresh_nav_btn_icon(btn)
            self._nav_active_btn = active_button
        finally:
            stage.setUpdatesEnabled(True)
            stage.update()
        if index in getattr(self, "_theme_stage_dirty", ()):
            self._sync_stage_theme_if_dirty(index)
        if index in (
            MAIN_STAGE_MEMORY,
            MAIN_STAGE_TELEMETRY,
            MAIN_STAGE_MODEL_MANAGER,
            MAIN_STAGE_SETTINGS,
        ):
            self._set_tools_pane_expanded(False, animate=False)
        if index == MAIN_STAGE_CONVERSATIONS and hasattr(self, "conversations_view"):
            QTimer.singleShot(0, self.conversations_view.focus_composer_if_ready)

    @staticmethod
    def _load_theme_qss(style_path) -> str:
        """Read and cache theme QSS so profiling can separate disk read from apply."""
        cache = getattr(MainWindow, "_theme_qss_cache", None)
        if cache is None:
            cache = {}
            MainWindow._theme_qss_cache = cache
        key = str(style_path)
        cached = cache.get(key)
        if cached is not None:
            return cached
        with open(style_path, "r") as f:
            cached = f.read()
        cache[key] = cached
        return cached

    def _theme_toggle_profile_context(self) -> dict[str, int | str]:
        """Snapshot widget counts to correlate timings with session heaviness."""
        from PyQt6.QtWidgets import QApplication

        from core.theme_toggle_profile import (
            collect_application_widget_metrics,
            is_theme_stylesheet_clear_skipped,
        )

        ctx: dict[str, int | str] = {
            "target_theme": "dark" if self._is_dark_theme else "light",
            "skip_stylesheet_clear": int(is_theme_stylesheet_clear_skipped()),
        }
        ctx.update(collect_application_widget_metrics(QApplication.instance()))
        if hasattr(self, "main_stage"):
            ctx["main_stage_index"] = int(self.main_stage.currentIndex())
        ctx["built_main_stages"] = len(getattr(self, "_main_stage_built", ()))

        cv = getattr(self, "conversations_view", None)
        if cv is not None:
            history = getattr(cv, "history_list", None)
            if history is not None:
                ctx["conversation_rows"] = int(history.count())
            transcript = getattr(cv, "transcript_layout", None)
            if transcript is not None:
                ctx["transcript_widgets"] = int(transcript.count())

        lv = self._library_view
        if lv is not None:
            doc_list = getattr(lv, "doc_list", None)
            if doc_list is not None:
                ctx["library_rows"] = int(doc_list.count())

        sv = self._settings_view
        if sv is not None:
            trigger_list = getattr(sv, "trigger_list", None)
            if trigger_list is not None:
                ctx["trigger_rows"] = int(trigger_list.count())

        mm = self._model_manager_view
        if mm is not None:
            hub_list = getattr(mm, "hub_model_list", None)
            if hub_list is not None:
                ctx["hub_rows"] = int(hub_list.count())
            readme = getattr(mm, "_last_readme_markdown", None)
            if readme:
                ctx["readme_chars"] = len(str(readme))

        return ctx

    def _active_main_stage_index(self) -> int:
        if hasattr(self, "main_stage"):
            return int(self.main_stage.currentIndex())
        return 0

    def _refresh_conversations_theme(
        self, is_dark: bool, *, include_transcript: bool = True
    ) -> None:
        cv = getattr(self, "conversations_view", None)
        if cv is None:
            return
        if hasattr(cv, "refresh_menu_themes"):
            cv.refresh_menu_themes(is_dark)
        if hasattr(cv, "refresh_button_themes"):
            cv.refresh_button_themes(is_dark)
        if include_transcript and hasattr(cv, "_refresh_transcript_theme"):
            cv._refresh_transcript_theme()
        if hasattr(cv, "_update_row_colors"):
            cv._update_row_colors()

    def _refresh_library_theme(self, is_dark: bool) -> None:
        lv = self._library_view
        if lv is None:
            return
        if hasattr(lv, "refresh_menu_themes"):
            lv.refresh_menu_themes(is_dark)
        if hasattr(lv, "refresh_button_themes"):
            lv.refresh_button_themes(is_dark)
        if hasattr(lv, "_update_row_colors"):
            lv._update_row_colors()

    def _refresh_stage_theme(self, stage_index: int, is_dark: bool) -> None:
        if stage_index == 0:
            self._refresh_conversations_theme(is_dark, include_transcript=True)
        elif stage_index == 1:
            self._refresh_library_theme(is_dark)
        elif stage_index == 2:
            mmv = self._memory_manager_view
            if mmv is not None and hasattr(mmv, "refresh_theme"):
                mmv.refresh_theme(is_dark)
        elif stage_index == 3:
            tv = self._telemetry_view
            if tv is not None and hasattr(tv, "refresh_after_theme_toggle"):
                tv.refresh_after_theme_toggle()
        elif stage_index == 4:
            mm = self._model_manager_view
            if mm is not None and hasattr(mm, "refresh_after_theme_toggle"):
                mm.refresh_after_theme_toggle()
        elif stage_index == 5:
            sv = self._settings_view
            if sv is not None and hasattr(sv, "refresh_menu_themes"):
                sv.refresh_menu_themes(is_dark)

    def _refresh_global_theme_chrome(self, is_dark: bool, profiler) -> None:
        with profiler.step("topbar_menus"):
            if hasattr(self, "_topbar_mic_menu"):
                self._apply_menu_theme(self._topbar_mic_menu, is_dark)
            self._apply_topbar_mic_chevron_style()

            if hasattr(self, "global_voice_selector"):
                toolbar_menu = self.global_voice_selector.menu()
                if toolbar_menu:
                    self._apply_menu_theme(toolbar_menu, is_dark)

            if hasattr(self, "toolbar_native_model_selector"):
                native_menu = self.toolbar_native_model_selector.menu()
                if native_menu:
                    self._apply_menu_theme(native_menu, is_dark)
                self._apply_settings_menu_button_chevron_state(
                    self.toolbar_native_model_selector
                )
                self._apply_native_model_eject_button_style()

            selector = getattr(self, "toolbar_privacy_tier_selector", None)
            if selector is not None:
                if isinstance(selector, SelectorButton):
                    selector.apply_theme(is_dark)
                menu = selector.menu()
                if menu:
                    self._apply_menu_theme(menu, is_dark)

        if hasattr(self, "background_progress_row"):
            with profiler.step("background_progress_row.apply_theme"):
                self.background_progress_row.apply_theme(is_dark)

        if hasattr(self, "notification_center"):
            with profiler.step("notification_center.apply_theme"):
                self.notification_center.apply_theme(is_dark)

        if hasattr(self, "_modal_backdrop"):
            with profiler.step("modal_backdrop.apply_theme"):
                self._modal_backdrop.apply_theme(is_dark)

        if self.tray_controller is not None:
            with profiler.step("tray_controller.apply_theme"):
                self.tray_controller.apply_theme(is_dark)

        if self._companion_controller is not None:
            with profiler.step("companion_controller.apply_theme"):
                self._companion_controller.apply_theme(is_dark)

        if self.canonical_trace_diff_view is not None:
            with profiler.step("canonical_trace_diff.apply_theme"):
                self.canonical_trace_diff_view.apply_theme(is_dark)

        if self._scenario_workflow_dialog is not None:
            with profiler.step("scenario_workflow.refresh_theme"):
                self._scenario_workflow_dialog.refresh_theme(is_dark)

        with profiler.step("prestige_toggles.apply_theme"):
            for toggle in self.findChildren(PrestigeToggle):
                toggle.apply_theme(is_dark=is_dark)

        with profiler.step("nav_buttons.refresh_icons"):
            for btn in getattr(self, "nav_buttons", ()):
                self._refresh_nav_btn_icon(btn)

        with profiler.step("shell_chrome.refresh"):
            theme = self._theme_manager.current
            if hasattr(self, "vu_meter"):
                self.vu_meter.apply_theme(theme)
            self._sync_retrieval_indicator_palette()
            cpu, ram, gpu = telemetry_metric_colors(theme)
            for lbl, color in (
                (getattr(self, "side_cpu_lbl", None), cpu),
                (getattr(self, "side_ram_lbl", None), ram),
                (getattr(self, "side_gpu_lbl", None), gpu),
            ):
                if lbl is not None:
                    lbl.setStyleSheet(
                        f"color: {color}; font-weight: bold; font-size: 10px; letter-spacing: 0.5px;"
                    )
            if hasattr(self, "toggle_tools_btn"):
                expanded = not self._is_tools_pane_collapsed()
                chevron = "fa5s.chevron-right" if expanded else "fa5s.chevron-left"
                self.toggle_tools_btn.setIcon(qta.icon(chevron, color=theme.link))
            self._refresh_toolbar_control_icons(theme)
            self._on_topbar_mic_attention_tick()

    def _sync_stage_theme_if_dirty(self, stage_index: int) -> None:
        dirty = getattr(self, "_theme_stage_dirty", None)
        if not dirty or stage_index not in dirty:
            return
        if stage_index == MAIN_STAGE_CONVERSATIONS:
            self._refresh_conversations_theme(
                self._is_dark_theme, include_transcript=True
            )
        else:
            self._refresh_stage_theme(stage_index, self._is_dark_theme)
        dirty.discard(stage_index)

    def _schedule_deferred_theme_refreshes(self, hidden_stage_indices: list[int]) -> None:
        if not hidden_stage_indices:
            return
        self._theme_stage_dirty.update(hidden_stage_indices)
        QTimer.singleShot(0, self._flush_deferred_theme_refreshes)

    def _flush_deferred_theme_refreshes(self) -> None:
        dirty = getattr(self, "_theme_stage_dirty", None)
        if not dirty:
            return

        from core.theme_toggle_profile import ThemeToggleProfiler

        profiler = ThemeToggleProfiler.maybe_enabled()
        profiler.begin()
        is_dark = self._is_dark_theme
        active_stage = self._active_main_stage_index()
        deferred_indices = sorted(dirty)
        for stage_index in deferred_indices:
            with profiler.step(f"stage_{stage_index}.refresh_deferred"):
                if stage_index == MAIN_STAGE_CONVERSATIONS:
                    include_transcript = stage_index == active_stage
                    if not include_transcript:
                        # Hidden Conversations stays dirty until show; skip chrome
                        # refresh here (saves ~170ms) — full refresh on navigate back.
                        continue
                    self._refresh_conversations_theme(
                        is_dark,
                        include_transcript=True,
                    )
                    dirty.discard(stage_index)
                else:
                    self._refresh_stage_theme(stage_index, is_dark)
                    dirty.discard(stage_index)
        profiler.finish(
            context={
                "deferred_batch": 1,
                "target_theme": "dark" if is_dark else "light",
            }
        )

    def _prepare_theme_toggle_apply(self, profiler) -> None:
        from PyQt6.QtWidgets import QApplication

        from core.theme_toggle_profile import is_theme_stylesheet_clear_forced

        app = QApplication.instance()
        with profiler.step("freeze_updates"):
            self.setUpdatesEnabled(False)
        with profiler.step("palette_reset"):
            app.setPalette(app.style().standardPalette())

        if is_theme_stylesheet_clear_forced():
            with profiler.step("stylesheet_clear"):
                app.setStyleSheet("")
        else:
            with profiler.step("stylesheet_clear_skipped"):
                pass

    def _on_theme_polarity_no_sibling(self, request):
        from core.theme.polarity_toggle import PolarityToggleAction
        from ui.components.theme_polarity_fallback_dialog import prompt_theme_polarity_fallback
        from ui.onboarding.tour_helpers import open_settings_section

        action = prompt_theme_polarity_fallback(
            self,
            request,
            is_dark=self._is_dark_theme,
        )
        if action is PolarityToggleAction.CHOOSE_THEME:
            open_settings_section(self, "appearance.themes")
        return action

    def _toggle_theme(self):
        """Toggle light/dark via ``ThemeManager.toggle_polarity()`` (family-aware)."""
        from core.theme_toggle_profile import ThemeToggleProfiler

        profiler = ThemeToggleProfiler.maybe_enabled()
        profiler.begin()
        self._active_theme_profiler = profiler

        active_stage = self._active_main_stage_index()
        hidden_stages = [
            idx
            for idx in range(self.main_stage.count())
            if idx != active_stage and self._is_main_stage_built(idx)
        ]
        self._pending_hidden_stages = hidden_stages

        resolved = self._theme_manager.toggle_polarity(
            on_no_sibling=self._on_theme_polarity_no_sibling,
            prepare_apply=lambda: self._prepare_theme_toggle_apply(profiler),
            persist=True,
            profiler=profiler,
        )

        self._pending_hidden_stages = None
        self._active_theme_profiler = None

        if resolved is None:
            self.setUpdatesEnabled(True)
            profiler.finish(
                context={
                    **self._theme_toggle_profile_context(),
                    "deferred_stage_count": len(hidden_stages),
                    "cancelled": True,
                }
            )
            return

        profiler.finish(
            context={
                **self._theme_toggle_profile_context(),
                "deferred_stage_count": len(hidden_stages),
            }
        )
        self._schedule_deferred_theme_refreshes(hidden_stages)

    def _setup_notification_service(self) -> None:
        self._notification_service.set_window_state_providers(
            visible=lambda: self.isVisible() and not self.isMinimized(),
            focused=lambda: self.isActiveWindow(),
            tts_playing=self._is_tts_playing,
            companion_visible=self._is_companion_visible,
            companion_attention=self._is_companion_attention,
        )
        self._notification_service.set_show_handlers(
            in_app=self._show_in_app_notification,
            os_notify=self._os_notification_adapter.show,
        )
        self._notification_service.action_triggered.connect(self._on_notification_service_action)
        self._notification_service.notification_shown.connect(self._on_notification_shown)
        self._setup_provider_limit_notifications()

    def _setup_provider_limit_notifications(self) -> None:
        from core.knowledge.provider_limit_events import register_provider_limit_handler
        from core.notification_types import provider_limit_notification_event

        register_provider_limit_handler(
            lambda event: self.emit_notification(provider_limit_notification_event(event))
        )

    def _is_tts_playing(self) -> bool:
        cv = getattr(self, "conversations_view", None)
        return bool(getattr(cv, "_tts_playing", False)) if cv is not None else False

    def _is_companion_visible(self) -> bool:
        if self._companion_controller is None:
            return False
        return self._companion_controller.is_visible_for_policy

    def _is_companion_attention(self) -> bool:
        return companion_attention_mode(self._presence_service.snapshot())

    def _show_in_app_notification(self, event: NotificationEvent) -> None:
        if hasattr(self, "notification_center"):
            self.notification_center.show_notification(event.to_app_request())

    def _on_notification_shown(self, event: NotificationEvent) -> None:
        if self.tray_controller is None:
            return
        items = [(e.title, e.body) for e in self._notification_service.history.recent(5)]
        self.tray_controller.update_recent_notifications(items)
        if self._companion_controller is not None:
            self._companion_controller.pulse_notification()

    def _on_notification_service_action(self, action_id: str, _event_id: str) -> None:
        self._on_notification_action(action_id)

    def emit_notification(self, event: NotificationEvent) -> None:
        """Public entry for workers/adapters to raise a notification."""
        self._notification_service.emit(event)

    def schedule_auto_state_backup(self) -> None:
        """Defer an automatic backup check until after startup settles."""
        if getattr(self, "_auto_state_backup_scheduled", False):
            return
        self._auto_state_backup_scheduled = True
        from core.state_backup.scheduler import STARTUP_AUTO_BACKUP_DELAY_MS

        QTimer.singleShot(STARTUP_AUTO_BACKUP_DELAY_MS, self._run_auto_state_backup_if_due)

    def _run_auto_state_backup_if_due(self) -> None:
        from core import app_settings

        if not app_settings.get_backup_auto_enabled():
            return
        if getattr(self, "_state_backup_auto_worker", None) is not None:
            return
        from workers.state_backup_auto_worker import StateBackupAutoWorker

        worker = StateBackupAutoWorker()
        worker.finished_with_result.connect(self._on_auto_state_backup_finished)
        worker.start()
        self._state_backup_auto_worker = worker

    def _on_auto_state_backup_finished(self, result: object) -> None:
        from core.state_backup.scheduler import AutoBackupResult

        worker = getattr(self, "_state_backup_auto_worker", None)
        if worker is not None:
            worker.deleteLater()
            self._state_backup_auto_worker = None
        if not isinstance(result, AutoBackupResult) or not result.ran:
            return
        from core.notification_types import (
            auto_backup_complete_event,
            auto_backup_failed_event,
        )

        if result.ok and result.destination is not None:
            self.emit_notification(auto_backup_complete_event(destination=result.destination))
        elif not result.ok:
            self.emit_notification(auto_backup_failed_event(error=result.error or ""))
        settings_view = getattr(self, "_settings_view", None)
        if settings_view is not None and hasattr(
            settings_view, "notify_auto_state_backup_finished"
        ):
            settings_view.notify_auto_state_backup_finished(result)

    @property
    def notification_service(self) -> NotificationService:
        return self._notification_service

    def _restore_workspace_from_tray(self) -> None:
        """Show the main window after hide-to-tray or minimize; raise and focus."""
        if self._force_app_exit:
            return
        if (
            self._companion_controller is not None
            and self._companion_controller.is_shutting_down
        ):
            return
        self.show()
        if self.isMinimized():
            self.showNormal()
        self.raise_()
        self.activateWindow()
        if self._companion_controller is not None:
            self._companion_controller.on_main_shown()

    def _on_companion_open_chat(self) -> None:
        self._restore_workspace_from_tray()
        if hasattr(self, "nav_chat"):
            self.nav_chat.setChecked(True)
            self._route_view(0, self.nav_chat)

    def _on_companion_new_chat(self) -> None:
        self._restore_workspace_from_tray()
        if hasattr(self, "nav_chat"):
            self.nav_chat.setChecked(True)
            self._route_view(0, self.nav_chat)
        if hasattr(self, "conversations_view"):
            self.conversations_view._start_new_chat()

    def open_chat_with_library_document(self, filename: str) -> None:
        """Navigate to Conversations, start a new thread, and prefill a file attachment token."""
        filename = (filename or "").strip()
        if not filename:
            return
        if hasattr(self, "nav_chat"):
            self.nav_chat.setChecked(True)
            self._route_view(0, self.nav_chat)
        cv = getattr(self, "conversations_view", None)
        if cv is None or not hasattr(cv, "start_new_chat_with_composer_prefill"):
            return
        from core.composer_attachments import ComposerAttachment, format_token, validate_file_token

        if not validate_file_token(filename):
            return
        token = format_token(ComposerAttachment(kind="file", id=filename, label=filename))
        cv.start_new_chat_with_composer_prefill(f"{token} ")

    def _on_companion_load_model(self, path: str) -> None:
        self._restore_workspace_from_tray()
        self.load_native_model_from_path(path)

    def load_native_model_from_path(self, path: str) -> None:
        """Activate a downloaded .gguf for the native engine (toolbar + companion menus)."""
        path = resolve_internal_model_path(path)
        if not path or not Path(path).is_file():
            return
        set_internal_model_path(path)
        if self._llm_worker:
            cv = getattr(self, "conversations_view", None)
            if cv is not None and hasattr(cv, "interrupt_active_response"):
                cv.interrupt_active_response()
            self._pending_native_model_path = path
            self._native_model_loading = True
            self._native_model_loaded_success = False
            self._set_native_model_progress_loading(True)
            self._llm_worker.refresh_native_model_from_settings()
        if hasattr(self, "refresh_toolbar_native_model_dropdown"):
            self.refresh_toolbar_native_model_dropdown()

    def _setup_tray(self) -> None:
        self.tray_controller = TrayController(
            self,
            voice_input_enabled=lambda: bool(
                getattr(self, "voice_input_toggle", None)
                and self.voice_input_toggle.isChecked()
            ),
            voice_output_enabled=lambda: bool(
                getattr(self, "voice_bypass_toggle", None)
                and self.voice_bypass_toggle.isChecked()
            ),
            tray_logo_path=self._resolve_logo_asset("qube_logo_256.png"),
        )
        self.tray_icon = self.tray_controller.tray_icon
        if not self.tray_controller.available:
            logger.warning("System tray unavailable — hide-to-tray disabled.")
            return

        self.tray_controller.open_requested.connect(self._restore_workspace_from_tray)
        self.tray_controller.exit_requested.connect(self._request_app_exit)
        self.tray_controller.restart_requested.connect(self.request_application_restart)
        self.tray_controller.voice_input_toggled.connect(self._on_tray_voice_input_toggled)
        self.tray_controller.voice_output_toggled.connect(self._on_tray_voice_output_toggled)
        self.tray_controller.navigate_requested.connect(self._on_tray_navigate)
        self.tray_controller.dnd_toggled.connect(self._on_tray_dnd_toggled)
        self.tray_controller.companion_toggled.connect(self._on_tray_companion_toggled)

        self._os_notification_adapter.set_tray_icon(self.tray_controller.tray_icon)
        self._setup_notification_service()

        if hasattr(self, "voice_input_toggle"):
            self.voice_input_toggle.toggled.connect(self._sync_tray_voice_toggles)
        if hasattr(self, "voice_bypass_toggle"):
            self.voice_bypass_toggle.toggled.connect(self._sync_tray_voice_toggles)

        self._sync_tray_presence()

    def _setup_companion(self) -> None:
        from core import app_settings as _app_settings

        self._companion_controller = CompanionController(self._presence_service, self)
        self._companion_controller.bind_main_window(self)
        self._companion_controller.open_requested.connect(self._restore_workspace_from_tray)
        self._companion_controller.open_chat_requested.connect(self._on_companion_open_chat)
        self._companion_controller.new_chat_requested.connect(self._on_companion_new_chat)
        self._companion_controller.load_model_requested.connect(self._on_companion_load_model)
        self._companion_controller.open_model_manager_requested.connect(
            lambda: self._on_notification_action("open_models")
        )
        self._companion_controller.voice_input_toggled.connect(self._on_tray_voice_input_toggled)
        self._companion_controller.voice_output_toggled.connect(self._on_tray_voice_output_toggled)
        self._companion_controller.hide_companion_requested.connect(
            lambda: self._on_tray_companion_toggled(False)
        )
        self._companion_controller.navigate_settings_requested.connect(
            lambda: self._on_notification_action("open_settings")
        )
        self._companion_controller.window.snap_zone_changed.connect(
            self._on_companion_snap_zone_changed
        )
        voice_out = (
            self.voice_bypass_toggle.isChecked()
            if hasattr(self, "voice_bypass_toggle")
            else True
        )
        self._presence_service.set_voice_output_muted(not voice_out)
        self._presence_service.set_dnd(_app_settings.get_notifications_dnd())

    def _on_tray_dnd_toggled(self, enabled: bool) -> None:
        self._presence_service.set_dnd(enabled)
        if self._companion_controller is not None:
            self._companion_controller.on_settings_changed()

    def _on_tray_companion_toggled(self, enabled: bool) -> None:
        sv = self._settings_view
        if sv is not None and hasattr(sv, "companion_enabled_cb"):
            sv.companion_enabled_cb.blockSignals(True)
            sv.companion_enabled_cb.setChecked(enabled)
            sv.companion_enabled_cb.blockSignals(False)
        if self._companion_controller is not None:
            self._companion_controller.set_user_enabled(enabled)

    def _on_companion_snap_zone_changed(self, _zone: str) -> None:
        settings = self._settings_view
        if settings is not None and hasattr(settings, "_sync_companion_snap_compass"):
            settings._sync_companion_snap_compass()

    def _on_tray_voice_input_toggled(self, enabled: bool) -> None:
        if hasattr(self, "voice_input_toggle"):
            self.voice_input_toggle.blockSignals(True)
            self.voice_input_toggle.setChecked(enabled)
            self.voice_input_toggle.blockSignals(False)
        if self._audio_worker is not None:
            self._audio_worker.set_paused(not enabled)
        if hasattr(self, "vu_meter") and not enabled:
            self.vu_meter.set_level(0.0)
        self._sync_settings_voice_input_enabled_toggle(enabled)
        self._sync_tray_presence()
        self._sync_tray_voice_toggles()

    def _on_tray_voice_output_toggled(self, enabled: bool) -> None:
        if hasattr(self, "voice_bypass_toggle"):
            self.voice_bypass_toggle.blockSignals(True)
            self.voice_bypass_toggle.setChecked(enabled)
            self.voice_bypass_toggle.blockSignals(False)
        if self._tts_worker is not None:
            self._tts_worker.set_mute(not enabled)
        self._presence_service.set_voice_output_muted(not enabled)
        self._sync_settings_tts_voice_enabled_toggle(enabled)
        self._sync_tray_voice_toggles()

    def _sync_tray_voice_toggles(self, *_args) -> None:
        if self.tray_controller is None:
            return
        voice_in = self.voice_input_toggle.isChecked() if hasattr(self, "voice_input_toggle") else True
        voice_out = self.voice_bypass_toggle.isChecked() if hasattr(self, "voice_bypass_toggle") else True
        self.tray_controller.sync_voice_toggles(voice_in=voice_in, voice_out=voice_out)
        self._presence_service.set_voice_output_muted(not voice_out)
        self._sync_tray_presence()

    def _on_tray_navigate(self, action_id: str) -> None:
        self._on_notification_action(action_id)

    def _sync_tray_presence(self) -> None:
        if self.tray_controller is None:
            return
        self.tray_controller.set_activity(
            self._activity_reducer.activity,
            voice_output_muted=self._presence_service.snapshot().voice_output_muted,
        )

    def _should_hide_to_tray(self) -> bool:
        """True when close should minimize to tray instead of quitting."""
        if self._force_app_exit:
            return False
        tc = self.tray_controller
        return tc is not None and tc.available

    def _request_app_exit(self) -> None:
        """Force a real app exit instead of hide-to-tray."""
        self._force_app_exit = True
        if self._companion_controller is not None:
            self._companion_controller.shutdown()
        if hasattr(self, "_notification_service"):
            self._notification_service.shutdown()
        if self.tray_controller is not None:
            self.tray_controller.hide_tray()
        elif self.tray_icon is not None:
            self.tray_icon.hide()
        app = QApplication.instance()
        if app is not None:
            app.quit()
        else:
            self.close()

    def _start_timers(self) -> None:
        # Repurposed telemetry timer for the new Mini-Telemetry block
        self.telemetry_timer = QTimer()
        self.telemetry_timer.timeout.connect(self._update_mini_telemetry)
        self.telemetry_timer.start(1000) # Once per second is fine for mini text

    def _update_mini_telemetry(self):
        """Refreshes the sidebar metrics and syncs the main dashboard."""
        # 1. Gather fresh stats
        ram = int(psutil.virtual_memory().percent)
        cpu = int(psutil.cpu_percent())
        # Note: Using self._gpu_monitor to match your existing logic
        gpu = int(self._gpu_monitor.get_load()) if self._gpu_monitor else 0

        # 2. Update the three individual sidebar labels
        # We use hasattr as a safety check in case this fires during a theme change/rebuild
        if hasattr(self, 'side_cpu_lbl'):
            self.side_cpu_lbl.setText(f"CPU {cpu}%")
            self.side_ram_lbl.setText(f"RAM {ram}%")
            self.side_gpu_lbl.setText(f"GPU {gpu}%")
            
        # 3. Keep the Advanced Telemetry screen in sync
        # This prevents the sidebar and the main graph from ever showing different numbers
        if self._telemetry_view is not None:
            tv = self._telemetry_view
            tv.live_cpu_lbl.setText(f"CPU: {cpu}%")
            tv.live_ram_lbl.setText(f"RAM: {ram}%")
            tv.live_gpu_lbl.setText(f"GPU: {gpu}%")

    # ------------------------------------------------------------------ #
    #  FRAMELESS DRAG & DROP EVENT ROUTING                               #
    # ------------------------------------------------------------------ #

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.top_bar.underMouse():
            if self._workspace_maximized:
                return
            self._old_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if self._old_pos is not None:
            delta = event.globalPosition().toPoint() - self._old_pos
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self._old_pos = event.globalPosition().toPoint()

    def mouseReleaseEvent(self, event):
        self._old_pos = None

    def mouseDoubleClickEvent(self, event):
        """Trigger maximize toggle when the top bar is double-clicked."""
        if event.button() == Qt.MouseButton.LeftButton and self.top_bar.underMouse():
            self._toggle_maximize()

    def closeEvent(self, event):
        if self._should_hide_to_tray():
            self.hide()
            if self.tray_controller is not None:
                self.tray_controller.show_tray()
            if self._companion_controller is not None:
                self._companion_controller.on_main_hidden()
            event.ignore()
        else:
            if self.routing_debug_tool_view is not None:
                self.routing_debug_tool_view.close()
            if self.canonical_trace_diff_view is not None:
                self.canonical_trace_diff_view.close()
            if hasattr(self, "_notification_service"):
                self._notification_service.shutdown()
            if self._companion_controller is not None:
                self._companion_controller.shutdown()
            if self.tray_controller is not None:
                self.tray_controller.hide_tray()
            event.accept()

    # ------------------------------------------------------------------ #
    #  PUBLIC STUBS (Keeps main.py running during transition)            #
    # ------------------------------------------------------------------ #
    # These methods receive signals from workers. Once we build the 
    # ConversationsView, we will forward these calls directly to it.

    def show_app_notification(self, request: AppNotificationRequest) -> None:
        """Show a bottom-right toast (updates, release notes, post-command actions)."""
        from core.notification_types import NotificationEvent, NotificationSeverity

        event = NotificationEvent(
            title=request.title,
            body=request.body,
            severity=NotificationSeverity(getattr(request, "severity", "info")),
            category=getattr(request, "category", "update"),  # type: ignore[arg-type]
            action_label=request.action_label,
            action_id=request.action_id,
            auto_dismiss_ms=request.auto_dismiss_ms,
            event_id=getattr(request, "event_id", "") or "",
        )
        self._notification_service.emit(event)

    def _on_notification_action(self, action_id: str) -> None:
        if action_id == "restart_app":
            self.request_application_restart()
        elif action_id == "open_main_window":
            self._restore_workspace_from_tray()
        elif action_id == "open_settings":
            self._restore_workspace_from_tray()
            if hasattr(self, "nav_settings"):
                self.nav_settings.setChecked(True)
                self._route_view(5, self.nav_settings)
        elif action_id == "open_help_wakeword_models":
            self._restore_workspace_from_tray()
            if hasattr(self, "nav_settings"):
                self.nav_settings.setChecked(True)
                self._route_view(5, self.nav_settings)

            # Scroll to the wakeword models help anchor after the Settings
            # view is routed/constructed.
            def _jump() -> None:
                sv = self._settings_view
                if sv is not None and hasattr(sv, "select_settings_section"):
                    sv.select_settings_section(
                        "help", anchor="wakeword-models"
                    )

            QTimer.singleShot(0, _jump)
        elif action_id == "open_models":
            self._open_model_manager_page()
        elif action_id == "open_local_model_picker":
            self._open_local_model_picker_from_toolbar()
        elif action_id == "open_library":
            self._restore_workspace_from_tray()
            if hasattr(self, "nav_library"):
                self.nav_library.setChecked(True)
                self._route_view(1, self.nav_library)
        elif action_id == "open_memories":
            self._restore_workspace_from_tray()
            if hasattr(self, "nav_memory"):
                self.nav_memory.setChecked(True)
                self._route_view(2, self.nav_memory)
        elif action_id == "open_settings_voice_stt":
            self._open_settings_section("voice.audio", anchor="stt_models")
        elif action_id == "open_settings_voice_tts":
            self._open_settings_section("voice.audio", anchor="tts_models")
        elif action_id == "open_settings_knowledge_embedding":
            self._open_settings_section("knowledge", anchor="embedding_mode")
        elif action_id == "open_settings_knowledge_web_discovery":
            self._open_settings_section("knowledge", anchor="web_discovery")
        elif action_id == "open_settings_knowledge_credentials" or action_id.startswith(
            "open_settings_knowledge_credentials:"
        ):
            provider_id = None
            if ":" in action_id:
                provider_id = action_id.split(":", 1)[1].strip() or None
            self._open_settings_section(
                "knowledge",
                anchor="knowledge_live_sources",
                configure_provider_id=provider_id,
            )
        elif action_id == "open_settings_ai_cognition":
            self._open_settings_section("ai.models", anchor="cognition")

    def _open_settings_section(
        self,
        section: str,
        *,
        anchor: str | None = None,
        configure_provider_id: str | None = None,
    ) -> None:
        self._restore_workspace_from_tray()
        if hasattr(self, "nav_settings"):
            self.nav_settings.setChecked(True)
            self._route_view(5, self.nav_settings)
        sv = self._settings_view
        if sv is not None and hasattr(sv, "select_settings_section"):
            QTimer.singleShot(
                0,
                lambda: sv.select_settings_section(
                    section,
                    anchor=anchor,
                    configure_provider_id=configure_provider_id,
                ),
            )

    def _guard_embedding_feature_toggle(self, toggle, checked: bool) -> bool:
        if not checked:
            return True
        from ui.bootstrap_feature_prompts import ensure_search_models_for_feature

        label = "Knowledge and library features"
        if not ensure_search_models_for_feature(
            self,
            feature_label=label,
            is_dark=self._is_dark_theme,
        ):
            toggle.blockSignals(True)
            toggle.setChecked(False)
            toggle.blockSignals(False)
            return False
        return True

    def _on_voice_input_toggle(self, checked: bool) -> None:
        if checked:
            from core.bootstrap_manifest import BootstrapModelId
            from ui.bootstrap_feature_prompts import ensure_bootstrap_model_downloaded

            if not ensure_bootstrap_model_downloaded(
                self,
                BootstrapModelId.WHISPER_SMALL,
                feature_label="Voice input",
                is_dark=self._is_dark_theme,
            ):
                self.voice_input_toggle.blockSignals(True)
                self.voice_input_toggle.setChecked(False)
                self.voice_input_toggle.blockSignals(False)
                self._sync_settings_voice_input_enabled_toggle(False)
                return
        if self._audio_worker:
            self._audio_worker.set_paused(not checked)
        if hasattr(self, "vu_meter") and not checked:
            self.vu_meter.set_level(0.0)
        self._sync_settings_voice_input_enabled_toggle(
            self.voice_input_toggle.isChecked()
            if hasattr(self, "voice_input_toggle")
            else checked
        )

    def _on_voice_bypass_toggle(self, checked: bool) -> None:
        if checked:
            from core.bootstrap_manifest import BootstrapModelId
            from core.tts_models import any_supported_tts_model_on_disk
            from ui.bootstrap_feature_prompts import ensure_bootstrap_model_downloaded

            if not any_supported_tts_model_on_disk() and not ensure_bootstrap_model_downloaded(
                self,
                BootstrapModelId.KOKORO_TTS,
                feature_label="Voice output (TTS)",
                is_dark=self._is_dark_theme,
            ):
                self.voice_bypass_toggle.blockSignals(True)
                self.voice_bypass_toggle.setChecked(False)
                self.voice_bypass_toggle.blockSignals(False)
                return
        if self._tts_worker:
            self._tts_worker.set_mute(not checked)
        self._sync_settings_tts_voice_enabled_toggle(
            self.voice_bypass_toggle.isChecked()
            if hasattr(self, "voice_bypass_toggle")
            else checked
        )

    def request_application_restart(self) -> None:
        self._force_app_exit = True
        if not relaunch_and_quit():
            PrestigeDialog(
                self,
                "Restart failed",
                manual_restart_instructions(),
                is_dark=self._is_dark_theme,
            ).exec()

    def begin_background_progress(self, detail: str = "") -> None:
        """Show the app-wide progress strip (visible on every page)."""
        self.background_progress_row.apply_theme(self._is_dark_theme)
        self.background_progress_row.begin(detail=detail)
        self.background_progress_banner.show()

    def update_background_progress(self, percent: int, *, detail: str | None = None) -> None:
        if not self.background_progress_banner.isVisible():
            self.begin_background_progress(detail=detail or "")
        self.background_progress_row.update_progress(percent, detail=detail)

    def set_background_progress_detail(self, detail: str) -> None:
        if not self.background_progress_banner.isVisible():
            self.begin_background_progress(detail=detail)
        else:
            self.background_progress_row.set_detail(detail)

    def finish_background_progress(self) -> None:
        self.background_progress_row.finish()
        self.background_progress_banner.hide()

    def update_status(self, message: str, force: bool = False) -> None:
        """Updates the top bar with a priority-based logic to prevent signal clobbering."""
        if self._force_app_exit:
            return
        transition = self._presence_service.reduce(message, force=force)
        if transition.blocked:
            return

        new_state = transition.bubble_state
        self.status_bubble.setText(f" {transition.display_text}")
        self.status_bubble.setProperty("state", new_state)
        self.status_bubble.style().unpolish(self.status_bubble)
        self.status_bubble.style().polish(self.status_bubble)

        self._sync_tray_presence()

        if hasattr(self, "conversations_view"):
            from core.assistant_activity import _is_assistant_working_message

            msg_upper = message.upper().strip()
            conv = self.conversations_view
            voice_turn_active = bool(getattr(conv, "_voice_turn_active", False))
            llm_in_progress = bool(getattr(conv, "_llm_in_progress", False))
            voice_capture_active = bool(getattr(conv, "_voice_capture_active", False))

            if new_state == "idle":
                # Mic gate closed — STT/LLM may still be running; do not tear down the turn UI.
                if msg_upper != "VOICE CAPTURE IDLE" or not (voice_turn_active or llm_in_progress):
                    conv.on_turn_complete_idle()
            elif new_state == "listening":
                conv.on_voice_capture_started()
            elif _is_assistant_working_message(msg_upper) and (
                voice_capture_active or voice_turn_active
            ):
                conv.on_voice_capture_processing()
            elif voice_capture_active:
                conv.on_voice_capture_ended()

            deep_research_active = bool(getattr(conv, "_deep_research_in_progress", False))
            if deep_research_active and not llm_in_progress and not voice_turn_active:
                conv.set_input_enabled(True)
            else:
                conv.set_input_enabled(new_state in ("idle", "speaking", "needs_model"))
            conv.apply_presence_label(transition.presence_label)

        msg_upper = message.upper().strip()
        if "MIC ERROR" in msg_upper and "NO INPUT DEVICE" not in msg_upper:
            from core.notification_types import voice_input_unavailable_event

            notify_key = "voice_input_unavailable"
            if self._last_mic_notification_detail != notify_key:
                self._last_mic_notification_detail = notify_key
                self.emit_notification(voice_input_unavailable_event())
        else:
            if self._last_mic_notification_detail is not None and new_state == "idle":
                self._last_mic_notification_detail = None
            if new_state == "needs_model":
                from core.notification_types import needs_model_event

                self.emit_notification(needs_model_event())

    def update_rag_indicator(self, active: bool) -> None:
        """Called by the LLM Worker when the Knowledge Base is used for a turn."""
        if active:
            self.set_rag_state('active')
        else:
            self.set_rag_state(
                'standby' if self.tool_rag_toggle.isChecked() else 'off'
            )

    def log_user_message(self, text: str) -> None:
        pass # Will be forwarded to ConversationsView

    def log_agent_token(self, token: str) -> None:
        pass # Will be forwarded to ConversationsView

    def update_stt_latency(self, ms: float) -> None:
        tv = self._telemetry_view
        if tv is not None:
            tv.update_stt_latency(ms)

    def update_ttft_latency(self, ms: float) -> None:
        tv = self._telemetry_view
        if tv is not None:
            tv.update_ttft_latency(ms)

    def on_audio_volume_update(self, level: float) -> None:
        self._presence_service.set_audio_level(level)

    def on_tts_playback_level(self, level: float) -> None:
        if self._companion_controller is not None:
            self._companion_controller.set_speech_level(level)

    def update_tts_latency(self, ms: float) -> None:
        tv = self._telemetry_view
        if tv is not None:
            tv.update_tts_latency(ms)

    def _sync_tts_voice_selector_labels(self, voice_name: str) -> None:
        if hasattr(self, "global_voice_selector"):
            self.global_voice_selector.setText(voice_name)
        settings = self._settings_view
        if settings is not None and hasattr(settings, "voice_selector"):
            settings.voice_selector.setText(voice_name)
            settings.voice_selector.update()

    def _on_tts_voice_selected(self, voice_name: str) -> None:
        if self._tts_worker:
            self._tts_worker.set_voice(voice_name)
        self._sync_tts_voice_selector_labels(voice_name)

    def _sync_settings_tts_voice_enabled_toggle(self, enabled: bool) -> None:
        settings = self._settings_view
        if settings is None or not hasattr(settings, "tts_voice_enabled_toggle"):
            return
        toggle = settings.tts_voice_enabled_toggle
        toggle.blockSignals(True)
        toggle.setChecked(enabled)
        toggle.blockSignals(False)

    def _sync_settings_voice_input_enabled_toggle(self, enabled: bool) -> None:
        settings = self._settings_view
        if settings is None or not hasattr(settings, "voice_input_enabled_toggle"):
            return
        toggle = settings.voice_input_enabled_toggle
        toggle.blockSignals(True)
        toggle.setChecked(enabled)
        toggle.blockSignals(False)

    def _on_settings_voice_input_enabled_toggled(self, checked: bool) -> None:
        if hasattr(self, "voice_input_toggle"):
            self.voice_input_toggle.blockSignals(True)
            self.voice_input_toggle.setChecked(checked)
            self.voice_input_toggle.blockSignals(False)
        self._on_voice_input_toggle(checked)

    def _on_settings_tts_voice_enabled_toggled(self, checked: bool) -> None:
        if hasattr(self, "voice_bypass_toggle"):
            self.voice_bypass_toggle.blockSignals(True)
            self.voice_bypass_toggle.setChecked(checked)
            self.voice_bypass_toggle.blockSignals(False)
        self._on_voice_bypass_toggle(checked)

    def _sync_settings_tts_voice_selector(self) -> None:
        """Populate Settings TTS voice menu when built after the model already loaded."""
        voices = self._cached_tts_voices
        if not voices:
            return
        settings = self._settings_view
        if settings is None or not hasattr(settings, "voice_selector"):
            return

        menu_items = [(voice, voice) for voice in voices]
        settings._build_prestige_menu(
            settings.voice_selector,
            menu_items,
            self._on_tts_voice_selected,
        )
        from ui.views.settings.widgets import register_settings_selector_width
        from core.tts_models import resolve_default_tts_voice

        register_settings_selector_width(settings.voice_selector, *voices)
        active = (
            self._tts_worker.active_voice_name
            if self._tts_worker and hasattr(self._tts_worker, "active_voice_name")
            else resolve_default_tts_voice(voices)
        )
        if active not in voices:
            active = resolve_default_tts_voice(voices)
        settings.voice_selector.setText(active)
        settings.voice_selector.update()

    def update_tts_voice_dropdowns(self, model_name: str, voices: list) -> None:
        """Populate Settings and toolbar TTS voice selectors when voices load."""
        if not voices:
            self._cached_tts_model_name = ""
            self._cached_tts_voices = []
            return

        self._cached_tts_model_name = model_name
        self._cached_tts_voices = list(voices)

        from core.tts_models import resolve_default_tts_voice

        active = (
            self._tts_worker.active_voice_name
            if self._tts_worker and hasattr(self._tts_worker, "active_voice_name")
            else resolve_default_tts_voice(voices)
        )
        if active not in voices:
            active = resolve_default_tts_voice(voices)

        menu_items = [(v, v) for v in voices]
        self._build_prestige_menu(
            self.global_voice_selector,
            menu_items,
            self._on_tts_voice_selected,
        )
        self._sync_settings_tts_voice_selector()

        self._on_tts_voice_selected(active)

    def update_global_voice_dropdown(self, model_name: str, voices: list) -> None:
        """Backward-compatible alias for TTS voice menu population."""
        self.update_tts_voice_dropdowns(model_name, voices)