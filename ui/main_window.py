"""Main window. Prefer starting the app with `python main.py` from the repo root."""

import sys
from pathlib import Path

# Running `python ui/main_window.py` does not set a package; absolute `ui.*` imports need repo root on sys.path.
if __package__ in (None, ""):
    _repo_root = Path(__file__).resolve().parent.parent
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

import psutil
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QToolButton, QLabel, QFrame,
    QSizeGrip, QMenu, QSystemTrayIcon, QStackedWidget, QSizePolicy,
    QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox, QProgressBar
)
from PyQt6.QtCore import Qt, QSize, QTimer, QEasingCurve, QPropertyAnimation, QRect
from PyQt6.QtGui import QAction, QPainter, QColor, QLinearGradient, QPixmap, QIcon, QFontMetrics
import qtawesome as qta
from core.paths import install_root, resource_path
from ui.views.conversations_view import ConversationsView
from ui.views.settings_view import SettingsView
from ui.views.library_view import LibraryView
from ui.views.memory_manager_view import MemoryManagerView
from ui.views.telemetry_view import TelemetryView
from ui.views.model_manager_view import ModelManagerView
from ui.components.toggle import PrestigeToggle
from ui.components.prestige_dialog import PrestigeDialog
from ui.components.app_notifications import AppNotificationCenter
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
from ui.onboarding.local_llm_setup_tour import build_local_llm_setup_tour
import logging

logger = logging.getLogger("Qube.UI")

class VUMeter(QWidget):
    """A sleek, custom-painted VU meter with a Green-Yellow-Red gradient."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(100, 6) # Thin, modern horizontal bar
        self._level = 0.0 # Range: 0.0 to 1.0

    def set_level(self, level: float):
        """Updates the visual level and triggers a repaint."""
        # Clamp the value between 0.0 and 1.0 for safety
        self._level = max(0.0, min(1.0, float(level)))
        self.update() 

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 1. Draw the dark background track
        painter.setBrush(QColor("#313244"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 3, 3)

        if self._level > 0:
            # 2. Calculate how far the bar should fill
            active_width = int(self.width() * self._level)
            active_rect = QRect(0, 0, active_width, self.height())

            # 3. Create the Green -> Yellow -> Red gradient
            gradient = QLinearGradient(0, 0, self.width(), 0)
            gradient.setColorAt(0.0, QColor("#a6e3a1")) # Green (Normal)
            gradient.setColorAt(0.7, QColor("#f9e2af")) # Yellow (Loud)
            gradient.setColorAt(1.0, QColor("#f38ba8")) # Red (Clipping)

            # 4. Paint the active level
            painter.setBrush(gradient)
            painter.drawRoundedRect(active_rect, 3, 3)

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
    ):
        super().__init__()
        self._project_root = install_root()
        # 🔑 Explicitly tell the OS what icon to use for the Taskbar/Window
        logo_icon_path = self._resolve_logo_asset("qube_logo_256.png")
        if logo_icon_path is not None:
            self.setWindowIcon(QIcon(str(logo_icon_path)))
        self.setWindowTitle("Qube - Workspace")
        self.setMinimumSize(1200, 800)
        self.resize(1200, 800) 

        self.workers = workers
        self.db = workers.get("db") # Ensure your DB manager is in the workers dict

        # 1. Frameless Window Setup
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
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

        # Global State
        self._is_dark_theme = True

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
        self.settings_view.engine_mode_changed.connect(
            lambda _mode: self._local_llm_tour.refresh_layout()
        )

    def showEvent(self, event):
        super().showEvent(event)
        if not getattr(self, "_onboarding_start_scheduled", False):
            self._onboarding_start_scheduled = True
            QTimer.singleShot(900, self._maybe_start_local_llm_onboarding)
        QTimer.singleShot(1500, self.schedule_scenario_replay)

    def _maybe_start_local_llm_onboarding(self) -> None:
        if get_onboarding_local_llm_tour_completed():
            return
        if not hasattr(self, "_local_llm_tour") or self._local_llm_tour.is_active:
            return
        self._local_llm_tour.start()

    def start_local_llm_onboarding_tour(self) -> None:
        """Public entry to replay the local LLM setup tour."""
        if hasattr(self, "_local_llm_tour"):
            self._local_llm_tour.start()

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

        workspace_layout = QHBoxLayout()
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(0)

        self.nav_sidebar = self._build_nav_sidebar()
        workspace_layout.addWidget(self.nav_sidebar)

        # MAIN STAGE: The QStackedWidget Router
        self.main_stage = QStackedWidget()
        self.main_stage.setStyleSheet("background-color: transparent;")
        
        # 🔑 THE FIX: Renaming to match our Titling and Hardware logic
        self.conversations_view = ConversationsView(self.workers, self.workers.get("db"))
        self.library_view = LibraryView(self.workers, self.workers.get("db"))
        self.memory_manager_view = MemoryManagerView(self.workers, self.workers.get("db"))
        self.telemetry_view = TelemetryView(
            self.workers,
            self._gpu_monitor,
            native_engine=self._native_engine,
        )
        self.model_manager_view = ModelManagerView(self.workers, self.workers.get("db"))
        self.settings_view = SettingsView(self.workers, self.workers.get("db"))
        
        # 🔑 THE FIX: Prevent UI Stretching (Policy Ignored)
        from PyQt6.QtWidgets import QSizePolicy
        self.main_stage.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        # Add them to the Stack in the correct order
        self.main_stage.addWidget(self.conversations_view)   # Index 0
        self.main_stage.addWidget(self.library_view)         # Index 1
        self.main_stage.addWidget(self.memory_manager_view)  # Index 2
        self.main_stage.addWidget(self.telemetry_view)       # Index 3
        self.main_stage.addWidget(self.model_manager_view)   # Index 4
        self.main_stage.addWidget(self.settings_view)        # Index 5

        workspace_layout.addWidget(self.main_stage, stretch=1)
        
        # GLOBAL RIGHT TOOLBAR
        self.global_tools = self._build_tools_pane()
        workspace_layout.addWidget(self.global_tools)

        root_layout.addLayout(workspace_layout)

        self.notification_center = AppNotificationCenter(self.main_container)
        self.notification_center.action_triggered.connect(self._on_notification_action)
        self.notification_center.apply_theme(self._is_dark_theme)

        # Resize Grip
        self.grip = QSizeGrip(self.main_container) 
        self.grip.setFixedSize(16, 16)

        # --- THE SYNC WIRING (Updated for new names) ---
        
        # 1. Toolbar audio extra controls visibility ↔ Settings "Pin Audio Controls"
        self.settings_view.audio_pin_toggle.connect(self.audio_extra_controls.setVisible)
        self.audio_extra_controls.setVisible(
            self.settings_view.pin_audio_cb.isChecked()
        )

        # 2. Sync Settings -> Toolbar
        self.settings_view.timeout_spinner.valueChanged.connect(self.toolbar_timeout_spin.setValue)
        self.settings_view.threshold_spinner.valueChanged.connect(self.toolbar_threshold_spin.setValue)
        if hasattr(self.settings_view, "wakeword_sensitivity"):
            self.settings_view.wakeword_sensitivity.valueChanged.connect(
                self.toolbar_wakeword_sensitivity_spin.setValue
            )

        # 3. Sync Toolbar -> Settings
        self.toolbar_timeout_spin.valueChanged.connect(self.settings_view.timeout_spinner.setValue)
        self.toolbar_threshold_spin.valueChanged.connect(self.settings_view.threshold_spinner.setValue)
        if hasattr(self.settings_view, "wakeword_sensitivity"):
            self.toolbar_wakeword_sensitivity_spin.valueChanged.connect(
                self.settings_view.wakeword_sensitivity.setValue
            )

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

        # 4b. Generation parameters: Settings ↔ Toolbar (both write through LLMWorker)
        self._wire_generation_settings_toolbar_sync()

        # 5. 🔑 Sync Auto-Activator Toggles
        self.settings_view.auto_activator_toggle.connect(self.rag_auto_toggle.setChecked)
        self.rag_auto_toggle.toggled.connect(self.settings_view.auto_activator_cb.setChecked)

        # 5b. Auto-load last model on startup (toolbar PrestigeToggle ↔ Settings checkbox)
        self.settings_view.auto_load_last_model_changed.connect(
            self._sync_toolbar_auto_load_model_toggle
        )
        self.toolbar_auto_load_model_toggle.toggled.connect(
            self._on_toolbar_auto_load_model_toggle_changed
        )

        # 6. Internal engine model list (toolbar) — refresh when engine mode or downloads change
        # Pass the emitted mode so UI updates before/without relying on QSettings (slot order vs llm_worker).
        self.settings_view.engine_mode_changed.connect(self._refresh_toolbar_native_model_from_settings_signal)
        if hasattr(self.model_manager_view, "native_library_changed"):
            self.model_manager_view.native_library_changed.connect(
                self.refresh_toolbar_native_model_dropdown
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
        cb = self.settings_view.auto_load_last_model_cb
        cb.blockSignals(True)
        cb.setChecked(checked)
        cb.blockSignals(False)

    def resizeEvent(self, event):
        """Ensures the floating resize grip stays in the bottom-right corner."""
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
        if hasattr(self, "_local_llm_tour"):
            self._local_llm_tour.refresh_layout()

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
        mic_icon = QLabel()
        mic_icon.setPixmap(qta.icon('fa5s.microphone', color='#64748b').pixmap(QSize(14, 14)))
        self.vu_meter = VUMeter()

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

        left_layout.addWidget(self.app_logo)
        left_layout.addWidget(mic_icon)
        left_layout.addWidget(self.vu_meter)
        left_layout.addWidget(self.mic_selector_btn)
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
        dummy_spacer.setFixedWidth(60) 
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
            "Knowledge base status: gray = off, blue = ready, green = retrieving"
        )
        self.rag_status_dot.setStyleSheet("color: #45475a; font-weight: bold; font-size: 11px;") 
        center_layout.addWidget(self.rag_status_dot)

        layout.addWidget(center_container)

        layout.addStretch(1)

        # --- 3. FAR RIGHT: WINDOW CONTROLS ---
        win_controls = QWidget()
        win_controls.setFixedWidth(180) # 🔑 Matches left_container to keep the center balanced
        win_layout = QHBoxLayout(win_controls)
        win_layout.setContentsMargins(0, 0, 0, 0)
        win_layout.setSpacing(8)
        
        min_btn = QPushButton()
        min_btn.setIcon(qta.icon('fa5s.minus'))
        min_btn.setProperty("class", "WindowControlButton")
        min_btn.setToolTip("Minimize window")
        min_btn.clicked.connect(self.showMinimized)

        self.max_btn = QPushButton()
        self.max_btn.setIcon(qta.icon('fa5s.expand-arrows-alt'))
        self.max_btn.setProperty("class", "WindowControlButton")
        self.max_btn.setToolTip("Maximize window")
        self.max_btn.clicked.connect(self._toggle_maximize)

        close_btn = QPushButton()
        close_btn.setIcon(qta.icon('fa5s.times'))
        close_btn.setProperty("class", "WindowControlButton")
        close_btn.setToolTip("Minimize to system tray")
        close_btn.clicked.connect(self.hide)

        win_layout.addStretch()
        win_layout.addWidget(min_btn)
        win_layout.addWidget(self.max_btn)
        win_layout.addWidget(close_btn)

        layout.addWidget(win_controls)
        
        return bar

    def _apply_topbar_mic_chevron_style(self) -> None:
        chevron_color = "#64748b"
        hover = "rgba(148, 163, 184, 0.18)" if not getattr(self, "_is_dark_theme", True) else "rgba(205, 214, 244, 0.08)"
        self.mic_selector_btn.setIcon(qta.icon("fa5s.chevron-down", color=chevron_color))
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
        settings = getattr(self, "settings_view", None)
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
            self.vu_meter.set_level(level)
    
    def set_rag_state(self, state: str) -> None:
        """Manages the Traffic Light colors of the RAG indicator."""
        if state == 'off':
            color = "#45475a" # Dark Slate / Black
        elif state == 'standby':
            color = "#89b4fa" # Qube Blue (User activated it)
        elif state == 'active':
            color = "#a6e3a1" # Green (App is fetching data)
        else:
            return

        self.rag_status_dot.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 11px;")
    
    def _toggle_maximize(self):
        """Toggles between maximized and normal window states."""
        if self.isMaximized():
            self.showNormal()
            # Update to 'Maximize' icon
            self.max_btn.setIcon(qta.icon('fa5s.expand-arrows-alt'))
            self.max_btn.setToolTip("Maximize window")
            # Restore rounded corners
            self.main_container.setStyleSheet(self.main_container.styleSheet().replace("border-radius: 0px;", "border-radius: 12px;"))
        else:
            self.showMaximized()
            # Update to 'Restore' icon
            self.max_btn.setIcon(qta.icon('fa5s.compress-arrows-alt'))
            self.max_btn.setToolTip("Restore window")
            # Flatten corners for full-screen look
            self.main_container.setStyleSheet(self.main_container.styleSheet().replace("border-radius: 12px;", "border-radius: 0px;"))

    def _build_nav_sidebar(self) -> QFrame:
        """Global Left Navigation: Switches views and shows mini-telemetry."""
        sidebar = QFrame()
        sidebar.setFixedWidth(70)
        sidebar.setObjectName("NavSidebar")

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 20, 0, 20)
        layout.setSpacing(25)

        # Helper to create consistent Nav Buttons
        def create_nav_btn(icon_name, index=None, size=24, tooltip=None):
            btn = QPushButton()
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
        self.nav_memory = create_nav_btn('fa5s.brain', 2, size=22, tooltip="Memory Manager")
        self.nav_memory.setObjectName("NavMemory")
        self.nav_telemetry = create_nav_btn('fa5s.tachometer-alt', 3, tooltip="Telemetry")
        self.nav_telemetry.setObjectName("NavTelemetry")
        self.nav_models = create_nav_btn('fa5s.microchip', 4, size=20, tooltip="Model Manager")
        self.nav_models.setObjectName("NavModels")

        layout.addWidget(self.nav_chat, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.nav_library, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.nav_memory, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.nav_telemetry, alignment=Qt.AlignmentFlag.AlignHCenter)

        layout.addStretch()

        # Bottom Controls
        self.nav_theme = QPushButton()
        self.nav_theme.setObjectName("NavThemeToggle")
        self.nav_theme.setProperty("class", "NavButton")
        self.nav_theme.setIcon(qta.icon('fa5s.moon', color='#f9e2af'))
        self.nav_theme.setIconSize(QSize(20, 20))
        self.nav_theme.setFixedSize(44, 44)
        self.nav_theme.setToolTip("Switch to light theme")
        self.nav_theme.clicked.connect(self._toggle_theme)
        layout.addWidget(self.nav_theme, alignment=Qt.AlignmentFlag.AlignHCenter)

        layout.addWidget(self.nav_models, alignment=Qt.AlignmentFlag.AlignHCenter)

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

        # Style mapping: Hex colors match TelemetryView exactly
        metrics = [
            (self.side_cpu_lbl, "#10b981"), # Emerald
            (self.side_ram_lbl, "#3b82f6"), # Blue
            (self.side_gpu_lbl, "#8b5cf6")  # Purple
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
            self.nav_telemetry,
            self.nav_models,
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
        self.toggle_tools_btn.setIcon(qta.icon('fa5s.chevron-right', color='#89b4fa'))
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
        self.toolbar_native_model_selector.setIcon(qta.icon("fa5s.chevron-down", color="#64748b"))
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
            qta.icon("fa5s.info-circle", color="#64748b").pixmap(QSize(12, 12))
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
                info_icon.setPixmap(qta.icon("fa5s.info-circle", color="#64748b").pixmap(QSize(12, 12)))
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
        self.global_voice_selector.setIcon(qta.icon('fa5s.chevron-down', color='#64748b'))
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
            info_icon.setPixmap(qta.icon("fa5s.info-circle", color="#64748b").pixmap(QSize(12, 12)))
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
            "Creativity Slider: Lower values (0.1-0.3) produce strict, factual answers. "
            "Higher values (0.7-1.0) make Qube more creative."
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
            info_icon.setPixmap(qta.icon('fa5s.info-circle', color='#64748b').pixmap(QSize(12, 12)))
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

        # --- 5. MCP TOOLS ---
        tools_layout = QVBoxLayout()
        tools_layout.setSpacing(_TOOLS_INNER_V_SPACING)
        t_title = QLabel("MCP TOOLS")
        t_title.setProperty("class", "ToolsPaneHeader")
        tools_layout.addWidget(t_title)

        # 🔑 NEW: Cognitive/Hybrid Internet Mode
        desc_hybrid = "Hybrid Mode: Let Qube automatically decide when to search the internet based on context and cognitive routing."
        hybrid_row, self.tool_internet_hybrid_toggle = create_toggle_row(
            "Hybrid Internet Mode", desc_hybrid, checked=get_mcp_internet_hybrid_enabled()
        )
        tools_layout.addLayout(hybrid_row)
        main_layout.addLayout(tools_layout)
        outer_layout.addWidget(self.tools_content)
        # --------------------------------------------------------- #
        #  WIRING TO WORKERS                                        #
        # --------------------------------------------------------- #
        if self._audio_worker:
            self.voice_input_toggle.toggled.connect(lambda checked: self._audio_worker.set_paused(not checked))
            # 🔑 Catch the volume signal and route it to the VU meter
            self._audio_worker.volume_update.connect(self.update_mic_level)

        if self._tts_worker:
            self.voice_bypass_toggle.toggled.connect(lambda checked: self._tts_worker.set_mute(not checked))
        if self._llm_worker:
            self._llm_worker.response_finished.connect(self._check_for_titling)
            self.temp_spin.valueChanged.connect(self._llm_worker.set_temperature)
            self.ctx_spin.valueChanged.connect(self._llm_worker.set_context_window)
            self.history_spin.valueChanged.connect(self._llm_worker.set_max_history_messages)

            # 🔑 THE NEW RAG WIRING
            def on_rag_toggled(checked):
                self.set_rag_state('standby' if checked else 'off')
                self._llm_worker.set_mcp_rag(checked)
                
            self.tool_rag_toggle.toggled.connect(on_rag_toggled)
            
            # Force initial state check on boot
            self.set_rag_state('standby' if self.tool_rag_toggle.isChecked() else 'off')

            # 🔑 THE NEW STRICT WIRE
            self.rag_strict_toggle.toggled.connect(self._llm_worker.set_mcp_strict)
            # 🔑 THE NEW AUTO-ACTIVATOR WIRE
            self.rag_auto_toggle.toggled.connect(self._llm_worker.set_mcp_auto)

            # Hybrid toggle controls web search + cognitive auto-web routing.
            def on_hybrid_toggled(checked: bool):
                self._llm_worker.set_mcp_internet_hybrid(checked)

            self.tool_internet_hybrid_toggle.toggled.connect(on_hybrid_toggled)
            # Seed worker state from the current toggle value.
            on_hybrid_toggled(self.tool_internet_hybrid_toggle.isChecked())

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

    def _wire_generation_settings_toolbar_sync(self) -> None:
        """Keep Settings and toolbar generation spinboxes aligned (audio-style sync)."""
        if not self._llm_worker:
            return
        sv = getattr(self, "settings_view", None)
        if sv is None:
            return
        pairs = (
            (self.temp_spin, getattr(sv, "llm_temp_spin", None)),
            (self.ctx_spin, getattr(sv, "llm_ctx_spin", None)),
            (self.history_spin, getattr(sv, "llm_history_spin", None)),
        )
        for toolbar_spin, settings_spin in pairs:
            if toolbar_spin is None or settings_spin is None:
                continue
            settings_spin.valueChanged.connect(toolbar_spin.setValue)
            toolbar_spin.valueChanged.connect(settings_spin.setValue)
        self._apply_toolbar_generation_spin_values()

    def _toggle_tools_pane(self):
        """Animates the collapse of the content while keeping the handle visible."""
        # Check if we are currently collapsed (width is small)
        is_collapsed = self.tools_content.maximumWidth() == 0
        
        # 1. Animate the Content Area
        self.content_anim = QPropertyAnimation(self.tools_content, b"maximumWidth")
        self.content_anim.setDuration(350)
        self.content_anim.setEasingCurve(QEasingCurve.Type.InOutQuart)

        # 2. Animate the Outer Frame (The 'Bar' background)
        self.frame_anim = QPropertyAnimation(self.tools_frame, b"maximumWidth")
        self.frame_anim.setDuration(350)
        self.frame_anim.setEasingCurve(QEasingCurve.Type.InOutQuart)

        if is_collapsed:
            # Expand to full size
            self.content_anim.setEndValue(260)
            self.frame_anim.setEndValue(300)
            self.toggle_tools_btn.setIcon(qta.icon('fa5s.chevron-right', color='#89b4fa'))
            self.toggle_tools_btn.setToolTip("Hide tools panel")
        else:
            # Collapse to just the button handle
            self.content_anim.setEndValue(0)
            self.frame_anim.setEndValue(40)
            self.toggle_tools_btn.setIcon(qta.icon('fa5s.chevron-left', color='#89b4fa'))
            self.toggle_tools_btn.setToolTip("Show tools panel")

        self.content_anim.start()
        self.frame_anim.start()

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
        """QtAwesome icons ignore QSS; match chevron to #SettingsMenuButton enabled/disabled look."""
        is_dark = getattr(self, "_is_dark_theme", True)
        muted = "#3f3f46" if is_dark else "#a1a1aa"
        active = "#64748b"
        color = active if button.isEnabled() else muted
        button.setIcon(qta.icon("fa5s.chevron-down", color=color))

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
            if hasattr(self, "settings_view") and hasattr(
                self.settings_view, "sync_active_native_model_label"
            ):
                self.settings_view.sync_active_native_model_label()

    def _apply_native_model_eject_button_style(self) -> None:
        if not hasattr(self, "toolbar_native_model_eject_btn"):
            return
        btn = self.toolbar_native_model_eject_btn
        btn.setStyleSheet(
            """
            QPushButton#NativeModelEjectButton {
                background: transparent;
                border: none;
                border-radius: 6px;
            }
            QPushButton#NativeModelEjectButton:hover:enabled {
                background: rgba(139, 92, 246, 0.12);
            }
            QPushButton#NativeModelEjectButton:disabled {
                background: transparent;
            }
            """
        )
        is_dark = getattr(self, "_is_dark_theme", True)
        muted = "#3f3f46" if is_dark else "#a1a1aa"
        color = "#8b5cf6" if btn.isEnabled() else muted
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
        if loading:
            bar.setRange(0, 0)  # indeterminate, no layout shift
            bar.setStyleSheet(
                """
                QProgressBar {
                    background: rgba(255, 255, 255, 0.08);
                    border: none;
                    border-radius: 2px;
                }
                QProgressBar::chunk {
                    background-color: #8b5cf6;
                    border-radius: 2px;
                }
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
        if inactive:
            btn.setStyleSheet(
                "color: #64748b; font-style: italic;"
            )
        elif success:
            btn.setStyleSheet("color: #10b981; font-weight: 600;")
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
        from PyQt6.QtGui import QPalette, QColor
        palette = QPalette()

        if is_dark:
            bg      = QColor("#1e1e2e")
            fg      = QColor("#cdd6f4")
            sel_bg  = QColor("#313244")
            sel_fg  = QColor("#cdd6f4")
            border  = "rgba(255, 255, 255, 0.1)"
            hover   = "#313244"
        else:
            bg      = QColor("#ffffff")
            fg      = QColor("#1e293b")
            sel_bg  = QColor("#f1f5f9")
            sel_fg  = QColor("#0f172a")
            border  = "#cbd5e1"
            hover   = "#f1f5f9"

        for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base):
            palette.setColor(role, bg)
        palette.setColor(QPalette.ColorRole.WindowText, fg)
        palette.setColor(QPalette.ColorRole.Text, fg)
        palette.setColor(QPalette.ColorRole.Highlight, sel_bg)
        palette.setColor(QPalette.ColorRole.HighlightedText, sel_fg)

        menu.setPalette(palette)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {bg.name()};
                border: 1px solid {border};
                border-radius: 6px;
                padding: 4px;
            }}
            /* Style the embedded list */
            QListWidget#PrestigeMenuList {{
                background-color: transparent;
                border: none;
                outline: none;
            }}
            QListWidget#PrestigeMenuList::item {{
                background-color: transparent;
                color: {fg.name()};
                padding: 8px 25px;
                border-radius: 4px;
                min-height: 24px;
            }}
            QListWidget#PrestigeMenuList::item:selected, 
            QListWidget#PrestigeMenuList::item:hover {{
                background-color: {hover};
                color: {sel_fg.name()};
            }}
            
            /* Sleek internal scrollbar */
            QScrollBar:vertical {{
                border: none;
                background: transparent;
                width: 6px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {border};
                border-radius: 3px;
                min-height: 20px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)

    def _handle_selection(self, button, label, data, callback):
        button.setText(label)
        callback(data)

    def _nav_icon_colors(self) -> tuple[str, str]:
        active = "#89b4fa"
        inactive = "#cdd6f4" if self._is_dark_theme else "#64748b"
        return active, inactive

    def _refresh_nav_btn_icon(self, btn: QPushButton) -> None:
        icon_name = getattr(btn, "_nav_fa_icon", None)
        if not icon_name:
            return
        size = getattr(btn, "_nav_icon_size", 24)
        active_color, inactive_color = self._nav_icon_colors()
        color = active_color if btn.isChecked() else inactive_color
        btn.setIcon(qta.icon(icon_name, color=color))
        btn.setIconSize(QSize(size, size))

    def _route_view(self, index: int, active_button: QPushButton):
        """Switches the QStackedWidget and manages button highlights.

        Updates icons only for the previous and newly active buttons to avoid
        rebuilding all nav pixmaps each click (noticeable flicker on Windows).
        """
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
        if index == 0 and hasattr(self, "conversations_view"):
            QTimer.singleShot(0, self.conversations_view.focus_composer_if_ready)

    def _toggle_theme(self):
        """Toggles the global theme and resets the system palette to prevent 'Ghosting'."""
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtGui import QPalette
        import os
        import qtawesome as qta

        app = QApplication.instance()
        
        # 1. THE FULL RESET
        # This kills the 'Black' system background that is haunting your Light Mode
        app.setPalette(app.style().standardPalette()) 
        app.setStyleSheet("") 

        if self._is_dark_theme:
            # --- Load Light Theme ---
            style_path = resource_path("assets", "styles", "light.qss")
            if style_path.is_file():
                with open(style_path, "r") as f:
                    app.setStyleSheet(f.read())
            
            self.nav_theme.setIcon(qta.icon('fa5s.sun', color='#d7827e'))
            self.nav_theme.setToolTip("Switch to dark theme")
            self._is_dark_theme = False
            qube_tooltip_set_theme(False)
            logger.info("Theme switched to Light Mode.")
        else:
            # --- Load Dark Theme ---
            style_path = resource_path("assets", "styles", "base.qss")
            if style_path.is_file():
                with open(style_path, "r") as f:
                    app.setStyleSheet(f.read())
                    
            self.nav_theme.setIcon(qta.icon('fa5s.moon', color='#f9e2af'))
            self.nav_theme.setToolTip("Switch to light theme")
            self._is_dark_theme = True
            qube_tooltip_set_theme(True)
            logger.info("Theme switched to Dark Mode.")

        from core.richtext_styles import apply_app_link_palette

        apply_app_link_palette(app)

        # --- RE-THEME ATTACHED MENUS & LISTS ---
        
        # 1. Update the Settings Page menus
        if hasattr(self, 'settings_view') and hasattr(self.settings_view, 'refresh_menu_themes'):
            self.settings_view.refresh_menu_themes(self._is_dark_theme)
        if hasattr(self, "_topbar_mic_menu"):
            self._apply_menu_theme(self._topbar_mic_menu, self._is_dark_theme)
        self._apply_topbar_mic_chevron_style()
            
        # 2. Update the Toolbar Voice Menu
        if hasattr(self, 'global_voice_selector'):
            toolbar_menu = self.global_voice_selector.menu()
            if toolbar_menu:
                self._apply_menu_theme(toolbar_menu, self._is_dark_theme)

        # 2b. Toolbar internal LLM model menu
        if hasattr(self, "toolbar_native_model_selector"):
            native_menu = self.toolbar_native_model_selector.menu()
            if native_menu:
                self._apply_menu_theme(native_menu, self._is_dark_theme)
            self._apply_settings_menu_button_chevron_state(self.toolbar_native_model_selector)
            self._apply_native_model_eject_button_style()

        # 3. 🔑 THE FIX: Update Conversations View
        if hasattr(self, 'conversations_view'):
            if hasattr(self.conversations_view, 'refresh_menu_themes'):
                self.conversations_view.refresh_menu_themes(self._is_dark_theme)
            if hasattr(self.conversations_view, 'refresh_button_themes'):
                self.conversations_view.refresh_button_themes(self._is_dark_theme)
            if hasattr(self.conversations_view, '_update_row_colors'):
                self.conversations_view._update_row_colors() # Force text repaint instantly!

        # 4. 🔑 THE FIX: Update Library View
        if hasattr(self, 'library_view'):
            if hasattr(self.library_view, 'refresh_menu_themes'):
                self.library_view.refresh_menu_themes(self._is_dark_theme)
            if hasattr(self.library_view, 'refresh_button_themes'):
                self.library_view.refresh_button_themes(self._is_dark_theme)
            if hasattr(self.library_view, '_update_row_colors'):
                self.library_view._update_row_colors() # Force text repaint instantly!

        if hasattr(self, "memory_manager_view") and hasattr(
            self.memory_manager_view, "refresh_theme"
        ):
            self.memory_manager_view.refresh_theme(self._is_dark_theme)
        if hasattr(self, "model_manager_view") and hasattr(
            self.model_manager_view, "refresh_after_theme_toggle"
        ):
            self.model_manager_view.refresh_after_theme_toggle()
        if hasattr(self, "telemetry_view") and hasattr(
            self.telemetry_view, "refresh_after_theme_toggle"
        ):
            self.telemetry_view.refresh_after_theme_toggle()
        if hasattr(self, "notification_center"):
            self.notification_center.apply_theme(self._is_dark_theme)
        if self.tray_controller is not None:
            self.tray_controller.apply_theme(self._is_dark_theme)
        if self._companion_controller is not None:
            self._companion_controller.apply_theme(self._is_dark_theme)

        for btn in getattr(self, "nav_buttons", ()):
            self._refresh_nav_btn_icon(btn)

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
        if hasattr(self, "settings_view") and hasattr(self.settings_view, "companion_enabled_cb"):
            self.settings_view.companion_enabled_cb.blockSignals(True)
            self.settings_view.companion_enabled_cb.setChecked(enabled)
            self.settings_view.companion_enabled_cb.blockSignals(False)
        if self._companion_controller is not None:
            self._companion_controller.set_user_enabled(enabled)

    def _on_tray_voice_input_toggled(self, enabled: bool) -> None:
        if hasattr(self, "voice_input_toggle"):
            self.voice_input_toggle.blockSignals(True)
            self.voice_input_toggle.setChecked(enabled)
            self.voice_input_toggle.blockSignals(False)
        if self._audio_worker is not None:
            self._audio_worker.set_paused(not enabled)
        self._activity_reducer.set_voice_paused(not enabled)
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
        self._sync_tray_voice_toggles()

    def _sync_tray_voice_toggles(self, *_args) -> None:
        if self.tray_controller is None:
            return
        voice_in = self.voice_input_toggle.isChecked() if hasattr(self, "voice_input_toggle") else True
        voice_out = self.voice_bypass_toggle.isChecked() if hasattr(self, "voice_bypass_toggle") else True
        self.tray_controller.sync_voice_toggles(voice_in=voice_in, voice_out=voice_out)
        self._activity_reducer.set_voice_paused(not voice_in)
        self._presence_service.set_voice_output_muted(not voice_out)
        self._sync_tray_presence()

    def _on_tray_navigate(self, action_id: str) -> None:
        self._on_notification_action(action_id)

    def _sync_tray_presence(self) -> None:
        if self.tray_controller is None:
            return
        voice_paused = bool(
            self._audio_worker and getattr(self._audio_worker, "is_paused", False)
        )
        self.tray_controller.set_activity(
            self._activity_reducer.activity,
            voice_paused=voice_paused,
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
        if hasattr(self, 'telemetry_view'):
            # These match the object names in your telemetry_view.py
            self.telemetry_view.live_cpu_lbl.setText(f"CPU: {cpu}%")
            self.telemetry_view.live_ram_lbl.setText(f"RAM: {ram}%")
            self.telemetry_view.live_gpu_lbl.setText(f"GPU: {gpu}%")

    # ------------------------------------------------------------------ #
    #  FRAMELESS DRAG & DROP EVENT ROUTING                               #
    # ------------------------------------------------------------------ #

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.top_bar.underMouse():
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
        elif action_id == "open_models":
            self._restore_workspace_from_tray()
            if hasattr(self, "nav_models"):
                self.nav_models.setChecked(True)
                self._route_view(4, self.nav_models)
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

    def request_application_restart(self) -> None:
        self._force_app_exit = True
        if not relaunch_and_quit():
            PrestigeDialog(
                self,
                "Restart failed",
                manual_restart_instructions(),
                is_dark=self._is_dark_theme,
            ).exec()

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
            label = transition.display_text.strip()
            self.conversations_view.update_action_placeholder(label)
            if new_state == "idle":
                self.conversations_view.on_turn_complete_idle()
            elif new_state == "listening":
                self.conversations_view.on_voice_capture_started()
            else:
                if getattr(self.conversations_view, "_voice_capture_active", False):
                    self.conversations_view.on_voice_capture_ended()
                self.conversations_view.set_input_enabled(
                    new_state in ("idle", "speaking", "needs_model")
                )

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
        """Called by the LLM Worker when actively retrieving documents."""
        # Only switch to green if the toggle is actually turned on
        if self.tool_rag_toggle.isChecked():
            self.set_rag_state('active' if active else 'standby')

    def log_user_message(self, text: str) -> None:
        pass # Will be forwarded to ConversationsView

    def log_agent_token(self, token: str) -> None:
        pass # Will be forwarded to ConversationsView

    def update_stt_latency(self, ms: float) -> None:
        if hasattr(self, 'telemetry_view'):
            self.telemetry_view.update_stt_latency(ms)

    def update_ttft_latency(self, ms: float) -> None:
        if hasattr(self, 'telemetry_view'):
            self.telemetry_view.update_ttft_latency(ms)

    def on_audio_volume_update(self, level: float) -> None:
        self._presence_service.set_audio_level(level)

    def on_tts_playback_level(self, level: float) -> None:
        if self._companion_controller is not None:
            self._companion_controller.set_speech_level(level)

    def update_tts_latency(self, ms: float) -> None:
        if hasattr(self, 'telemetry_view'):
            self.telemetry_view.update_tts_latency(ms)

    def update_global_voice_dropdown(self, model_name: str, voices: list) -> None:
        """Receives loaded voices from the TTS worker and populates the global toolbar."""
        if not voices:
            return

        self._build_prestige_menu(
            self.global_voice_selector,
            [(v, v) for v in voices],
            lambda v: self._tts_worker.set_voice(v) if self._tts_worker else None
        )
        
        self.global_voice_selector.setText(voices[0])
        if self._tts_worker:
            self._tts_worker.set_voice(voices[0])