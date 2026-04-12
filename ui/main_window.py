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
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QApplication, QLabel, QFrame, 
    QSizeGrip, QMenu, QSystemTrayIcon, QStackedWidget, QSizePolicy,
    QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox
)
from PyQt6.QtCore import Qt, QSize, QTimer, QEasingCurve, QPropertyAnimation, QRect
from PyQt6.QtGui import QAction, QPainter, QColor, QLinearGradient, QPixmap, QIcon, QFontMetrics
import qtawesome as qta
from ui.views.conversations_view import ConversationsView
from ui.views.settings_view import SettingsView
from ui.views.library_view import LibraryView
from ui.views.telemetry_view import TelemetryView
from ui.views.model_manager_view import ModelManagerView
from ui.components.toggle import PrestigeToggle
from core.app_settings import (
    get_engine_mode,
    get_internal_model_path,
    get_llm_models_dir,
    set_internal_model_path,
)
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

    def __init__(self, workers: dict, gpu_monitor, native_engine=None):
        super().__init__()
        # 🔑 Explicitly tell the OS what icon to use for the Taskbar/Window
        self.setWindowIcon(QIcon("assets/qube_logo_256.png"))
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

        # 🔑 3. Initialize the AI Titling Worker (FLAN-T5-Small)
        # We import it here or at the top of the file
        from workers.title_worker import TitleWorker
        self._title_worker = TitleWorker(self.db)

        # Global State
        self._is_dark_theme = True

        self._setup_ui()
        self._setup_tray()
        self._start_timers()

        # 🔑 4. Wire the AI Titling Logic
        # We wait until the UI is setup so we can access conversations_view
        self._setup_titling_connections()

    def _setup_titling_connections(self):
        """Wires the background AI to the Chat UI."""
        
        # 1. When the main LLM finishes a message, check if we need a title
        if self._llm_worker:
            self._llm_worker.response_finished.connect(self._check_for_titling)

        # 2. When the TitleWorker finishes its job, tell the sidebar to refresh
        # This keeps the UI responsive by handling the ~300ms inference in the background
        if hasattr(self, '_title_worker'):
            self._title_worker.title_generated.connect(
                lambda s_id, title: self.conversations_view._refresh_history_list()
            )

    def _check_for_titling(self, session_id, full_response):
        """Internal logic to only title 'New Conversations'."""
        # Check history length: 1 User + 1 Assistant = 2 total messages
        history = self.db.get_session_history(session_id)
        
        if len(history) == 2:
            # Get the first message (the user's prompt) to use for the title
            user_prompt = history[0]['content']
            
            # Fire and forget: TitleWorker handles the rest in the background
            self._title_worker.run_titling(user_prompt, session_id)

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
        self.main_stage.addWidget(self.telemetry_view)       # Index 2
        self.main_stage.addWidget(self.model_manager_view)   # Index 3
        self.main_stage.addWidget(self.settings_view)          # Index 4

        workspace_layout.addWidget(self.main_stage, stretch=1)
        
        # GLOBAL RIGHT TOOLBAR
        self.global_tools = self._build_tools_pane()
        workspace_layout.addWidget(self.global_tools)

        root_layout.addLayout(workspace_layout)

        # Resize Grip
        self.grip = QSizeGrip(self.main_container) 
        self.grip.setFixedSize(16, 16)

        # --- THE SYNC WIRING (Updated for new names) ---
        
        # 1. Handle Visibility Toggle
        self.settings_view.audio_pin_toggle.connect(self.audio_extra_controls.setVisible)

        # 2. Sync Settings -> Toolbar
        self.settings_view.timeout_spinner.valueChanged.connect(self.toolbar_timeout_spin.setValue)
        self.settings_view.threshold_spinner.valueChanged.connect(self.toolbar_threshold_spin.setValue)

        # 3. Sync Toolbar -> Settings
        self.toolbar_timeout_spin.valueChanged.connect(self.settings_view.timeout_spinner.setValue)
        self.toolbar_threshold_spin.valueChanged.connect(self.settings_view.threshold_spinner.setValue)

        # 4. Initialize Toolbar values from the worker
        if self._audio_worker:
            self.toolbar_timeout_spin.setValue(self._audio_worker.silence_timeout)
            self.toolbar_threshold_spin.setValue(int(self._audio_worker.speech_threshold))
            
            # Wire Toolbar directly to worker methods
            self.toolbar_timeout_spin.valueChanged.connect(self._audio_worker.set_silence_timeout)
            self.toolbar_threshold_spin.valueChanged.connect(self._audio_worker.set_speech_threshold)

        # 5. 🔑 Sync Auto-Activator Toggles
        self.settings_view.auto_activator_toggle.connect(self.rag_auto_toggle.setChecked)
        self.rag_auto_toggle.toggled.connect(self.settings_view.auto_activator_cb.setChecked)

        # 6. Internal engine model list (toolbar) — refresh when engine mode or downloads change
        # Pass the emitted mode so UI updates before/without relying on QSettings (slot order vs llm_worker).
        self.settings_view.engine_mode_changed.connect(self._refresh_toolbar_native_model_from_settings_signal)
        if hasattr(self.model_manager_view, "native_library_changed"):
            self.model_manager_view.native_library_changed.connect(
                self.refresh_toolbar_native_model_dropdown
            )
        QTimer.singleShot(0, self.refresh_toolbar_native_model_dropdown)

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

    def _build_top_bar(self) -> QFrame:
        bar = QFrame()
        bar.setFixedHeight(45)
        bar.setObjectName("TopBar")
        
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(15, 0, 15, 0)

        # --- 1. FAR LEFT: LOGO & VU METER ---
        left_container = QWidget()
        left_container.setFixedWidth(180) # Fits Logo + Padding + Mic + VU
        left_layout = QHBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12) # 12px padding between elements

        # --- Logo Setup ---
        self.app_logo = QLabel()
        from PyQt6.QtGui import QPixmap 

        logo_img = QPixmap("assets/qube_logo_256.png") 
        if not logo_img.isNull():
            self.app_logo.setPixmap(logo_img.scaled(24, 24, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            self.app_logo.setText("🧊") 
            self.app_logo.setStyleSheet("font-size: 18px;")

        # Mic Icon & VU Meter
        mic_icon = QLabel()
        mic_icon.setPixmap(qta.icon('fa5s.microphone', color='#64748b').pixmap(QSize(14, 14)))
        self.vu_meter = VUMeter()
        
        left_layout.addWidget(self.app_logo)
        left_layout.addWidget(mic_icon)
        left_layout.addWidget(self.vu_meter)
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
        min_btn.clicked.connect(self.showMinimized)

        self.max_btn = QPushButton()
        self.max_btn.setIcon(qta.icon('fa5s.expand-arrows-alt'))
        self.max_btn.setProperty("class", "WindowControlButton")
        self.max_btn.clicked.connect(self._toggle_maximize)

        close_btn = QPushButton()
        close_btn.setIcon(qta.icon('fa5s.times'))
        close_btn.setProperty("class", "WindowControlButton")
        close_btn.clicked.connect(self.hide)

        win_layout.addStretch()
        win_layout.addWidget(min_btn)
        win_layout.addWidget(self.max_btn)
        win_layout.addWidget(close_btn)

        layout.addWidget(win_controls)
        
        return bar
    
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
            # Restore rounded corners
            self.main_container.setStyleSheet(self.main_container.styleSheet().replace("border-radius: 0px;", "border-radius: 12px;"))
        else:
            self.showMaximized()
            # Update to 'Restore' icon
            self.max_btn.setIcon(qta.icon('fa5s.compress-arrows-alt'))
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
        def create_nav_btn(icon_name, index=None, size=24):
            btn = QPushButton()
            # Initial color is a muted gray; _route_view handles the active blue
            btn.setIcon(qta.icon(icon_name, color='#64748b')) 
            btn.setIconSize(QSize(size, size))
            btn.setCheckable(True)
            btn.setProperty("class", "NavButton")
            if index is not None:
                btn.clicked.connect(lambda: self._route_view(index, btn))
            return btn

        # Top Icons
        self.nav_chat = create_nav_btn('fa5s.comment-alt', 0)
        self.nav_chat.setChecked(True)
        # Highlight the first one active by default
        self.nav_chat.setIcon(qta.icon('fa5s.comment-alt', color='#89b4fa'))

        self.nav_library = create_nav_btn('fa5s.book', 1)
        self.nav_telemetry = create_nav_btn('fa5s.tachometer-alt', 2)
        self.nav_models = create_nav_btn('fa5s.microchip', 3, size=20)

        layout.addWidget(self.nav_chat, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.nav_library, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.nav_telemetry, alignment=Qt.AlignmentFlag.AlignHCenter)

        layout.addStretch()

        # Bottom Controls
        self.nav_theme = QPushButton()
        self.nav_theme.setProperty("class", "NavButton")
        self.nav_theme.setIcon(qta.icon('fa5s.moon', color='#f9e2af'))
        self.nav_theme.setIconSize(QSize(20, 20))
        self.nav_theme.clicked.connect(self._toggle_theme) 
        layout.addWidget(self.nav_theme, alignment=Qt.AlignmentFlag.AlignHCenter)

        layout.addWidget(self.nav_models, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.nav_settings = create_nav_btn('fa5s.cog', 4, size=20)
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
            self.nav_telemetry,
            self.nav_models,
            self.nav_settings,
        ]
        
        return sidebar
    
    def _build_tools_pane(self) -> QFrame:
        """Global Right Sidebar: Restored 'Card' look with animated content."""
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
        main_layout.setContentsMargins(10, 20, 20, 20)
        main_layout.setSpacing(25)

        # --- 0. LOCAL LLM (internal engine model picker) ---
        native_llm_layout = QVBoxLayout()
        native_llm_layout.setSpacing(10)
        llm_title = QLabel("LOCAL LLM")
        llm_title.setProperty("class", "ToolsPaneHeader")
        native_llm_layout.addWidget(llm_title)

        self.toolbar_native_model_selector = QPushButton()
        self.toolbar_native_model_selector.setObjectName("SettingsMenuButton")
        self.toolbar_native_model_selector.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.toolbar_native_model_selector.setIcon(qta.icon("fa5s.chevron-down", color="#64748b"))
        self.toolbar_native_model_selector.setMenu(QMenu(self.toolbar_native_model_selector))
        native_llm_layout.addWidget(self.toolbar_native_model_selector)
        main_layout.addLayout(native_llm_layout)

        # --- 1. AUDIO INPUT ---
        mic_layout = QVBoxLayout()
        mic_layout.setSpacing(10)
        m_title = QLabel("AUDIO INPUT")
        m_title.setProperty("class", "ToolsPaneHeader")
        mic_layout.addWidget(m_title)
        
        mic_row = QHBoxLayout()
        self.voice_input_toggle = PrestigeToggle()
        self.voice_input_toggle.setChecked(True)
        mic_lbl = QLabel("Enable Voice Input")
        mic_lbl.setProperty("class", "ToolsPaneControl")
        mic_row.addWidget(self.voice_input_toggle)
        mic_row.addWidget(mic_lbl)
        mic_row.addStretch()
        mic_layout.addLayout(mic_row)

        self.audio_extra_controls = QWidget()
        extra_layout = QVBoxLayout(self.audio_extra_controls)
        extra_layout.setContentsMargins(10, 5, 0, 5) 
        extra_layout.setSpacing(12)
        
        def create_mirrored_row(label_text, spinner):
            row = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setProperty("class", "ToolsPaneControl")
            lbl.setMinimumWidth(100) 
            spinner.setFixedWidth(90)
            spinner.setProperty("class", "ToolsPaneInput")
            row.addWidget(lbl)
            row.addStretch()
            row.addWidget(spinner)
            return row

        # Use the NoScroll versions we defined at the top
        self.toolbar_timeout_spin = NoScrollDoubleSpinBox()
        self.toolbar_timeout_spin.setRange(0.5, 5.0)
        self.toolbar_timeout_spin.setSingleStep(0.1)
        self.toolbar_timeout_spin.setSuffix(" sec")

        self.toolbar_threshold_spin = NoScrollSpinBox()
        self.toolbar_threshold_spin.setRange(1, 100)
        self.toolbar_threshold_spin.setSuffix("%")

        extra_layout.addLayout(create_mirrored_row("Silence Cutoff", self.toolbar_timeout_spin))
        extra_layout.addLayout(create_mirrored_row("Mic Sensitivity", self.toolbar_threshold_spin))

        mic_layout.addWidget(self.audio_extra_controls)
        self.audio_extra_controls.hide() 
        main_layout.addLayout(mic_layout)

       # --- 2. AUDIO OUTPUT & TTS ---
        voice_layout = QVBoxLayout()
        voice_layout.setSpacing(10)
        v_title = QLabel("AUDIO OUTPUT & VOICE")
        v_title.setProperty("class", "ToolsPaneHeader")
        voice_layout.addWidget(v_title)

        tts_row = QHBoxLayout()
        self.voice_bypass_toggle = PrestigeToggle()
        self.voice_bypass_toggle.setChecked(True)
        tts_label = QLabel("Enable TTS Voice")
        tts_label.setProperty("class", "ToolsPaneControl")
        tts_row.addWidget(self.voice_bypass_toggle)
        tts_row.addWidget(tts_label)
        tts_row.addStretch()
        voice_layout.addLayout(tts_row)
        
        self.global_voice_selector = QPushButton("Select Voice...")
        self.global_voice_selector.setObjectName("SettingsMenuButton")
        self.global_voice_selector.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.global_voice_selector.setIcon(qta.icon('fa5s.chevron-down', color='#64748b'))
        self.global_voice_selector.setMenu(QMenu(self.global_voice_selector))
        voice_layout.addWidget(self.global_voice_selector)
        main_layout.addLayout(voice_layout)

        def create_spinbox_row(label_text, tooltip_text, spinner):
            row = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setProperty("class", "ToolsPaneControl")
            info_icon = QLabel()
            info_icon.setPixmap(qta.icon("fa5s.info-circle", color="#64748b").pixmap(QSize(12, 12)))
            info_icon.setToolTip(tooltip_text)
            info_icon.setCursor(Qt.CursorShape.PointingHandCursor)
            row.addWidget(lbl)
            row.addWidget(info_icon)
            row.addStretch()
            spinner.setFixedWidth(90)
            row.addWidget(spinner)
            return row

        # --- 3. GENERATION PARAMETERS ---
        param_layout = QVBoxLayout()
        param_layout.setSpacing(10)
        p_title = QLabel("GENERATION PARAMETERS")
        p_title.setProperty("class", "ToolsPaneHeader")
        param_layout.addWidget(p_title)

        desc_temp = (
            "Creativity Slider: Lower values (0.1-0.3) produce strict, factual answers. "
            "Higher values (0.7-1.0) make Qube more creative."
        )
        desc_ctx = (
            "Memory Wall: Sets the absolute maximum number of tokens Qube is allowed to output in a single turn."
        )
        desc_history = (
            "Short-Term Memory: How many past messages to send to the AI. Higher values give the AI "
            "better context but consume significantly more system RAM (VRAM). Qube's background "
            "Long-Term Memory will still remember important facts even if this is set low."
        )

        self.temp_spin = NoScrollDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setValue(0.7)
        self.temp_spin.setProperty("class", "ToolsPaneInput")
        param_layout.addLayout(create_spinbox_row("Temperature:", desc_temp, self.temp_spin))

        self.ctx_spin = NoScrollSpinBox()
        self.ctx_spin.setRange(1024, 128000)
        self.ctx_spin.setValue(4096)
        self.ctx_spin.setProperty("class", "ToolsPaneInput")
        param_layout.addLayout(create_spinbox_row("Context Limit:", desc_ctx, self.ctx_spin))

        self.history_spin = NoScrollSpinBox()
        self.history_spin.setRange(2, 100)
        self.history_spin.setSingleStep(2)
        self.history_spin.setValue(10)
        self.history_spin.setProperty("class", "ToolsPaneInput")
        param_layout.addLayout(create_spinbox_row("Chat History:", desc_history, self.history_spin))

        main_layout.addLayout(param_layout)

        # --- 4. RAG ENGINE (Consolidated) ---
        rag_layout = QVBoxLayout()
        rag_layout.setSpacing(12)
        r_title = QLabel("RAG ENGINE")
        r_title.setProperty("class", "ToolsPaneHeader")
        rag_layout.addWidget(r_title)

        # 🔑 THE REFINED TOOLTIP-AWARE ROW BUILDER
        def create_toggle_row(label_text, tooltip_text, checked=False):
            row = QHBoxLayout()
            
            toggle = PrestigeToggle()
            toggle.setChecked(checked)
            # Tooltip removed from the switch
            
            lbl = QLabel(label_text)
            lbl.setProperty("class", "ToolsPaneControl")
            # Tooltip removed from the text
            
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
        
        local_row, self.tool_rag_toggle = create_toggle_row("Local Knowledge Base", desc_kb, checked=True)
        auto_row, self.rag_auto_toggle = create_toggle_row("NLP Auto-Activator", desc_auto, checked=True) 
        strict_row, self.rag_strict_toggle = create_toggle_row("Strict Isolation Mode", desc_strict)
        
        rag_layout.addLayout(local_row)
        rag_layout.addLayout(auto_row) 
        rag_layout.addLayout(strict_row)
        main_layout.addLayout(rag_layout)

        # --- 5. MCP TOOLS ---
        tools_layout = QVBoxLayout()
        tools_layout.setSpacing(12)
        t_title = QLabel("MCP TOOLS")
        t_title.setProperty("class", "ToolsPaneHeader")
        tools_layout.addWidget(t_title)

        # 🔑 FIX: Define the tooltip text and pass it as the second argument
        desc_web = "Internet Agent: Allows Qube to search the live web (via DuckDuckGo) for real-time information."
        web_row, self.tool_internet_toggle = create_toggle_row("Internet Search", desc_web)

        # 🔑 NEW: Cognitive/Hybrid Internet Mode
        desc_hybrid = "Hybrid Mode: Let Qube automatically decide when to search the internet based on context and cognitive routing."
        hybrid_row, self.tool_internet_hybrid_toggle = create_toggle_row("Hybrid Internet Mode", desc_hybrid, checked=False)
        tools_layout.addLayout(hybrid_row)
        
        tools_layout.addLayout(web_row)
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
            self._llm_worker.set_max_history_messages(self.history_spin.value())

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

            # Manual Internet Toggle (already present)
            self.tool_internet_toggle.toggled.connect(self._llm_worker.set_mcp_internet)

            # 🔑 NEW: Hybrid toggle
            def on_hybrid_toggled(checked: bool):
                # Keep the manual toggle independent
                self._llm_worker.USE_COGNITIVE_ROUTER_INTERNET = checked

            self.tool_internet_hybrid_toggle.toggled.connect(on_hybrid_toggled)

        main_layout.addStretch()
        
        # 🔑 FIX: This now matches the definition above
        outer_layout.addWidget(self.tools_content)

        return self.tools_frame
    
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
        else:
            # Collapse to just the button handle
            self.content_anim.setEndValue(0)
            self.frame_anim.setEndValue(40)
            self.toggle_tools_btn.setIcon(qta.icon('fa5s.chevron-left', color='#89b4fa'))

        self.content_anim.start()
        self.frame_anim.start()

    def _refresh_toolbar_native_model_from_settings_signal(self, mode: str) -> None:
        """Uses the value from Settings' Inference engine menu (authoritative for this UI tick)."""
        self.refresh_toolbar_native_model_dropdown(mode)

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
                btn.setEnabled(False)
                btn.setText("Managed by External Server")
                btn.setMenu(None)
                return

            btn.setEnabled(True)
            models_dir = Path(get_llm_models_dir())
            try:
                ggufs = sorted(models_dir.glob("*.gguf"), key=lambda p: p.name.lower())
            except OSError:
                ggufs = []

            fm = QFontMetrics(btn.font())
            cap_btn = max(100, btn.width() - 56)
            if btn.width() <= 1:
                cap_btn = max(100, self.tools_content.width() - 56)

            if not ggufs:
                btn.setText("(Download a model first)")
                btn.setMenu(None)
                return

            list_cap = max(100, self.tools_content.width() - 48)

            def on_pick(path: str) -> None:
                set_internal_model_path(path)
                if self._llm_worker:
                    self._llm_worker.refresh_native_model_from_settings()
                self.refresh_toolbar_native_model_dropdown()

            items = []
            for p in ggufs:
                abs_p = str(p.resolve())
                disp = fm.elidedText(p.name, Qt.TextElideMode.ElideMiddle, list_cap)
                items.append((disp, abs_p))

            self._build_prestige_menu(btn, items, on_pick)

            current = get_internal_model_path()
            matched: Path | None = None
            if current:
                try:
                    cur = Path(current).expanduser()
                    for p in ggufs:
                        if p.resolve() == cur.resolve():
                            matched = p
                            break
                except OSError:
                    matched = None

            if matched is not None:
                btn.setText(
                    fm.elidedText(matched.name, Qt.TextElideMode.ElideMiddle, cap_btn)
                )
            else:
                btn.setText(fm.elidedText("Select a model", Qt.TextElideMode.ElideMiddle, cap_btn))
        finally:
            self._apply_settings_menu_button_chevron_state(btn)

    # --- PRESTIGE MENU LOGIC ---
    def _build_prestige_menu(self, button, items, callback):
        """Builds a palette-forced QMenu with a dynamic, scrollable list."""
        from PyQt6.QtWidgets import QMenu, QWidgetAction, QListWidget, QListWidgetItem
        from PyQt6.QtCore import Qt

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

    def _route_view(self, index: int, active_button: QPushButton):
        """Switches the QStackedWidget and manages button highlights."""
        self.main_stage.setCurrentIndex(index)
        for btn in self.nav_buttons:
            if btn != active_button:
                btn.setChecked(False)
            
            # Reset icon colors to default, then highlight the active one
            if btn == self.nav_chat: btn.setIcon(qta.icon('fa5s.comment-alt', color='#89b4fa' if btn.isChecked() else '#cdd6f4'))
            elif btn == self.nav_library: btn.setIcon(qta.icon('fa5s.book', color='#89b4fa' if btn.isChecked() else '#cdd6f4'))
            elif btn == self.nav_telemetry: btn.setIcon(qta.icon('fa5s.tachometer-alt', color='#89b4fa' if btn.isChecked() else '#cdd6f4'))
            elif btn == self.nav_models: btn.setIcon(qta.icon('fa5s.microchip', color='#89b4fa' if btn.isChecked() else '#cdd6f4'))
            elif btn == self.nav_settings: btn.setIcon(qta.icon('fa5s.cog', color='#89b4fa' if btn.isChecked() else '#cdd6f4'))

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
            style_path = os.path.join("assets", "styles", "light.qss")
            if os.path.exists(style_path):
                with open(style_path, "r") as f:
                    app.setStyleSheet(f.read())
            
            self.nav_theme.setIcon(qta.icon('fa5s.sun', color='#d7827e'))
            self._is_dark_theme = False
            logger.info("Theme switched to Light Mode.")
        else:
            # --- Load Dark Theme ---
            style_path = os.path.join("assets", "styles", "base.qss")
            if os.path.exists(style_path):
                with open(style_path, "r") as f:
                    app.setStyleSheet(f.read())
                    
            self.nav_theme.setIcon(qta.icon('fa5s.moon', color='#f9e2af'))
            self._is_dark_theme = True
            logger.info("Theme switched to Dark Mode.")

        from core.richtext_styles import apply_app_link_palette

        apply_app_link_palette(app)

        # --- RE-THEME ATTACHED MENUS & LISTS ---
        
        # 1. Update the Settings Page menus
        if hasattr(self, 'settings_view') and hasattr(self.settings_view, 'refresh_menu_themes'):
            self.settings_view.refresh_menu_themes(self._is_dark_theme)
            
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

        if hasattr(self, "model_manager_view") and hasattr(
            self.model_manager_view, "refresh_after_theme_toggle"
        ):
            self.model_manager_view.refresh_after_theme_toggle()
    # ------------------------------------------------------------------ #
    #  TIMERS & TRAY                                                     #
    # ------------------------------------------------------------------ #

    def _setup_tray(self) -> None:
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(qta.icon('fa5s.cube', color='#89b4fa'))
        tray_menu = QMenu()
        tray_menu.setStyleSheet("background-color: #1e1e2e; color: #cdd6f4;")
        
        show_action = QAction("Open Workspace", self)
        show_action.triggered.connect(self.showNormal)
        quit_action = QAction("Exit Qube", self)
        quit_action.triggered.connect(QApplication.quit)

        tray_menu.addAction(show_action)
        tray_menu.addSeparator()
        tray_menu.addAction(quit_action)
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()

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
        if self.tray_icon.isVisible():
            self.hide()
            event.ignore() 
        else:
            event.accept()

    # ------------------------------------------------------------------ #
    #  PUBLIC STUBS (Keeps main.py running during transition)            #
    # ------------------------------------------------------------------ #
    # These methods receive signals from workers. Once we build the 
    # ConversationsView, we will forward these calls directly to it.

    def update_status(self, message: str) -> None:
        """Updates the top bar with a priority-based logic to prevent signal clobbering."""
        msg_upper = message.upper().strip()

        # 1. Determine the incoming state (TTS "Speaking" is separate from LLM "Thinking")
        if any(k in msg_upper for k in ["RECORDING", "LISTENING"]):
            new_state = "recording"
        elif "SPEAKING" in msg_upper:
            new_state = "speaking"
        elif any(k in msg_upper for k in ["THINKING", "GENERATING", "SYNTHESIZING", "TRANSCRIBING"]):
            new_state = "thinking"
        else:
            new_state = "idle"

        # 2. 🔑 THE PRIORITY GATE
        # Get the current state from the UI property
        current_state = self.status_bubble.property("state") or "idle"

        # Block stray Idle while recording or actively thinking (LLM), but NOT while TTS speaking —
        # playback_finished must be able to return the bar to Idle after audio ends.
        if new_state == "idle" and current_state in ["recording", "thinking"]:
            return

        # 3. Update the UI
        self.status_bubble.setText(f" {msg_upper}")
        self.status_bubble.setProperty("state", new_state)

        # Force Style Refresh
        self.status_bubble.style().unpolish(self.status_bubble)
        self.status_bubble.style().polish(self.status_bubble)

        # Lock input only for recording / LLM work; keep unlocked during TTS playback
        if hasattr(self, 'conversations_view'):
            self.conversations_view.set_input_enabled(new_state in ("idle", "speaking"))

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