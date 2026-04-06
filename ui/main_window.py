import psutil
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QApplication, QLabel, QFrame, 
    QSizeGrip, QMenu, QSystemTrayIcon, QStackedWidget, QSizePolicy
)
from PyQt6.QtCore import Qt, QSize, QTimer
from PyQt6.QtGui import QAction
import qtawesome as qta
from .views.conversations_view import ConversationsView
from .views.settings_view import SettingsView
import logging

logger = logging.getLogger("Qube.UI")

class MainWindow(QMainWindow):
    """
    MASTER GLOBAL SHELL
    Responsible for the frameless lifecycle, global navigation, and routing.
    All distinct screens are hosted within the QStackedWidget (Main Stage).
    """

    def __init__(self, workers: dict, gpu_monitor):
        super().__init__()
        self.setWindowTitle("Qube - Workspace")
        self.resize(1200, 800) 

        self.workers = workers

        # 1. Frameless Window Setup
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._old_pos = None

        # 2. Worker References
        self._audio_worker = workers.get("audio")
        self._tts_worker   = workers.get("tts")
        self._llm_worker   = workers.get("llm")
        self._gpu_monitor  = gpu_monitor

        # Global State
        self._is_dark_theme = True

        self._setup_ui()
        self._setup_tray()
        self._start_timers()

    # ------------------------------------------------------------------ #
    #  UI CONSTRUCTION                                                   #
    # ------------------------------------------------------------------ #

    def _setup_ui(self) -> None:
        # Base container matching the Catppuccin dark theme
        self.main_container = QFrame()
        self.main_container.setObjectName("MainContainer")
        self.main_container.setStyleSheet("""
            #MainContainer {
                background-color: #1e1e2e; 
                border: 1px solid #45475a; 
                border-radius: 12px;
            }
        """)
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
        
        # --- NEW: Injecting the actual Conversations View ---
        # Note: We assume 'db_manager' was either passed to MainWindow or is in your workers dict. 
        # If it's in your workers dict, use: workers.get("db")
        self.view_conversations = ConversationsView(self.workers, self.workers.get("db"))

        # --- PLACEHOLDER VIEWS ---
        # We will replace these with actual imported View classes next.
        self.view_library = QLabel("<h2 style='color:#cdd6f4; text-align:center;'>Library View (Pending)</h2>")
        self.view_telemetry = QLabel("<h2 style='color:#cdd6f4; text-align:center;'>Telemetry View (Pending)</h2>")
        

        self.view_settings = SettingsView(self.workers, self.workers.get("db"))
        
        self.main_stage.addWidget(self.view_conversations) # Index 0
        self.main_stage.addWidget(self.view_library)       # Index 1
        self.main_stage.addWidget(self.view_telemetry)     # Index 2
        self.main_stage.addWidget(self.view_settings)      # Index 3

        workspace_layout.addWidget(self.main_stage, stretch=1)
        root_layout.addLayout(workspace_layout)

        # Resize Grip
        self.grip = QSizeGrip(self)
        root_layout.addWidget(self.grip, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)

    def _build_top_bar(self) -> QFrame:
        """Global Top Bar: Houses global AI status, RAG indicator, and window controls."""
        bar = QFrame()
        bar.setFixedHeight(45)
        bar.setStyleSheet("background-color: rgba(17, 17, 27, 0.9); border-top-left-radius: 12px; border-top-right-radius: 12px; border-bottom: 1px solid #313244;")
        
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(15, 0, 15, 0)

        # Left Spacer to balance the layout
        layout.addStretch(1)

        # Center: Global Worker Status Bubble
        self.status_bubble = QLabel(" IDLE")
        self.status_bubble.setFixedSize(200, 26)
        self.status_bubble.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_bubble.setStyleSheet("""
            background-color: #313244; color: #a6e3a1; font-weight: bold; 
            border-radius: 13px; font-size: 11px; letter-spacing: 1px;
        """)
        layout.addWidget(self.status_bubble)

        # Right Spacing
        layout.addStretch(1)

        # Right: RAG Indicator
        self.rag_status_dot = QLabel("● RAG")
        self.rag_status_dot.setStyleSheet("color: #6c7086; font-weight: bold; font-size: 11px; margin-right: 15px;")
        layout.addWidget(self.rag_status_dot)

        # Window Controls
        btn_style = "QPushButton { border: none; padding: 5px; border-radius: 4px; } QPushButton:hover { background-color: #45475a; }"
        
        min_btn = QPushButton()
        min_btn.setIcon(qta.icon('fa5s.minus', color='#cdd6f4'))
        min_btn.setStyleSheet(btn_style)
        min_btn.clicked.connect(self.showMinimized)

        close_btn = QPushButton()
        close_btn.setIcon(qta.icon('fa5s.times', color='#f38ba8'))
        close_btn.setStyleSheet(btn_style)
        close_btn.clicked.connect(self.hide)

        layout.addWidget(min_btn)
        layout.addWidget(close_btn)
        
        return bar

    def _build_nav_sidebar(self) -> QFrame:
        """Global Left Navigation: Switches views and shows mini-telemetry."""
        sidebar = QFrame()
        sidebar.setFixedWidth(70)
        sidebar.setStyleSheet("background-color: rgba(24, 24, 37, 0.9); border-bottom-left-radius: 12px; border-right: 1px solid #313244;")
        
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 20, 0, 20)
        layout.setSpacing(25)

        # Reusable Nav Button Style
        nav_style = """
            QPushButton { border: none; background: transparent; border-radius: 8px; padding: 10px; }
            QPushButton:hover { background-color: #313244; }
            QPushButton:checked { background-color: #45475a; }
        """

        # Top Icons
        self.nav_chat = QPushButton()
        self.nav_chat.setIcon(qta.icon('fa5s.comment-alt', color='#89b4fa'))
        self.nav_chat.setIconSize(QSize(24, 24))
        self.nav_chat.setCheckable(True)
        self.nav_chat.setChecked(True) # Default active
        self.nav_chat.setStyleSheet(nav_style)
        self.nav_chat.clicked.connect(lambda: self._route_view(0, self.nav_chat))
        layout.addWidget(self.nav_chat, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.nav_library = QPushButton()
        self.nav_library.setIcon(qta.icon('fa5s.book', color='#cdd6f4'))
        self.nav_library.setIconSize(QSize(24, 24))
        self.nav_library.setCheckable(True)
        self.nav_library.setStyleSheet(nav_style)
        self.nav_library.clicked.connect(lambda: self._route_view(1, self.nav_library))
        layout.addWidget(self.nav_library, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.nav_telemetry = QPushButton()
        self.nav_telemetry.setIcon(qta.icon('fa5s.tachometer-alt', color='#cdd6f4'))
        self.nav_telemetry.setIconSize(QSize(24, 24))
        self.nav_telemetry.setCheckable(True)
        self.nav_telemetry.setStyleSheet(nav_style)
        self.nav_telemetry.clicked.connect(lambda: self._route_view(2, self.nav_telemetry))
        layout.addWidget(self.nav_telemetry, alignment=Qt.AlignmentFlag.AlignHCenter)

        layout.addStretch()

        # Bottom Controls
        self.nav_theme = QPushButton()
        self.nav_theme.setIcon(qta.icon('fa5s.moon', color='#f9e2af'))
        self.nav_theme.setIconSize(QSize(20, 20))
        self.nav_theme.setStyleSheet(nav_style)
        layout.addWidget(self.nav_theme, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.nav_settings = QPushButton()
        self.nav_settings.setIcon(qta.icon('fa5s.cog', color='#cdd6f4'))
        self.nav_settings.setIconSize(QSize(20, 20))
        self.nav_settings.setCheckable(True)
        self.nav_settings.setStyleSheet(nav_style)
        self.nav_settings.clicked.connect(lambda: self._route_view(3, self.nav_settings))
        layout.addWidget(self.nav_settings, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Mini Telemetry Block
        self.mini_telemetry = QLabel("CPU: --\nRAM: --\nGPU: --")
        self.mini_telemetry.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mini_telemetry.setStyleSheet("color: #6c7086; font-size: 9px; font-weight: bold; font-family: monospace;")
        layout.addWidget(self.mini_telemetry, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Track navigation buttons to manage visual 'active' state
        self.nav_buttons = [self.nav_chat, self.nav_library, self.nav_telemetry, self.nav_settings]
        
        return sidebar

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
            elif btn == self.nav_settings: btn.setIcon(qta.icon('fa5s.cog', color='#89b4fa' if btn.isChecked() else '#cdd6f4'))

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
        ram = int(psutil.virtual_memory().percent)
        cpu = int(psutil.cpu_percent())
        gpu = int(self._gpu_monitor.get_load()) if self._gpu_monitor else 0
        self.mini_telemetry.setText(f"CPU: {cpu}%\nRAM: {ram}%\nGPU: {gpu}%")

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
        if event.button() == Qt.MouseButton.LeftButton and self.top_bar.underMouse():
            if self.isMaximized():
                self.showNormal()
                self.main_container.setStyleSheet(self.main_container.styleSheet().replace("border-radius: 0px;", "border-radius: 12px;"))
            else:
                self.showMaximized()
                self.main_container.setStyleSheet(self.main_container.styleSheet().replace("border-radius: 12px;", "border-radius: 0px;"))

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
        self.status_bubble.setText(f" {message.upper()}")
        if "RECORDING" in message:
            self.status_bubble.setStyleSheet("background-color: #313244; color: #f38ba8; font-weight: bold; border-radius: 13px; font-size: 11px;")
        elif "Thinking" in message:
            self.status_bubble.setStyleSheet("background-color: #313244; color: #f9e2af; font-weight: bold; border-radius: 13px; font-size: 11px;")
        else:
            self.status_bubble.setStyleSheet("background-color: #313244; color: #a6e3a1; font-weight: bold; border-radius: 13px; font-size: 11px;")

    def update_rag_indicator(self, active: bool) -> None:
        color = "#a6e3a1" if active else "#6c7086"
        self.rag_status_dot.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 11px; margin-right: 15px;")

    def log_user_message(self, text: str) -> None:
        pass # Will be forwarded to ConversationsView

    def log_agent_token(self, token: str) -> None:
        pass # Will be forwarded to ConversationsView

    def update_stt_latency(self, ms: float) -> None:
        pass # Will be forwarded to ConversationsView

    def update_ttft_latency(self, ms: float) -> None:
        pass # Will be forwarded to ConversationsView

    def update_tts_latency(self, ms: float) -> None:
        pass # Will be forwarded to ConversationsView