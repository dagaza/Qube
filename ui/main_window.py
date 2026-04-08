import psutil
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QApplication, QLabel, QFrame, 
    QSizeGrip, QMenu, QSystemTrayIcon, QStackedWidget, QSizePolicy,
    QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox
)
from PyQt6.QtCore import Qt, QSize, QTimer
from PyQt6.QtGui import QAction
import qtawesome as qta
from .views.conversations_view import ConversationsView
from .views.settings_view import SettingsView
from .views.library_view import LibraryView
from .views.telemetry_view import TelemetryView
from ui.components.toggle import PrestigeToggle
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
        
        # --- NEW: Injecting the actual Conversations View ---
        self.view_conversations = ConversationsView(self.workers, self.workers.get("db"))

        self.view_library = LibraryView(self.workers, self.workers.get("db"))
        self.main_stage.addWidget(self.view_library) # Make sure this matches the index (1) you set in `_route_view`

        # Pass the workers and the gpu_monitor to the Telemetry View
        self.view_telemetry = TelemetryView(self.workers, self._gpu_monitor)
        

        self.view_settings = SettingsView(self.workers, self.workers.get("db"))
        
        self.main_stage.addWidget(self.view_conversations) # Index 0
        self.main_stage.addWidget(self.view_library)       # Index 1
        self.main_stage.addWidget(self.view_telemetry)     # Index 2
        self.main_stage.addWidget(self.view_settings)      # Index 3

        workspace_layout.addWidget(self.main_stage, stretch=1)
        # --- NEW: GLOBAL RIGHT TOOLBAR ---
        self.global_tools = self._build_tools_pane()
        workspace_layout.addWidget(self.global_tools)

        root_layout.addLayout(workspace_layout)

        # Resize Grip
        self.grip = QSizeGrip(self)
        root_layout.addWidget(self.grip, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)

    def _build_top_bar(self) -> QFrame:
        bar = QFrame()
        bar.setFixedHeight(45)
        bar.setObjectName("TopBar")
        
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(15, 0, 15, 0)

        layout.addStretch(1)

        self.status_bubble = QLabel(" IDLE")
        self.status_bubble.setFixedSize(200, 26)
        self.status_bubble.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_bubble.setObjectName("StatusBubble")
        layout.addWidget(self.status_bubble)

        layout.addStretch(1)

        self.rag_status_dot = QLabel("● RAG")
        self.rag_status_dot.setObjectName("RagStatusDot")
        layout.addWidget(self.rag_status_dot)

        min_btn = QPushButton()
        min_btn.setIcon(qta.icon('fa5s.minus'))
        min_btn.setProperty("class", "WindowControlButton")
        min_btn.clicked.connect(self.showMinimized)

        close_btn = QPushButton()
        close_btn.setIcon(qta.icon('fa5s.times'))
        close_btn.setProperty("class", "WindowControlButton")
        close_btn.clicked.connect(self.hide)

        layout.addWidget(min_btn)
        layout.addWidget(close_btn)
        
        return bar

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

        self.nav_settings = create_nav_btn('fa5s.cog', 3, size=20)
        layout.addWidget(self.nav_settings, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Mini Telemetry Block
        self.mini_telemetry = QLabel("CPU: --\nRAM: --\nGPU: --")
        self.mini_telemetry.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mini_telemetry.setProperty("class", "MiniTelemetryText")
        layout.addWidget(self.mini_telemetry, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.nav_buttons = [self.nav_chat, self.nav_library, self.nav_telemetry, self.nav_settings]
        
        return sidebar
    
    def _build_tools_pane(self) -> QFrame:
        """Global Right Sidebar: Houses LLM generation settings, MCP toggles, and RAG options."""
        frame = QFrame()
        frame.setFixedWidth(260)
        frame.setObjectName("ToolsPane")
        
        # We call this 'main_layout' to avoid any shadowing!
        main_layout = QVBoxLayout(frame)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(25)

        # --- 1. AUDIO INPUT ---
        mic_layout = QVBoxLayout()
        mic_layout.setSpacing(10)
        
        m_title = QLabel("AUDIO INPUT")
        m_title.setProperty("class", "ToolsPaneHeader")
        mic_layout.addWidget(m_title)
        
        # Create a horizontal row for the toggle
        mic_row = QHBoxLayout()
        self.voice_input_toggle = PrestigeToggle()
        self.voice_input_toggle.setChecked(True)
        mic_lbl = QLabel("Enable Voice Input")
        mic_lbl.setProperty("class", "ToolsPaneControl")
        
        mic_row.addWidget(self.voice_input_toggle)
        mic_row.addWidget(mic_lbl)
        mic_row.addStretch()
        mic_layout.addLayout(mic_row)
        
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
        self.global_voice_selector.setObjectName("SettingsMenuButton") # Uses your new CSS!
        self.global_voice_selector.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.global_voice_selector.setIcon(qta.icon('fa5s.chevron-down', color='#64748b'))
        self.global_voice_selector.setMenu(QMenu(self.global_voice_selector))
        voice_layout.addWidget(self.global_voice_selector)
        
        main_layout.addLayout(voice_layout)

        # --- 3. GENERATION PARAMETERS ---
        param_layout = QVBoxLayout()
        param_layout.setSpacing(10)
        
        p_title = QLabel("GENERATION PARAMETERS")
        p_title.setProperty("class", "ToolsPaneHeader")
        param_layout.addWidget(p_title)

        temp_row = QHBoxLayout()
        temp_label = QLabel("Temperature:")
        temp_label.setProperty("class", "ToolsPaneControl")
        temp_row.addWidget(temp_label)
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setValue(0.7)
        self.temp_spin.setProperty("class", "ToolsPaneInput")
        temp_row.addWidget(self.temp_spin)
        param_layout.addLayout(temp_row)

        ctx_row = QHBoxLayout()
        ctx_label = QLabel("Context Limit:")
        ctx_label.setProperty("class", "ToolsPaneControl")
        ctx_row.addWidget(ctx_label)
        self.ctx_spin = QSpinBox()
        self.ctx_spin.setRange(1024, 128000)
        self.ctx_spin.setValue(4096)
        self.ctx_spin.setProperty("class", "ToolsPaneInput")
        ctx_row.addWidget(self.ctx_spin)
        param_layout.addLayout(ctx_row)
        
        main_layout.addLayout(param_layout)

        # --- 4. EXPERIMENTAL RAG ---
        rag_layout = QVBoxLayout()
        rag_layout.setSpacing(12)
        r_title = QLabel("EXPERIMENTAL RAG")
        r_title.setProperty("class", "ToolsPaneHeader")
        rag_layout.addWidget(r_title)

        def create_toggle_row(label_text, checked=False):
            row = QHBoxLayout()
            toggle = PrestigeToggle()
            toggle.setChecked(checked)
            lbl = QLabel(label_text)
            lbl.setProperty("class", "ToolsPaneControl")
            row.addWidget(toggle)
            row.addWidget(lbl)
            row.addStretch()
            return row, toggle

        hybrid_row, self.rag_hybrid_toggle = create_toggle_row("Hybrid Search (Alpha)")
        strict_row, self.rag_strict_toggle = create_toggle_row("Strict Document Context")
        rag_layout.addLayout(hybrid_row)
        rag_layout.addLayout(strict_row)
        main_layout.addLayout(rag_layout)

        # --- 5. MCP TOOLS ---
        tools_layout = QVBoxLayout()
        tools_layout.setSpacing(12)
        t_title = QLabel("MCP TOOLS (AGENTIC)")
        t_title.setProperty("class", "ToolsPaneHeader")
        tools_layout.addWidget(t_title)

        local_row, self.tool_rag_toggle = create_toggle_row("Local Knowledge Base", checked=True)
        web_row, self.tool_internet_toggle = create_toggle_row("Internet Search")
        tools_layout.addLayout(local_row)
        tools_layout.addLayout(web_row)
        main_layout.addLayout(tools_layout)

        main_layout.addStretch()

        # --------------------------------------------------------- #
        #  WIRING TO WORKERS                                        #
        # --------------------------------------------------------- #
        if self._audio_worker:
            self.voice_input_toggle.toggled.connect(lambda checked: self._audio_worker.set_paused(not checked))
            
        if self._tts_worker:
            self.voice_bypass_toggle.toggled.connect(lambda checked: self._tts_worker.set_mute(not checked))

        if self._llm_worker:
            self.temp_spin.valueChanged.connect(self._llm_worker.set_temperature)
            self.ctx_spin.valueChanged.connect(self._llm_worker.set_context_window)
            self.tool_rag_toggle.toggled.connect(self._llm_worker.set_mcp_rag)
            self.tool_internet_toggle.toggled.connect(self._llm_worker.set_mcp_internet)

        return frame

    # --- PRESTIGE MENU LOGIC ---
    def _build_prestige_menu(self, button, items, callback):
        """Builds a palette-forced QMenu with a dynamic, scrollable list."""
        from PyQt6.QtWidgets import QMenu, QWidgetAction, QListWidget
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

        # 2. Populate the List
        for label, data in items:
            list_widget.addItem(label)
            
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
            list_widget.setFixedWidth(button.width() - 8)

        menu.aboutToShow.connect(sync_dropdown_width)

        # 4. Handle Selection
        def on_item_clicked(item):
            selected_label = item.text()
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

        # --- RE-THEME ATTACHED MENUS ---
        # 1. Update the Settings Page menus (Fixed to use 'view_settings'!)
        if hasattr(self, 'view_settings'):
            self.view_settings.refresh_menu_themes(self._is_dark_theme)
            
        # 2. Update the Toolbar Voice Menu
        if hasattr(self, 'global_voice_selector'):
            toolbar_menu = self.global_voice_selector.menu()
            if toolbar_menu:
                self._apply_menu_theme(toolbar_menu, self._is_dark_theme)
        # 3. Update the Conversations Kebab Menus
        if hasattr(self, 'view_conversations'):
            self.view_conversations.refresh_menu_themes(self._is_dark_theme)
        if hasattr(self, 'view_conversations'):
            self.view_conversations._refresh_history_list() # Force icon re-color
        # 4. Update the Library Kebab Menus and Icons
        if hasattr(self, 'view_library'):
            self.view_library.refresh_library_list()
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
        """Updates the top bar status bubble with dynamic CSS states."""
        msg_upper = message.upper().strip()
        self.status_bubble.setText(f" {msg_upper}")
        
        # 1. Assign the state based on keywords
        if "RECORDING" in msg_upper or "LISTENING" in msg_upper:
            state = "recording"
        elif "THINKING" in msg_upper or "GENERATING" in msg_upper:
            state = "thinking"
        else:
            state = "idle"
            
        # 2. Set the property
        self.status_bubble.setProperty("state", state)
        
        # 3. FORCE REPAINT (The 'Unpolish/Polish' trick)
        self.status_bubble.style().unpolish(self.status_bubble)
        self.status_bubble.style().polish(self.status_bubble)

    def update_rag_indicator(self, active: bool) -> None:
        color = "#a6e3a1" if active else "#6c7086"
        self.rag_status_dot.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 11px; margin-right: 15px;")

    def log_user_message(self, text: str) -> None:
        pass # Will be forwarded to ConversationsView

    def log_agent_token(self, token: str) -> None:
        pass # Will be forwarded to ConversationsView

    def update_stt_latency(self, ms: float) -> None:
        self.view_telemetry.update_stt_latency(ms)

    def update_ttft_latency(self, ms: float) -> None:
        self.view_telemetry.update_ttft_latency(ms)

    def update_tts_latency(self, ms: float) -> None:
        self.view_telemetry.update_tts_latency(ms)

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