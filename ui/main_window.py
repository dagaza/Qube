import psutil
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QProgressBar, QPushButton, QTextEdit, QApplication,
    QLabel, QFrame, QFileDialog, QCheckBox, QSizeGrip, QMenu, QSystemTrayIcon, QSizePolicy,
)
from PyQt6 import QtCore
from PyQt6.QtCore import Qt, QPoint, QSize, QTimer
from PyQt6.QtGui import QAction, QIcon, QPixmap, QFont

import qtawesome as qta
from .settings_dialog import SettingsDialog
from core.network import is_port_open
from core.audio_utils import get_input_devices, get_output_devices
import logging
logger = logging.getLogger("Qube.UI")

class MainWindow(QMainWindow):
    """
    Pure UI layer for Qube.

    All business/worker logic lives in main.py (Qube).
    This class is responsible for:
      - Building every widget
      - Exposing named widget references so Qube can wire signals
      - Owning the two UI-side timers (telemetry poll, VU-meter animation)
    """

    def __init__(self, workers: dict, gpu_monitor):
        """
        Parameters
        ----------
        workers : dict
            Keys: 'audio', 'stt', 'llm', 'tts'
        gpu_monitor : GPUMonitor
            Abstraction layer for GPU telemetry.
        """
        super().__init__()
        self.setWindowTitle("Qube - Control Center")
        self.resize(1100, 700)

        # 1. THE "FRAMELESS" SECRET SAUCE
        # This removes the OS title bar but keeps the window on top
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # State for manual window dragging
        self._old_pos = None

        # Store references we need for timer callbacks
        self._audio_worker = workers["audio"]
        self._tts_worker   = workers["tts"]
        self._llm_worker   = workers["llm"]
        self._gpu_monitor  = gpu_monitor

        # VU-meter smooth-decay state
        self.current_ui_volume: int = 0

        # Tracks whether the agent is mid-stream so we can prefix once
        self.is_agent_typing: bool = False

        self._setup_ui()
        self._start_timers()
        self._setup_tray()

    # ------------------------------------------------------------------ #
    #  Private – Layout construction                                       #
    # ------------------------------------------------------------------ #

    def _setup_tray(self) -> None:
        """Initializes the system tray icon and its context menu."""
        self.tray_icon = QSystemTrayIcon(self)
        
        # FIX: Use the QIcon object directly. 
        # qta.icon() returns a QIcon, which is what setIcon wants.
        icon = qta.icon('fa5s.cube', color='#89b4fa')
        self.tray_icon.setIcon(icon)

        # Create the menu
        tray_menu = QMenu()
        tray_menu.setStyleSheet("background-color: #1e1e2e; color: #cdd6f4;")

        show_action = QAction("Open Control Center", self)
        show_action.triggered.connect(self.showNormal)
        
        settings_action = QAction("Quick Settings", self)
        settings_action.triggered.connect(self.settings_btn.click)

        quit_action = QAction("Exit Qube", self)
        quit_action.triggered.connect(self._fully_exit)

        tray_menu.addAction(show_action)
        tray_menu.addAction(settings_action)
        tray_menu.addSeparator()
        tray_menu.addAction(quit_action)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()

        # If the user clicks the tray icon directly, show the window
        self.tray_icon.activated.connect(self._on_tray_icon_activated)
    
    def _on_tray_icon_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            if self.isVisible():
                self.hide()
            else:
                self.showNormal()
                self.activateWindow()

    def _fully_exit(self):
        """Ensures background workers stop before the app quits."""
        logger.info("Deep shutdown initiated from Tray.")
        self.tray_icon.hide()
        # Call your app's actual quit logic
        QApplication.quit()
    
    def _setup_ui(self) -> None:
        # Main background container (since the window itself is translucent)
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
        
        layout = QVBoxLayout(self.main_container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 2. THE CUSTOM TITLE BAR
        self.title_bar = self._build_title_bar()
        layout.addWidget(self.title_bar)

        # 3. CONTENT AREA
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(20, 10, 20, 20)
        content_layout.setSpacing(20)
        
        content_layout.addLayout(self._build_left_pane(), stretch=1)
        content_layout.addLayout(self._build_right_pane(), stretch=3)
        
        layout.addLayout(content_layout)

        # 4. RESIZE GRIP (Bottom Right)
        # Without a frame, we need a way to resize the window
        self.grip = QSizeGrip(self)
        layout.addWidget(self.grip, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)

    def _build_title_bar(self) -> QFrame:
        bar = QFrame()
        bar.setFixedHeight(40)
        bar.setStyleSheet("background-color: rgba(30, 30, 46, 0.8); border-top-left-radius: 12px; border-top-right-radius: 12px;")
        
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(15, 0, 10, 0)

        # Icon and Title
        title_icon = QLabel()
        title_icon.setPixmap(qta.icon('fa5s.cube', color='#89b4fa').pixmap(QSize(16, 16)))
        title_label = QLabel("QUBE ASSISTANT")
        title_label.setStyleSheet("font-weight: bold; color: #bac2de; letter-spacing: 1px; font-size: 11px;")
        
        layout.addWidget(title_icon)
        layout.addWidget(title_label)
        layout.addStretch()

        self.rag_status_dot = QLabel("● RAG")
        self.rag_status_dot.setStyleSheet("color: #6c7086; font-weight: bold; font-size: 10px; margin-left: 10px;")
        layout.insertWidget(2, self.rag_status_dot) # Insert it after the title

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

    # --- DRAG LOGIC (Manual implementation for Frameless) ---
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Only allow dragging from the title bar area
            if self.title_bar.underMouse():
                self._old_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if self._old_pos is not None:
            delta = event.globalPosition().toPoint() - self._old_pos
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self._old_pos = event.globalPosition().toPoint()

    def mouseReleaseEvent(self, event):
        self._old_pos = None

    def mouseDoubleClickEvent(self, event):
        """Toggles maximize/restore when the title bar is double-clicked."""
        if event.button() == Qt.MouseButton.LeftButton:
            # Check if the click happened inside the title bar widget
            if self.title_bar.underMouse():
                if self.isMaximized():
                    self.showNormal()
                    # Optional: Re-apply border radius for non-maximized mode
                    self.main_container.setStyleSheet(self.main_container.styleSheet().replace("border-radius: 0px;", "border-radius: 12px;"))
                else:
                    self.showMaximized()
                    # Optional: Remove border radius when maximized to fit screen edges
                    self.main_container.setStyleSheet(self.main_container.styleSheet().replace("border-radius: 12px;", "border-radius: 0px;"))

    def _build_left_pane(self) -> QVBoxLayout:
        left_pane = QVBoxLayout()
        left_pane.setContentsMargins(15, 15, 15, 15)  # Add breathing room
        left_pane.setSpacing(15)                      # Space out the widgets

        # -- Status bar --------------------------------------------------
        status_container = QFrame()
        status_container.setMinimumHeight(60)
        
        status_layout = QVBoxLayout(status_container)
        self.status_label = QLabel("SYSTEM: Initializing...")
        status_layout.addWidget(self.status_label)
        left_pane.addWidget(status_container)

        # -- RAG indicator -----------------------------------------------
        self.rag_indicator = QLabel("● RAG Inactive")
        self.rag_indicator.setStyleSheet(
            "color: grey; font-weight: bold; padding-left: 5px;"
        )
        left_pane.addWidget(self.rag_indicator)

        # -- RAG manual override toggle ----------------------------------
        self.manual_rag_toggle = QCheckBox("🧠 Force Document Search")
        self.manual_rag_toggle.setStyleSheet(
            "color: #cdd6f4; font-weight: bold; padding: 5px;"
        )
        self.manual_rag_toggle.toggled.connect(self._llm_worker.set_dashboard_rag)
        left_pane.addWidget(self.manual_rag_toggle)

        # -- Mic selector (populated but not added to pane directly) -----
        self.mic_selector = QComboBox()
        self._populate_mic_devices()
        self.mic_selector.currentIndexChanged.connect(self._on_mic_changed)

        # -- Audio output selector ---------------------------------------
        self.device_selector = QComboBox()
        self._populate_audio_devices()
        self.device_selector.currentIndexChanged.connect(self._on_audio_device_changed)

        # -- TTS voice selector ------------------------------------------
        self.voice_selector = QComboBox()
        self.voice_selector.setStyleSheet("background-color: #313244; padding: 5px;")
        self.voice_selector.currentIndexChanged.connect(self._on_tts_voice_changed)

        # -- Load Model button
        self.load_model_btn = QPushButton(" Load Model")
        self.load_model_btn.setIcon(qta.icon('fa5s.file-upload', color='#cdd6f4'))

        # -- LLM provider selector (auto-discovery) ----------------------
        self.provider_selector = QComboBox()
        self.provider_selector.setStyleSheet(
            "background-color: #313244; padding: 5px; font-weight: bold;"
        )
        self._setup_provider_selector()

       # -- Settings button with a sharp FontAwesome icon
        self.settings_btn = QPushButton(" Settings")
        self.settings_btn.setIcon(qta.icon('fa5s.cog', color='#cdd6f4'))
        self.settings_btn.setIconSize(QtCore.QSize(18, 18))
        
        left_pane.addWidget(self.settings_btn)

        # -- VU meter ----------------------------------------------------
        self.vu_meter = QProgressBar()
        self.vu_meter.setTextVisible(False)
        self.vu_meter.setRange(0, 100)
        self.vu_meter.setFixedHeight(8)
        self.vu_meter.setStyleSheet("""
            QProgressBar { background-color: #181825; border-radius: 4px; }
            QProgressBar::chunk { background-color: #a6e3a1; border-radius: 3px; }
        """)
        left_pane.addWidget(self.vu_meter)

        # -- Interrupt button with a solid warning icon
        self.interrupt_btn = QPushButton(" INTERRUPT")
        self.interrupt_btn.setIcon(qta.icon('fa5s.stop-circle', color='#11111b'))
        self.interrupt_btn.setIconSize(QtCore.QSize(18, 18))

        # -- Hardware telemetry graph (MODERNIZED with SCALE) --
        self.telemetry_plot = pg.PlotWidget()
        self.telemetry_plot.setBackground(None) # Transparent to inherit the theme
        self.telemetry_plot.setYRange(0, 100, padding=0.05)
        
        # Configure Left Axis for minimalist scale
        left_axis = self.telemetry_plot.getAxis('left')
        left_axis.setPen('#45475a')       # Dark, subtle line color
        left_axis.setTextPen('#6c7086')   # Subdued gray for the labels
        left_axis.setWidth(35)            # Keeps the layout stable when values change
        
        # Custom Ticks: Only show 0, 50, and 100% to provide scale without clutter
        tick_values = [(0, '0'), (50, '50'), (100, '100%')]
        left_axis.setTicks([tick_values])

        # Legend styling
        self.telemetry_plot.addLegend(offset=(10, 10), labelTextColor='#6c7086')
        
        # Hide the bottom axis and disable interactions to keep it a static dashboard
        self.telemetry_plot.hideAxis('bottom')
        self.telemetry_plot.showGrid(x=False, y=True, alpha=0.05) # Very faint grid lines
        self.telemetry_plot.setMouseEnabled(x=False, y=False)
        self.telemetry_plot.setMenuEnabled(False)

        # RAM Curve with soft blue fill
        self.ram_curve = self.telemetry_plot.plot(
            pen=pg.mkPen('#89b4fa', width=2),
            fillLevel=0, 
            brush=(137, 180, 250, 30), # Lowered opacity (30) for better layering
            name="RAM %"
        )
        self.ram_data = [0] * 60

        # CPU Curve with soft green fill
        self.cpu_curve = self.telemetry_plot.plot(
            pen=pg.mkPen('#a6e3a1', width=2), 
            fillLevel=0, 
            brush=(166, 227, 161, 30),
            name="CPU %"
        )
        self.cpu_data = [0] * 60

        # GPU Curve with soft peach fill
        self.gpu_curve = self.telemetry_plot.plot(
            pen=pg.mkPen('#fab387', width=2), 
            fillLevel=0, 
            brush=(250, 179, 135, 30),
            name="GPU %"
        )
        self.gpu_data = [0] * 60

        left_pane.addWidget(self.telemetry_plot)

        return left_pane

    def _build_right_pane(self) -> QVBoxLayout:
        right_pane = QVBoxLayout()
        right_pane.setSpacing(10)
        right_pane.setContentsMargins(0, 0, 0, 0)

        # 1. THE TRANSCRIPT BOX (The "Greedy" Widget)
        self.transcript_box = QTextEdit()
        self.transcript_box.setReadOnly(True)
        # This is the magic line: It tells the box to expand as much as possible
        self.transcript_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        self.transcript_box.setStyleSheet("""
            QTextEdit {
                background-color: #11111b; 
                border: 1px solid #313244;
                border-radius: 10px;
                padding: 15px;
                color: #cdd6f4;
                font-size: 14px;
            }
        """)
        right_pane.addWidget(self.transcript_box)

        # 2. THE LATENCY WATERFALL (The "Footer")
        # We wrap these in a small layout or frame to keep them at the bottom
        latency_container = QVBoxLayout()
        latency_container.setSpacing(2)
        
        self.ww_latency_lbl = QLabel("WakeWord: 0ms")
        self.stt_latency_lbl = QLabel("STT Processing: 0ms")
        self.ttft_latency_lbl = QLabel("Time-to-First-Token: 0ms")
        self.tts_latency_lbl = QLabel("TTS Gen: 0ms")

        # Style them globally to be subtle and small
        latency_style = "color: #6c7086; font-size: 11px; font-family: 'Consolas', monospace;"
        for lbl in [self.ww_latency_lbl, self.stt_latency_lbl, self.ttft_latency_lbl, self.tts_latency_lbl]:
            lbl.setStyleSheet(latency_style)
            latency_container.addWidget(lbl)

        right_pane.addLayout(latency_container)

        return right_pane

    # ------------------------------------------------------------------ #
    #  Private – Widget initialisation helpers                            #
    # ------------------------------------------------------------------ #

    def _setup_provider_selector(self) -> None:
        self.provider_selector.addItem("Ollama (Port 11434)", 11434)
        self.provider_selector.addItem("LM Studio (Port 1234)", 1234)

        # FIX 1: Connect the signal BEFORE setting the index. 
        # This ensures that when we auto-select an item below, the llm_worker is actually notified.
        self.provider_selector.currentIndexChanged.connect(self._on_provider_changed)

        ollama_active   = is_port_open(11434)
        lmstudio_active = is_port_open(1234)

        # FIX 2: Do not disable (gray out) the items. If the port scanner misses 
        # LM Studio because it's slow to boot, we still want to let the user manually select it!
        if lmstudio_active:
            self.provider_selector.setCurrentIndex(1)
        elif ollama_active:
            self.provider_selector.setCurrentIndex(0)
        else:
            self.update_status("WARNING: No LLM detected. Start Ollama or LM Studio.")
            # Default to LM Studio (index 1) without locking the UI
            self.provider_selector.setCurrentIndex(1)

    def _populate_mic_devices(self) -> None:
        self.mic_map: dict[int, int] = {}
        for dropdown_idx, (real_idx, name) in enumerate(get_input_devices()):
            self.mic_selector.addItem(name)
            self.mic_map[dropdown_idx] = real_idx

    def _populate_audio_devices(self) -> None:
        self.device_map: dict[int, int] = {}
        for dropdown_idx, (real_idx, name) in enumerate(get_output_devices()):
            self.device_selector.addItem(name)
            self.device_map[dropdown_idx] = real_idx

    # ------------------------------------------------------------------ #
    #  Private – Timer setup                                              #
    # ------------------------------------------------------------------ #

    def _start_timers(self) -> None:
        self.telemetry_timer = QTimer()
        self.telemetry_timer.timeout.connect(self._update_telemetry)
        self.telemetry_timer.start(500)

        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self._update_vu_meter)
        self.animation_timer.start(33)  # ~30 fps

    # ------------------------------------------------------------------ #
    #  Private – Slot / callback implementations                          #
    # ------------------------------------------------------------------ #

    def _on_mic_changed(self, dropdown_index: int) -> None:
        real_device_index = self.mic_map.get(dropdown_index)
        if real_device_index is not None:
            self._audio_worker.set_input_device(real_device_index)

    def _on_audio_device_changed(self, dropdown_index: int) -> None:
        real_device_index = self.device_map.get(dropdown_index)
        if real_device_index is not None:
            device_name = self.device_selector.currentText()
            self.update_status(f"Switching to {device_name}...")
            self._tts_worker.set_device(real_device_index)

    def _on_provider_changed(self, index: int) -> None:
        selected_port = self.provider_selector.itemData(index)
        if selected_port:
            self._llm_worker.set_provider(selected_port)

    def _on_tts_voice_changed(self, index: int) -> None:
        if index >= 0:
            voice_name = self.voice_selector.itemText(index)
            self._tts_worker.set_voice(voice_name)

    def _browse_for_model(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Piper TTS Model", "", "ONNX Models (*.onnx)"
        )
        if file_path:
            self._tts_worker.load_voice(file_path)

    def _open_settings(self) -> None:
        self.settings_dialog.exec()

    def _interrupt_pipeline(self) -> None:
        logger.warning("USER INITIATED INTERRUPT. Halting workers and clearing queues.")
        self.update_status("INTERRUPTED. Clearing queues...")
        while not self._tts_worker.sentence_queue.empty():
            try:
                self._tts_worker.sentence_queue.get_nowait()
            except Exception:
                break
        self.transcript_box.append(
            '\n<i style="color:#f38ba8;">[SYSTEM: Interaction Halted]</i>'
        )

    def _update_telemetry(self) -> None:
        ram_percent = psutil.virtual_memory().percent
        self.ram_data.pop(0)
        self.ram_data.append(ram_percent)
        self.ram_curve.setData(self.ram_data)

        cpu_percent = psutil.cpu_percent()
        self.cpu_data.pop(0)
        self.cpu_data.append(cpu_percent)
        self.cpu_curve.setData(self.cpu_data)

        gpu_percent = self._gpu_monitor.get_load()
        self.gpu_data.pop(0)
        self.gpu_data.append(gpu_percent)
        self.gpu_curve.setData(self.gpu_data)

    def _update_vu_meter(self) -> None:
        target_volume = self._audio_worker.current_volume
        if target_volume > self.current_ui_volume:
            self.current_ui_volume = target_volume
        else:
            self.current_ui_volume -= 5
        self.current_ui_volume = max(0, self.current_ui_volume)
        self.vu_meter.setValue(int(self.current_ui_volume))

    # ------------------------------------------------------------------ #
    #  Public – called by Qube (signal handlers wired externally)         #
    # ------------------------------------------------------------------ #

    def update_status(self, message: str) -> None:
        self.status_label.setText(f" {message.upper()}")
        
        # Dynamic Icons based on status using qtawesome
        if "RECORDING" in message:
            icon = qta.icon('fa5s.microphone', color='#f38ba8', animation=qta.Pulse(self.status_label))
            self.status_label.setStyleSheet("color: #f38ba8; font-weight: bold;")
        elif "Thinking" in message:
            icon = qta.icon('fa5s.brain', color='#f9e2af', animation=qta.Spin(self.status_label))
            self.status_label.setStyleSheet("color: #f9e2af; font-weight: bold;")
        else:
            icon = qta.icon('fa5s.check-circle', color='#a6e3a1')
            self.status_label.setStyleSheet("color: #a6e3a1; font-weight: bold;")
            
        # You can't directly set an icon on a QLabel, so we'll use a clever trick:
        # If you want the icon next to the text, we'll need to wrap it in a QHBoxLayout 
        # or use a QPushButton styled to look like a label.

    def log_user_message(self, text: str) -> None:
        """Renders user input as a distinct, slightly highlighted block."""
        # We use a simple HTML table for a 'pseudo-bubble' look
        user_html = f"""
            <div style="margin-bottom: 20px;">
                <table width="100%" cellpadding="8" cellspacing="0">
                    <tr>
                        <td bgcolor="#313244" style="border-radius: 10px; color: #a6e3a1; font-family: 'Segoe UI';">
                            <span style="font-size: 10px; color: #9399b2; font-weight: bold;">USER</span><br/>
                            <span style="color: #cdd6f4; font-size: 14px;">{text}</span>
                        </td>
                    </tr>
                </table>
            </div>
        """
        self.transcript_box.append(user_html)
        self.is_agent_typing = False
        self._scroll_to_bottom()

    def log_agent_token(self, token: str) -> None:
        """Streams agent tokens into a clean typography block."""
        if not self.is_agent_typing:
            agent_header = f"""
                <div style="margin-top: 10px;">
                    <span style="font-size: 10px; color: #cba6f7; font-weight: bold;">QUBE</span><br/>
                </div>
            """
            self.transcript_box.append(agent_header)
            self.is_agent_typing = True
        
        cursor = self.transcript_box.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(token)
        self.transcript_box.setTextCursor(cursor)
        self._scroll_to_bottom()

    def _scroll_to_bottom(self):
        bar = self.transcript_box.verticalScrollBar()
        bar.setValue(bar.maximum())

    def update_stt_latency(self, ms: float) -> None:
        self.ww_latency_lbl.setText(f"STT Processing: {ms:.0f} ms")

    def update_ttft_latency(self, ms: float) -> None:
        self.ttft_latency_lbl.setText(f"Time-to-First-Token: {ms:.0f} ms")

    def update_tts_latency(self, ms: float) -> None:
        self.tts_latency_lbl.setText(f"TTS Gen: {ms:.0f} ms")

    def update_voice_dropdown(self, model_name: str, voices: list) -> None:
        self.voice_selector.blockSignals(True)
        self.voice_selector.clear()
        self.voice_selector.addItems(voices)
        self.voice_selector.blockSignals(False)
        if voices:
            self._tts_worker.set_voice(voices[0])
            self.update_status(f"Loaded {model_name} with {len(voices)} voices.")

    def update_rag_indicator(self, active: bool) -> None:
        if active:
            self.rag_status_dot.setStyleSheet("color: #a6e3a1; font-weight: bold; font-size: 10px; margin-left: 10px;")
        else:
            self.rag_status_dot.setStyleSheet("color: #6c7086; font-weight: bold; font-size: 10px; margin-left: 10px;")

    # ------------------------------------------------------------------ #
    #  Qt lifecycle                                                        #
    # ------------------------------------------------------------------ #

    def closeEvent(self, event):
        """Intercept the OS-level close and redirect to tray."""
        if self.tray_icon.isVisible():
            self.hide()
            event.ignore()  # Keep the app running in background
        else:
            event.accept()