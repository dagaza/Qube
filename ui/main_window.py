import psutil
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QProgressBar, QPushButton, QTextEdit,
    QLabel, QFrame, QFileDialog, QCheckBox,
)
from PyQt6.QtCore import QTimer

from ui.settings_dialog import SettingsDialog
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
        self.resize(1000, 600)
        self.setStyleSheet("background-color: #1e1e2e; color: #cdd6f4;")

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

    # ------------------------------------------------------------------ #
    #  Private – Layout construction                                       #
    # ------------------------------------------------------------------ #

    def _setup_ui(self) -> None:
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        main_layout.addLayout(self._build_left_pane(), stretch=1)
        main_layout.addLayout(self._build_right_pane(), stretch=3)

    def _build_left_pane(self) -> QVBoxLayout:
        left_pane = QVBoxLayout()

        # -- Status bar --------------------------------------------------
        status_container = QFrame()
        status_container.setMinimumHeight(60)
        status_container.setStyleSheet(
            "background-color: #313244; border-radius: 5px; border: 1px solid #45475a;"
        )
        status_layout = QVBoxLayout(status_container)
        self.status_label = QLabel("SYSTEM: Initializing...")
        self.status_label.setStyleSheet(
            "font-weight: bold; color: #f9e2af; font-size: 13px; border: none;"
        )
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

        # -- Load TTS model button ---------------------------------------
        self.load_model_btn = QPushButton("📂 Load TTS Model (.onnx)")
        self.load_model_btn.clicked.connect(self._browse_for_model)

        # -- LLM provider selector (auto-discovery) ----------------------
        self.provider_selector = QComboBox()
        self.provider_selector.setStyleSheet(
            "background-color: #313244; padding: 5px; font-weight: bold;"
        )
        self._setup_provider_selector()

        # -- Settings button ---------------------------------------------
        self.settings_btn = QPushButton("⚙️ Open Settings")
        self.settings_btn.setStyleSheet(
            "background-color: #45475a; padding: 8px; font-weight: bold;"
        )
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

        # -- Interrupt button --------------------------------------------
        self.interrupt_btn = QPushButton("🛑 INTERRUPT ASSISTANT")
        self.interrupt_btn.setStyleSheet(
            "background-color: #f38ba8; color: black; font-weight: bold; padding: 10px;"
        )
        self.interrupt_btn.clicked.connect(self._interrupt_pipeline)
        left_pane.addWidget(self.interrupt_btn)

        # -- Hardware telemetry graph ------------------------------------
        self.telemetry_plot = pg.PlotWidget(title="Hardware Telemetry")
        self.telemetry_plot.setBackground('#181825')
        self.telemetry_plot.setYRange(0, 100)
        self.telemetry_plot.addLegend()

        self.ram_curve = self.telemetry_plot.plot(
            pen=pg.mkPen('#89b4fa', width=2), name="RAM %"
        )
        self.ram_data = [0] * 60

        self.cpu_curve = self.telemetry_plot.plot(
            pen=pg.mkPen('#a6e3a1', width=2), name="CPU %"
        )
        self.cpu_data = [0] * 60

        self.gpu_curve = self.telemetry_plot.plot(
            pen=pg.mkPen('#fab387', width=2), name="GPU %"
        )
        self.gpu_data = [0] * 60

        left_pane.addWidget(self.telemetry_plot)

        return left_pane

    def _build_right_pane(self) -> QVBoxLayout:
        right_pane = QVBoxLayout()

        # -- Transcript --------------------------------------------------
        self.transcript_box = QTextEdit()
        self.transcript_box.setReadOnly(True)
        self.transcript_box.setStyleSheet(
            "background-color: #11111b; font-size: 14px; padding: 10px; border-radius: 5px;"
        )
        right_pane.addWidget(self.transcript_box, stretch=4)

        # -- Latency waterfall -------------------------------------------
        latency_container = QFrame()
        latency_container.setStyleSheet(
            "background-color: #181825; border-radius: 5px; padding: 5px;"
        )
        latency_layout = QHBoxLayout(latency_container)

        self.ww_latency_lbl   = QLabel("Wake Word: -- ms")
        self.ttft_latency_lbl = QLabel("Time-to-First-Token: -- ms")
        self.tts_latency_lbl  = QLabel("TTS Gen: -- ms")

        latency_style = "color: #bac2de; font-family: monospace; font-size: 12px;"
        for lbl in [self.ww_latency_lbl, self.ttft_latency_lbl, self.tts_latency_lbl]:
            lbl.setStyleSheet(latency_style)
            latency_layout.addWidget(lbl)

        right_pane.addWidget(latency_container, stretch=1)

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
        print(f"[STATUS UPDATE]: {message}")
        self.status_label.setText(f"SYSTEM: {message}")

        if "RECORDING" in message:
            self.status_label.setStyleSheet(
                "color: #f38ba8; font-weight: bold; font-size: 14px;"
            )
        elif "Listening" in message:
            self.status_label.setStyleSheet("color: #a6e3a1; font-weight: bold;")
        elif "Thinking" in message or "Transcribing" in message:
            self.status_label.setStyleSheet("color: #f9e2af; font-weight: bold;")
        else:
            self.status_label.setStyleSheet("color: #cdd6f4; font-weight: bold;")

    def log_user_message(self, text: str) -> None:
        self.transcript_box.append(
            f'\n<b style="color:#a6e3a1;">[You]:</b> {text}'
        )
        self.is_agent_typing = False

    def log_agent_token(self, token: str) -> None:
        if not self.is_agent_typing:
            self.transcript_box.append('<b style="color:#cba6f7;">[HAL]:</b> ')
            self.is_agent_typing = True
        cursor = self.transcript_box.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(token)
        self.transcript_box.setTextCursor(cursor)

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
            self.rag_indicator.setText("● RAG Active")
            self.rag_indicator.setStyleSheet(
                "color: #a6e3a1; font-weight: bold; padding-left: 5px;"
            )
        else:
            self.rag_indicator.setText("● RAG Inactive")
            self.rag_indicator.setStyleSheet(
                "color: grey; font-weight: bold; padding-left: 5px;"
            )

    # ------------------------------------------------------------------ #
    #  Qt lifecycle                                                        #
    # ------------------------------------------------------------------ #

    def closeEvent(self, event) -> None:
        self._gpu_monitor.cleanup()
        event.accept()