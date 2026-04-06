import qtawesome as qta
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QFrame, QPushButton,
    QLabel, QCheckBox, QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox, QScrollArea, QProgressBar
)
from PyQt6.QtCore import Qt, QSize
from pathlib import Path
import logging

from core.audio_utils import get_input_devices, get_output_devices
from core.network import is_port_open

logger = logging.getLogger("Qube.UI.Settings")

class SettingsView(QWidget):
    def __init__(self, workers: dict, db_manager):
        super().__init__()
        self.workers = workers
        self.db = db_manager
        
        self.audio_worker = workers.get("audio")
        self.tts_worker = workers.get("tts")
        self.llm_worker = workers.get("llm")

        self._setup_ui()
        self._populate_hardware_selectors()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 40, 40, 40)

        # Title
        title = QLabel("SYSTEM SETTINGS")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #cdd6f4; margin-bottom: 20px;")
        main_layout.addWidget(title)

        # Scrollable Area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent;")
        
        scroll_content = QWidget()
        content_layout = QVBoxLayout(scroll_content)
        content_layout.setContentsMargins(0, 0, 40, 0)
        content_layout.setSpacing(30)

        # Helper to style forms
        form_style = """
            QLabel { color: #bac2de; font-size: 13px; font-weight: bold; }
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
                background-color: #313244; color: #cdd6f4; border-radius: 5px; padding: 6px; border: 1px solid #45475a;
            }
        """

        # --- SECTION 1: AUDIO & HARDWARE ---
        content_layout.addWidget(self._build_section_header("fa5s.microchip", "AUDIO & HARDWARE"))
        hw_form = QFormLayout()
        hw_form.setSpacing(15)
        hw_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        
        self.mic_selector = QComboBox()
        self.mic_selector.currentIndexChanged.connect(self._on_mic_changed)
        
        self.device_selector = QComboBox()
        self.device_selector.currentIndexChanged.connect(self._on_audio_device_changed)
        
        self.timeout_spinner = QDoubleSpinBox()
        self.timeout_spinner.setRange(0.5, 5.0)
        self.timeout_spinner.setSingleStep(0.1)
        self.timeout_spinner.setValue(self.audio_worker.silence_timeout if self.audio_worker else 2.0)
        self.timeout_spinner.setSuffix(" sec")
        if self.audio_worker:
            self.timeout_spinner.valueChanged.connect(self.audio_worker.set_silence_timeout)

        self.threshold_spinner = QSpinBox()
        self.threshold_spinner.setRange(1, 100)
        self.threshold_spinner.setValue(int(self.audio_worker.speech_threshold) if self.audio_worker else 2)
        self.threshold_spinner.setSuffix("%")
        if self.audio_worker:
            self.threshold_spinner.valueChanged.connect(self.audio_worker.set_speech_threshold)

        hw_form.addRow("🎙️ Audio Input", self.mic_selector)
        hw_form.addRow("🔊 Audio Output", self.device_selector)
        hw_form.addRow("⏱️ Silence Cutoff", self.timeout_spinner)
        hw_form.addRow("🎚️ Mic Sensitivity", self.threshold_spinner)
        
        hw_widget = QWidget()
        hw_widget.setStyleSheet(form_style)
        hw_widget.setLayout(hw_form)
        content_layout.addWidget(hw_widget)
        content_layout.addWidget(self._build_divider())

        # --- SECTION 2: AI MODELS & ROUTING ---
        content_layout.addWidget(self._build_section_header("fa5s.network-wired", "AI MODELS & ROUTING"))
        ai_form = QFormLayout()
        ai_form.setSpacing(15)
        ai_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.wakeword_selector = QComboBox()
        if self.audio_worker:
            self.wakeword_selector.addItems(list(self.audio_worker.available_wakewords.keys()))
            self.wakeword_selector.currentTextChanged.connect(self.audio_worker.set_wakeword)

        self.provider_selector = QComboBox()
        self.provider_selector.currentIndexChanged.connect(self._on_provider_changed)
        
        self.voice_selector = QComboBox()
        if self.tts_worker:
            self.voice_selector.currentIndexChanged.connect(lambda idx: self.tts_worker.set_voice(self.voice_selector.currentText()))

        ai_form.addRow("🔔 Active Wakeword", self.wakeword_selector)
        ai_form.addRow("🤖 AI Provider", self.provider_selector)
        ai_form.addRow("🗣️ Active Voice", self.voice_selector)
        
        ai_widget = QWidget()
        ai_widget.setStyleSheet(form_style)
        ai_widget.setLayout(ai_form)
        content_layout.addWidget(ai_widget)

        content_layout.addStretch()
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

    def _build_section_header(self, icon_name, title_text):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        icon_label = QLabel()
        icon_label.setPixmap(qta.icon(icon_name, color="#fab387").pixmap(QSize(18, 18)))
        text_label = QLabel(title_text)
        text_label.setStyleSheet("font-weight: bold; color: #fab387; font-size: 14px; letter-spacing: 1px;")
        layout.addWidget(icon_label)
        layout.addWidget(text_label)
        layout.addStretch()
        return container

    def _build_divider(self):
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("background-color: #313244; margin-top: 10px; margin-bottom: 10px;")
        return line

    # --- HARDWARE POPULATION LOGIC ---
    def _populate_hardware_selectors(self):
        # Mics
        self.mic_map = {}
        for dropdown_idx, (real_idx, name) in enumerate(get_input_devices()):
            self.mic_selector.addItem(name)
            self.mic_map[dropdown_idx] = real_idx

        # Output Devices
        self.device_map = {}
        for dropdown_idx, (real_idx, name) in enumerate(get_output_devices()):
            self.device_selector.addItem(name)
            self.device_map[dropdown_idx] = real_idx

        # Providers
        self.provider_selector.addItem("Ollama (Port 11434)", 11434)
        self.provider_selector.addItem("LM Studio (Port 1234)", 1234)
        if is_port_open(1234):
            self.provider_selector.setCurrentIndex(1)
        elif is_port_open(11434):
            self.provider_selector.setCurrentIndex(0)

    def _on_mic_changed(self, dropdown_index: int):
        real_device_index = self.mic_map.get(dropdown_index)
        if real_device_index is not None and self.audio_worker:
            self.audio_worker.set_input_device(real_device_index)

    def _on_audio_device_changed(self, dropdown_index: int):
        real_device_index = self.device_map.get(dropdown_index)
        if real_device_index is not None and self.tts_worker:
            self.tts_worker.set_device(real_device_index)

    def _on_provider_changed(self, index: int):
        selected_port = self.provider_selector.itemData(index)
        if selected_port and self.llm_worker:
            self.llm_worker.set_provider(selected_port)