import os
import logging
from pathlib import Path

import qtawesome as qta
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QFrame, QPushButton,
    QLabel, QCheckBox, QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox, QScrollArea, QProgressBar,
    QStyledItemDelegate, QListView, QMenu, QListWidget, QListWidgetItem, QSlider,
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal

from core.audio_utils import get_input_devices, get_output_devices
from core.network import is_port_open
from core.app_settings import (
    get_enable_memory_enrichment,
    set_enable_memory_enrichment,
    get_engine_mode,
    get_internal_model_path,
    expected_gguf_shard_filenames,
    is_secondary_gguf_shard,
    parse_gguf_shard_info,
    resolve_internal_model_path,
    set_internal_model_path,
    get_internal_n_gpu_layers,
    set_internal_n_gpu_layers,
    get_internal_n_threads,
    set_internal_n_threads,
    get_llm_models_dir,
    get_internal_native_chat_format,
    set_internal_native_chat_format,
    get_auto_load_last_model_on_startup,
    set_auto_load_last_model_on_startup,
    get_audio_input_device_index,
    set_audio_input_device_index,
    get_audio_output_device_index,
    set_audio_output_device_index,
)
from core.cpu_threads import max_cpu_threads_for_ui
from core.gpu_layers_cap import max_safe_n_gpu_layers
from ui.components.brand_buttons import (
    apply_brand_primary,
    apply_brand_danger,
)
from ui.components.wakeword_testbed_dialog import WakewordTestbedDialog
from ui.components.toggle import PrestigeToggle
from ui.components.prestige_dialog import PrestigeDialog
from ui.components.selector_button import SelectorButton


logger = logging.getLogger("Qube.UI.Settings")
LOCAL_GGUF_SHARD_PATHS_ROLE = int(Qt.ItemDataRole.UserRole) + 1


class NoScrollSpinBox(QSpinBox):
    def wheelEvent(self, event):
        event.ignore()


class NoScrollDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):
        event.ignore()


class NoScrollComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()


class NoScrollSlider(QSlider):
    def wheelEvent(self, event):
        event.ignore()


class SettingsView(QWidget):
    audio_pin_toggle = pyqtSignal(bool)
    auto_activator_toggle = pyqtSignal(bool) # 🔑 ADD THIS
    auto_load_last_model_changed = pyqtSignal(bool)
    memory_enrichment_changed = pyqtSignal(bool)
    engine_mode_changed = pyqtSignal(str)
    def __init__(self, workers: dict, db_manager):
        super().__init__()
        self.workers = workers
        self.db = db_manager
        
        self.audio_worker = workers.get("audio")
        self.tts_worker = workers.get("tts")
        self.llm_worker = workers.get("llm")

        self._setup_ui()
        self.engine_mode_changed.connect(self._sync_ai_provider_enabled_for_inference)
        self._populate_hardware_selectors()
        os.makedirs(get_llm_models_dir(), exist_ok=True)
        self._sync_models_dir_label()
        self._sync_active_native_model_label()
        self._refresh_local_gguf_list()
        self._wakeword_testbed_dialog = None

    def _setup_ui(self):
        from PyQt6.QtWidgets import QMenu 

        # Resolved once here and reused for every SelectorButton in this view.
        # SettingsView is built before it's parented to MainWindow, so window()
        # may not yet expose _is_dark_theme — each SelectorButton's showEvent
        # re-checks and re-applies the theme once it becomes visible.
        is_dark = getattr(self.window(), "_is_dark_theme", True)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 40, 40, 40)

        # Title
        title = QLabel("System Settings")
        title.setObjectName("ViewTitle")
        title.setProperty("class", "PageTitle")
        main_layout.addWidget(title)

        # Scrollable Area
        scroll = QScrollArea()
        scroll.setObjectName("SettingsScrollArea")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        scroll_content = QWidget()
        scroll_content.setObjectName("SettingsContent")
        content_layout = QVBoxLayout(scroll_content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(30)

        # --- SECTION 1: AUDIO & HARDWARE ---
        content_layout.addWidget(self._build_section_header("fa5s.microchip", "AUDIO & HARDWARE"))
        
        hw_widget = QWidget()
        hw_widget.setObjectName("SettingsFormContainer")
        
        hw_form = QFormLayout(hw_widget)
        hw_form.setSpacing(15)
        hw_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        
        self.mic_selector = SelectorButton("Select Input Device...", is_dark=is_dark)
        self.device_selector = SelectorButton("Select Output Device...", is_dark=is_dark)

        for btn in [self.mic_selector, self.device_selector]:
            btn.setMaximumWidth(350)
            btn.setMenu(QMenu(btn))

        self.timeout_spinner = NoScrollDoubleSpinBox()
        self.timeout_spinner.setFixedWidth(90)
        self.timeout_spinner.setRange(0.5, 5.0)
        self.timeout_spinner.setSingleStep(0.1)
        self.timeout_spinner.setValue(self.audio_worker.silence_timeout if self.audio_worker else 2.0)
        self.timeout_spinner.setSuffix(" sec")
        self.timeout_spinner.setToolTip("The amount of silence (in seconds) the app waits before deciding you have finished speaking. Lower values make the app respond faster, but it might interrupt you if you pause to think.")
        if self.audio_worker:
            self.timeout_spinner.valueChanged.connect(self.audio_worker.set_silence_timeout)

        self.threshold_spinner = NoScrollSpinBox()
        self.threshold_spinner.setFixedWidth(90)
        self.threshold_spinner.setRange(1, 100)
        self.threshold_spinner.setValue(int(self.audio_worker.speech_threshold) if self.audio_worker else 2)
        self.threshold_spinner.setSuffix("%")
        self.threshold_spinner.setToolTip(
            "Controls how loud you must speak to trigger recording. A higher number acts as a "
            "stronger background noise filter, meaning you will need to speak louder to punch through. "
            "If you are in a quiet environment, use the lowest setting."
        )
        if self.audio_worker:
            self.threshold_spinner.valueChanged.connect(self.audio_worker.set_speech_threshold)

        hw_form.addRow("Audio Input", self.mic_selector)
        hw_form.addRow("Audio Output", self.device_selector)
        hw_form.addRow("Silence Cutoff", self.timeout_spinner)
        hw_form.addRow("VAD Threshold", self.threshold_spinner)

        self.pin_audio_cb = QCheckBox("Pin Audio Controls to Toolbar")
        self.pin_audio_cb.setToolTip(
            "When checked, Silence Cutoff and VAD Threshold appear in the right toolbar. "
            "Uncheck to hide them from the toolbar (settings still apply)."
        )
        self.pin_audio_cb.blockSignals(True)
        self.pin_audio_cb.setChecked(True)
        self.pin_audio_cb.blockSignals(False)
        self.pin_audio_cb.toggled.connect(self.audio_pin_toggle.emit)
        hw_form.addRow("", self.pin_audio_cb)

        content_layout.addWidget(hw_widget)
        content_layout.addWidget(self._build_divider())

        # --- SECTION 2: AI MODELS & ROUTING ---
        content_layout.addWidget(self._build_section_header("fa5s.network-wired", "AI MODELS & ROUTING"))
        
        ai_widget = QWidget()
        ai_widget.setObjectName("SettingsFormContainer")
        ai_form = QFormLayout(ai_widget)
        ai_form.setSpacing(15)
        ai_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.wakeword_selector = SelectorButton("Select Wakeword...", is_dark=is_dark)
        self.engine_selector = SelectorButton("Select engine...", is_dark=is_dark)
        self.provider_selector = SelectorButton("Select Provider...", is_dark=is_dark)
        self.voice_selector = SelectorButton("Select Voice...", is_dark=is_dark)

        self.wakeword_selector.setMenu(QMenu(self.wakeword_selector))
        self.wakeword_selector.setFixedWidth(300)

        for btn in [self.engine_selector, self.provider_selector, self.voice_selector]:
            btn.setMaximumWidth(250)
            btn.setMenu(QMenu(btn))
        self.wakeword_selector.setToolTip(
            "Always run Wakeword Testbed after selecting a wakeword. "
            "Both Community and Recommended wakewords can perform differently "
            "depending on your voice, mic setup, room noise, and sensitivity."
            "You can always download your own wakewords and place them in the wakewords folder."
        )

        wakeword_row = QWidget()
        wakeword_row_layout = QHBoxLayout(wakeword_row)
        wakeword_row_layout.setContentsMargins(0, 0, 0, 0)
        wakeword_row_layout.setSpacing(8)
        wakeword_row_layout.addWidget(self.wakeword_selector)
        self.wakeword_info_btn = QPushButton()
        self.wakeword_info_btn.setFixedSize(24, 24)
        self.wakeword_info_btn.setObjectName("WakewordInfoButton")
        self.wakeword_info_btn.setIcon(qta.icon("fa5s.info-circle", color="#64748b"))
        self.wakeword_info_btn.setToolTip(self.wakeword_selector.toolTip())
        self.wakeword_info_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        wakeword_row_layout.addWidget(self.wakeword_info_btn)
        wakeword_row_layout.addStretch()

        ai_form.addRow("Active Wakeword", wakeword_row)
        self.wakeword_test_lab_btn = QPushButton("Open Wakeword Test Lab")
        apply_brand_primary(self.wakeword_test_lab_btn)
        self.wakeword_test_lab_btn.clicked.connect(self._open_wakeword_test_lab)
        ai_form.addRow("", self.wakeword_test_lab_btn)
        ai_form.addRow("AI Engine", self.engine_selector)
        ai_form.addRow("External Provider", self.provider_selector)

        content_layout.addWidget(ai_widget)
        content_layout.addWidget(self._build_divider())

        # --- NATIVE ENGINE & LOCAL GGUF LIBRARY ---
        content_layout.addWidget(
            self._build_section_header("fa5s.layer-group", "NATIVE ENGINE & LOCAL LIBRARY")
        )
        native_widget = QWidget()
        native_widget.setObjectName("SettingsFormContainer")
        native_form = QFormLayout(native_widget)
        native_form.setSpacing(15)
        native_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._gpu_layers_cap = max_safe_n_gpu_layers()
        gpu_layers_row = QWidget()
        gpu_layers_row_layout = QHBoxLayout(gpu_layers_row)
        gpu_layers_row_layout.setContentsMargins(0, 0, 0, 0)
        gpu_layers_row_layout.setSpacing(12)

        self.gpu_layers_slider = NoScrollSlider(Qt.Orientation.Horizontal)
        self.gpu_layers_slider.setMinimum(0)
        self.gpu_layers_slider.setMaximum(self._gpu_layers_cap)
        self.gpu_layers_slider.setSingleStep(1)
        self.gpu_layers_slider.setPageStep(max(1, self._gpu_layers_cap // 10) if self._gpu_layers_cap else 1)
        _gpu_val = get_internal_n_gpu_layers()
        self.gpu_layers_slider.blockSignals(True)
        self.gpu_layers_slider.setValue(_gpu_val)
        self.gpu_layers_slider.blockSignals(False)

        self.gpu_layers_value_lbl = QLabel(str(_gpu_val))
        self.gpu_layers_value_lbl.setMinimumWidth(44)
        self.gpu_layers_value_lbl.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        _gpu_tip = "The number of AI 'brain layers' loaded into your graphics card (GPU). More layers make the AI generate text much faster, but setting this too high may use up all your video memory and cause crashes."
        self.gpu_layers_slider.setToolTip(_gpu_tip)
        self.gpu_layers_value_lbl.setToolTip(_gpu_tip)
        gpu_layers_row.setToolTip(_gpu_tip)

        self.gpu_layers_slider.valueChanged.connect(self._on_gpu_layers_slider_changed)

        gpu_layers_row_layout.addWidget(self.gpu_layers_slider, stretch=1)
        gpu_layers_row_layout.addWidget(self.gpu_layers_value_lbl)

        self._cpu_threads_max = max_cpu_threads_for_ui()
        cpu_threads_row = QWidget()
        cpu_threads_row_layout = QHBoxLayout(cpu_threads_row)
        cpu_threads_row_layout.setContentsMargins(0, 0, 0, 0)
        cpu_threads_row_layout.setSpacing(12)

        self.cpu_threads_slider = NoScrollSlider(Qt.Orientation.Horizontal)
        self.cpu_threads_slider.setMinimum(1)
        self.cpu_threads_slider.setMaximum(self._cpu_threads_max)
        self.cpu_threads_slider.setSingleStep(1)
        self.cpu_threads_slider.setPageStep(max(1, self._cpu_threads_max // 10))
        _cpu_val = get_internal_n_threads()
        self.cpu_threads_slider.blockSignals(True)
        self.cpu_threads_slider.setValue(_cpu_val)
        self.cpu_threads_slider.blockSignals(False)

        self.cpu_threads_value_lbl = QLabel(str(_cpu_val))
        self.cpu_threads_value_lbl.setMinimumWidth(44)
        self.cpu_threads_value_lbl.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        _cpu_tip = "How many processor cores the AI is allowed to use. Setting this close to your computer's total cores speeds up generation, but might slow down other applications running in the background."
        self.cpu_threads_slider.setToolTip(_cpu_tip)
        self.cpu_threads_value_lbl.setToolTip(_cpu_tip)
        cpu_threads_row.setToolTip(_cpu_tip)

        self.cpu_threads_slider.valueChanged.connect(self._on_cpu_threads_slider_changed)

        cpu_threads_row_layout.addWidget(self.cpu_threads_slider, stretch=1)
        cpu_threads_row_layout.addWidget(self.cpu_threads_value_lbl)

        self.native_chat_format_selector = SelectorButton("Select chat template...", is_dark=is_dark)
        self.native_chat_format_selector.setMaximumWidth(350)
        self.native_chat_format_selector.setMenu(QMenu(self.native_chat_format_selector))
        self.native_chat_format_selector.setToolTip("The specific conversational format this AI model was trained on. If the native engine is hallucinating or talking to itself, changing this to match the model's family (e.g., Llama 3, ChatML) usually fixes it.")
        _chat_format_items = [
            ("Auto (GGUF / library default)", "auto"),
            ("GGUF Jinja (tokenizer.chat_template)", "jinja"),
            ("ChatML", "chatml"),
            ("Llama 3 Instruct", "llama-3"),
            ("Mistral / Mixtral Instruct", "mistral"),
            ("Llama 2 Chat", "llama-2"),
        ]
        self._build_prestige_menu(
            self.native_chat_format_selector,
            _chat_format_items,
            self._on_native_chat_format_changed,
        )
        _fmt = get_internal_native_chat_format()
        _fmt_label = next((label for label, data in _chat_format_items if data == _fmt), _chat_format_items[0][0])
        self.native_chat_format_selector.setText(_fmt_label)

        self.models_dir_label = QLabel()
        self.models_dir_label.setWordWrap(True)

        local_row = QHBoxLayout()
        self.local_gguf_list = QListWidget()
        self.local_gguf_list.setMinimumHeight(100)
        self.local_gguf_list.setMaximumHeight(160)
        local_row.addWidget(self.local_gguf_list, stretch=1)
        local_btn_col = QVBoxLayout()
        local_btn_col.setSpacing(8)
        self.use_local_gguf_btn = QPushButton("Use selected")
        apply_brand_primary(self.use_local_gguf_btn)
        self.use_local_gguf_btn.setToolTip("Activate a downloaded .gguf for the native engine")
        self.use_local_gguf_btn.clicked.connect(self._apply_selected_local_gguf)
        local_btn_col.addWidget(self.use_local_gguf_btn, alignment=Qt.AlignmentFlag.AlignTop)
        self.delete_local_gguf_btn = QPushButton("Delete")
        apply_brand_danger(self.delete_local_gguf_btn)
        self.delete_local_gguf_btn.setToolTip("Permanently delete the selected .gguf file from disk")
        self.delete_local_gguf_btn.clicked.connect(self._delete_selected_local_gguf)
        local_btn_col.addWidget(self.delete_local_gguf_btn, alignment=Qt.AlignmentFlag.AlignTop)
        local_row.addLayout(local_btn_col)

        self.active_native_model_lbl = QLabel()

        native_form.addRow("GPU offload layers", gpu_layers_row)
        native_form.addRow("CPU thread pool", cpu_threads_row)
        native_form.addRow("Chat template (internal)", self.native_chat_format_selector)
        native_form.addRow("Model storage", self.models_dir_label)
        native_form.addRow("On this device", local_row)
        native_form.addRow("Active model", self.active_native_model_lbl)

        content_layout.addWidget(native_widget)
        content_layout.addWidget(self._build_divider())

        # --- STARTUP BEHAVIOR ---
        content_layout.addWidget(self._build_section_header("fa5s.power-off", "STARTUP BEHAVIOR"))
        startup_widget = QWidget()
        startup_widget.setObjectName("SettingsFormContainer")
        startup_form = QFormLayout(startup_widget)
        startup_form.setSpacing(15)
        startup_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.auto_load_last_model_cb = QCheckBox("Load last used model on startup")
        self.auto_load_last_model_cb.setToolTip(
            "Automatically loads the last used model at startup. This may significantly increase application startup time depending on the model size and your hardware."
        )
        self.auto_load_last_model_cb.setChecked(get_auto_load_last_model_on_startup())
        self.auto_load_last_model_cb.toggled.connect(set_auto_load_last_model_on_startup)
        self.auto_load_last_model_cb.toggled.connect(self.auto_load_last_model_changed.emit)
        startup_form.addRow("", self.auto_load_last_model_cb)

        content_layout.addWidget(startup_widget)
        content_layout.addWidget(self._build_divider())

        # --- SECTION: MEMORY & PERFORMANCE (Low-end / RAM) ---
        content_layout.addWidget(self._build_section_header("fa5s.memory", "MEMORY & PERFORMANCE"))
        perf_widget = QWidget()
        perf_widget.setObjectName("SettingsFormContainer")
        perf_form = QFormLayout(perf_widget)
        perf_form.setSpacing(15)
        perf_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.memory_enrichment_toggle = PrestigeToggle()
        self.mem_enrichment_label = QLabel("Enable Memory Enrichment (Requires more RAM)")
        self.mem_enrichment_label.setWordWrap(True)
        mem_row = QWidget()
        mem_row_layout = QHBoxLayout(mem_row)
        mem_row_layout.setContentsMargins(0, 0, 0, 0)
        mem_row_layout.addWidget(self.memory_enrichment_toggle, alignment=Qt.AlignmentFlag.AlignLeft)
        mem_row_layout.addWidget(self.mem_enrichment_label, stretch=1)

        self.memory_enrichment_toggle.blockSignals(True)
        self.memory_enrichment_toggle.setChecked(get_enable_memory_enrichment())
        self.memory_enrichment_toggle.blockSignals(False)
        self.memory_enrichment_toggle.toggled.connect(self._on_memory_enrichment_toggled)

        perf_form.addRow("", mem_row)
        content_layout.addWidget(perf_widget)
        content_layout.addWidget(self._build_divider())
        
        # --- 🔑 SECTION 3: NLP RAG TRIGGERS ---
        content_layout.addWidget(self._build_section_header("fa5s.bolt", "NLP RAG TRIGGERS"))
        content_layout.addWidget(self._build_triggers_manager())

        content_layout.addStretch()
        scroll.setWidget(scroll_content)
        
        # Ensure initial styling is applied
        is_dark = getattr(self.window(), '_is_dark_theme', True)
        self._apply_spinbox_style(is_dark)
        main_layout.addWidget(scroll)

    def _apply_settings_menu_button_chevron_state(self, button: QPushButton) -> None:
        """Keep chevrons / selector styling in sync with the button's enabled state.

        Every Settings dropdown is now a ``SelectorButton`` (custom-painted chevron
        + text); it handles disabled rendering internally via ``apply_theme(...)``.
        The legacy ``QtAwesome`` icon branch is kept for any remaining
        ``#SettingsMenuButton``-style buttons outside this view (chevrons don't
        follow QSS and need explicit re-tinting on enable/disable).
        """
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        if isinstance(button, SelectorButton):
            button.apply_theme(is_dark)
            return
        muted = "#3f3f46" if is_dark else "#a1a1aa"
        active = "#64748b"
        color = active if button.isEnabled() else muted
        button.setIcon(qta.icon("fa5s.chevron-down", color=color))

    def _apply_spinbox_style(self, is_dark: bool):
        """Forces borders to be visible on inputs, checkboxes, and the custom trigger elements."""
        border_color = "rgba(255, 255, 255, 0.15)" if is_dark else "#cbd5e1"
        bg_color = "#313244" if is_dark else "#ffffff"
        text_color = "#cdd6f4" if is_dark else "#1e293b"
        check_bg = "#45475a" if is_dark else "#f1f5f9"
        disabled_border = "rgba(255, 255, 255, 0.08)" if is_dark else "#e2e8f0"
        disabled_bg = "#252536" if is_dark else "#f1f5f9"
        disabled_text = "#71717a" if is_dark else "#94a3b8"
        disabled_check = "#3f3f46" if is_dark else "#e2e8f0"

        style = f"""
            QDoubleSpinBox, QSpinBox, QComboBox {{
                background-color: {bg_color};
                color: {text_color};
                border: 1px solid {border_color};
                border-radius: 8px;
                padding: 5px 10px;
            }}
            QDoubleSpinBox:disabled, QSpinBox:disabled, QComboBox:disabled {{
                background-color: {disabled_bg};
                color: {disabled_text};
                border: 1px solid {disabled_border};
            }}
            QCheckBox {{ color: {text_color}; font-size: 13px; }}
            QCheckBox:disabled {{ color: {disabled_text}; }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border: 1px solid {border_color};
                border-radius: 4px;
                background-color: {check_bg};
            }}
            QCheckBox::indicator:disabled {{
                background-color: {disabled_check};
                border: 1px solid {disabled_border};
            }}
            QCheckBox::indicator:checked {{
                background-color: #8b5cf6; 
                image: url(assets/icons/check_mark.png);
            }}
            QCheckBox::indicator:checked:disabled {{
                background-color: #6d28d9;
                border: 1px solid {disabled_border};
                image: url(assets/icons/check_mark.png);
            }}
        """
        self.timeout_spinner.setStyleSheet(style)
        self.threshold_spinner.setStyleSheet(style)
        if hasattr(self, "native_chat_format_selector"):
            self._apply_settings_menu_button_chevron_state(self.native_chat_format_selector)
        if hasattr(self, "gpu_layers_slider"):
            handle = "#8b5cf6" if is_dark else "#7c3aed"
            slider_css = f"""
                QSlider::groove:horizontal {{
                    height: 6px;
                    background: {bg_color};
                    border: 1px solid {border_color};
                    border-radius: 3px;
                }}
                QSlider::handle:horizontal {{
                    background: {handle};
                    border: 1px solid {border_color};
                    width: 16px;
                    margin: -6px 0;
                    border-radius: 8px;
                }}
                QSlider::sub-page:horizontal {{
                    background: {handle};
                    border-radius: 3px;
                }}
                QSlider:disabled {{
                    opacity: 0.5;
                }}
            """
            self.gpu_layers_slider.setStyleSheet(slider_css)
            self.gpu_layers_value_lbl.setStyleSheet(
                f"color: {text_color}; font-size: 13px; min-width: 44px;"
            )
            if hasattr(self, "cpu_threads_slider"):
                self.cpu_threads_slider.setStyleSheet(slider_css)
                self.cpu_threads_value_lbl.setStyleSheet(
                    f"color: {text_color}; font-size: 13px; min-width: 44px;"
                )
        self.pin_audio_cb.setStyleSheet(style)
        self.auto_activator_cb.setStyleSheet(style)
        if hasattr(self, "auto_load_last_model_cb"):
            self.auto_load_last_model_cb.setStyleSheet(style)
        if hasattr(self, 'mem_enrichment_label'):
            self.mem_enrichment_label.setStyleSheet(f"color: {text_color}; font-size: 13px;")
        
        # 🔑 Style the NLP Trigger input & list
        if hasattr(self, 'trigger_input'):
            self.trigger_input.setStyleSheet(f"""
                QLineEdit {{
                    background-color: {bg_color};
                    color: {text_color};
                    border: 1px solid {border_color};
                    border-radius: 8px;
                    padding: 8px 15px;
                    font-size: 13px;
                }}
                QLineEdit:disabled {{
                    background-color: {disabled_bg};
                    color: {disabled_text};
                    border: 1px solid {disabled_border};
                }}
            """)
            
        if hasattr(self, 'trigger_list'):
            self.trigger_list.setStyleSheet(f"""
                QListWidget {{
                    background-color: transparent;
                    border: 1px solid {border_color};
                    border-radius: 8px;
                }}
                QListWidget::item {{
                    border-bottom: 1px solid {border_color};
                }}
            """)

        if hasattr(self, "local_gguf_list"):
            self.local_gguf_list.setStyleSheet(f"""
                QListWidget {{
                    background-color: transparent;
                    border: 1px solid {border_color};
                    border-radius: 8px;
                }}
                QListWidget::item {{
                    border-bottom: 1px solid {border_color};
                }}
            """)
        if hasattr(self, "active_native_model_lbl"):
            self.active_native_model_lbl.setStyleSheet(f"color: {text_color}; font-size: 13px;")

    def _sync_ai_provider_enabled_for_inference(self, mode: str) -> None:
        """LM Studio / Ollama only applies when routing to an external OpenAI-compatible server."""
        if not hasattr(self, "provider_selector"):
            return
        m = str(mode).lower().strip()
        self.provider_selector.setEnabled(m == "external")
        self._apply_settings_menu_button_chevron_state(self.provider_selector)

    def _sync_wakeword_catalog(self, trigger: str = "manual") -> None:
        _ = trigger
        if not self.audio_worker:
            return
        try:
            self.audio_worker.refresh_wakewords(include_remote=False)
            recommended = [
                ("Recommended - " + spec.display_name, spec.display_name)
                for spec in self.audio_worker.wakeword_manager.list_recommended()
            ]
            community = [
                ("Community - " + spec.display_name, spec.display_name)
                for spec in self.audio_worker.wakeword_manager.list_community()
            ]
            wakeword_items = recommended + community
            if wakeword_items:
                self._build_prestige_menu(
                    self.wakeword_selector,
                    wakeword_items,
                    self._on_wakeword_selection_changed,
                )
                active_name = getattr(self.audio_worker, "active_wakeword_name", "") or wakeword_items[0][1]
                matching_label = next((label for label, data in wakeword_items if data == active_name), wakeword_items[0][0])
                self.wakeword_selector.setText(matching_label)
        except Exception as exc:
            logger.exception("Wakeword catalog sync failed: %s", exc)
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            PrestigeDialog(
                self.window(),
                "Wakeword load failed",
                f"{exc}",
                is_dark=is_dark,
            ).exec()

    def _on_wakeword_selector_pressed(self) -> None:
        self._sync_wakeword_catalog(trigger="dropdown")

    def _open_wakeword_test_lab(self) -> None:
        if not self.audio_worker:
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            PrestigeDialog(
                self.window(),
                "Wakeword test unavailable",
                "Audio worker is not available.",
                is_dark=is_dark,
            ).exec()
            return
        if self._wakeword_testbed_dialog is None:
            self._wakeword_testbed_dialog = WakewordTestbedDialog(self.window(), self.audio_worker)
        self._wakeword_testbed_dialog.on_wakeword_selection_changed()
        self._wakeword_testbed_dialog.show()
        self._wakeword_testbed_dialog.raise_()
        self._wakeword_testbed_dialog.activateWindow()

    def _sync_models_dir_label(self) -> None:
        self.models_dir_label.setText(get_llm_models_dir())

    def _sync_active_native_model_label(self) -> None:
        p = get_internal_model_path()
        name = os.path.basename(p) if p else "(none)"
        self.active_native_model_lbl.setText(name)

    def _on_gpu_layers_slider_changed(self, v: int) -> None:
        self.gpu_layers_value_lbl.setText(str(int(v)))
        self._on_native_gpu_layers_changed(int(v))

    def _on_cpu_threads_slider_changed(self, v: int) -> None:
        self.cpu_threads_value_lbl.setText(str(int(v)))
        set_internal_n_threads(int(v))
        llm = self.workers.get("llm")
        if llm and getattr(llm, "engine_mode", "external") == "internal":
            llm.refresh_native_model_from_settings()

    def _on_native_chat_format_changed(self, mode: str) -> None:
        if mode is not None:
            set_internal_native_chat_format(str(mode))
        llm = self.workers.get("llm")
        if llm and getattr(llm, "engine_mode", "external") == "internal":
            llm.refresh_native_model_from_settings()

    def _on_native_gpu_layers_changed(self, v: int) -> None:
        set_internal_n_gpu_layers(int(v))
        llm = self.workers.get("llm")
        if llm and getattr(llm, "engine_mode", "external") == "internal":
            llm.refresh_native_model_from_settings()

    def _refresh_local_gguf_list(self) -> None:
        if not hasattr(self, "local_gguf_list"):
            return
        self.local_gguf_list.clear()
        root = Path(get_llm_models_dir())
        if not root.is_dir():
            return
        for p in sorted(root.glob("*.gguf")):
            if is_secondary_gguf_shard(str(p)):
                continue
            resolved_primary = str(p.resolve())
            shard_paths: list[str] = [resolved_primary]
            display_name = p.name
            shard_info = parse_gguf_shard_info(str(p))
            if shard_info is not None:
                expected = expected_gguf_shard_filenames(str(p))
                found_paths: list[str] = []
                for fname in expected:
                    part = root / fname
                    if part.is_file():
                        found_paths.append(str(part.resolve()))
                if found_paths:
                    shard_paths = found_paths
                total = int(shard_info.get("total", len(shard_paths)))
                bundle_name = f"{Path(str(shard_info['prefix'])).name}.gguf"
                display_name = f"{bundle_name} ({len(shard_paths)}/{total} shards)"
            item = QListWidgetItem(display_name)
            item.setData(Qt.ItemDataRole.UserRole, resolved_primary)
            item.setData(LOCAL_GGUF_SHARD_PATHS_ROLE, shard_paths)
            self.local_gguf_list.addItem(item)

        active = get_internal_model_path()
        if active:
            for i in range(self.local_gguf_list.count()):
                it = self.local_gguf_list.item(i)
                if it.data(Qt.ItemDataRole.UserRole) == active:
                    self.local_gguf_list.setCurrentItem(it)
                    break

    def _apply_selected_local_gguf(self) -> None:
        item = self.local_gguf_list.currentItem()
        if not item:
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            PrestigeDialog(
                self.window(),
                "No model",
                "Select a downloaded .gguf from the list.",
                is_dark=is_dark,
            ).exec()
            return
        path = resolve_internal_model_path(item.data(Qt.ItemDataRole.UserRole))
        if not path or not os.path.isfile(path):
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            PrestigeDialog(
                self.window(),
                "Missing file",
                "That file is not available on disk.",
                is_dark=is_dark,
            ).exec()
            return
        set_internal_model_path(path)
        self._sync_active_native_model_label()
        llm = self.workers.get("llm")
        if llm:
            cv = getattr(self.window(), "conversations_view", None)
            if cv is not None and hasattr(cv, "interrupt_active_response"):
                cv.interrupt_active_response()
            llm.refresh_native_model_from_settings()
        self._refresh_toolbar_native_model_after_model_change()

    def _refresh_toolbar_native_model_after_model_change(self) -> None:
        """Keep the global toolbar Local LLM control in sync with Settings / active path."""
        mw = self.window()
        if mw and hasattr(mw, "refresh_toolbar_native_model_dropdown"):
            mw.refresh_toolbar_native_model_dropdown()

    def _delete_selected_local_gguf(self) -> None:
        item = self.local_gguf_list.currentItem()
        if not item:
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            PrestigeDialog(
                self.window(),
                "No model",
                "Select a .gguf in the list to delete.",
                is_dark=is_dark,
            ).exec()
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path or not os.path.isfile(path):
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            PrestigeDialog(
                self.window(),
                "Missing file",
                "That file is not available on disk.",
                is_dark=is_dark,
            ).exec()
            return
        shard_paths = item.data(LOCAL_GGUF_SHARD_PATHS_ROLE) or [path]
        shard_paths = [str(p) for p in shard_paths if isinstance(p, str) and p]
        if not shard_paths:
            shard_paths = [path]
        primary_name = os.path.basename(path)
        if len(shard_paths) > 1:
            confirm_msg = (
                f'Permanently delete "{primary_name}" and {len(shard_paths) - 1} related shard file(s) '
                "from this device? This cannot be undone."
            )
        else:
            confirm_msg = f'Permanently delete "{primary_name}" from this device? This cannot be undone.'
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        dlg = PrestigeDialog(
            self.window(),
            "Delete model",
            confirm_msg,
            is_dark=is_dark,
        )
        if not dlg.exec():
            return
        deleted_paths: list[str] = []
        failed_paths: list[tuple[str, OSError]] = []
        for shard_path in shard_paths:
            if not os.path.isfile(shard_path):
                continue
            try:
                os.remove(shard_path)
                deleted_paths.append(shard_path)
            except OSError as e:
                failed_paths.append((shard_path, e))
                logger.error("Failed to delete GGUF %s: %s", shard_path, e)
        if failed_paths:
            preview = "\n".join(f"- {os.path.basename(fp)}: {err}" for fp, err in failed_paths[:4])
            more = f"\n- ... and {len(failed_paths) - 4} more errors" if len(failed_paths) > 4 else ""
            PrestigeDialog(
                self.window(),
                "Delete failed",
                "Some files could not be removed:\n\n"
                f"{preview}{more}",
                is_dark=is_dark,
            ).exec()

        active = get_internal_model_path()
        try:
            active_resolved = str(Path(active).resolve()) if active else ""
            deleted_resolved = {str(Path(p).resolve()) for p in deleted_paths}
            was_active = bool(active_resolved and active_resolved in deleted_resolved)
        except OSError:
            was_active = False
        if was_active:
            set_internal_model_path("")
            llm = self.workers.get("llm")
            if llm:
                llm.refresh_native_model_from_settings()

        self._sync_active_native_model_label()
        self._refresh_local_gguf_list()
        self._refresh_toolbar_native_model_after_model_change()

    def refresh_native_local_library(self) -> None:
        """Call when a .gguf is saved elsewhere (e.g. Model Manager download)."""
        self._sync_models_dir_label()
        self._sync_active_native_model_label()
        self._refresh_local_gguf_list()

    # --------------------------------------------------------- #
    #  🔑 NEW RAG TRIGGER MANAGER                              #
    # --------------------------------------------------------- #

    def _build_triggers_manager(self) -> QWidget:
        """Builds the input box and list UI for custom RAG triggers."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(15, 0, 15, 10)
        layout.setSpacing(15)
        
        # Instruction Label
        instruction = QLabel("Add custom phrases that will trigger a semantic search of your Knowledge Base:")
        instruction.setStyleSheet("color: #64748b; font-size: 12px; font-style: italic;")
        layout.addWidget(instruction)

        # 🔑 NEW: Master Checkbox
        self.auto_activator_cb = QCheckBox("Enable NLP Auto-Activator")
        self.auto_activator_cb.setChecked(True)
        self.auto_activator_cb.toggled.connect(self.auto_activator_toggle.emit)
        layout.addWidget(self.auto_activator_cb)
        
        # Input Row
        input_row = QHBoxLayout()
        self.trigger_input = QLineEdit()
        self.trigger_input.setPlaceholderText("e.g. 'search my notes for...'")
        self.trigger_input.returnPressed.connect(self._on_add_trigger)
        
        self.trigger_add_btn = QPushButton()
        self.trigger_add_btn.setFixedSize(36, 36)
        self.trigger_add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # 🔑 FIX 1: Initialize the icon and CSS immediately upon creation
        is_dark = getattr(self.window(), '_is_dark_theme', True)
        icon_color = "#8b5cf6" if is_dark else "#4c4f69"
        btn_bg = "#313244" if is_dark else "#e2e8f0"
        btn_hover = "#45475a" if is_dark else "#cbd5e1"
        
        self.trigger_add_btn.setIcon(qta.icon('fa5s.plus', color=icon_color))
        self.trigger_add_btn.setStyleSheet(f"""
            QPushButton {{ background: {btn_bg}; border: none; border-radius: 8px; }}
            QPushButton:hover {{ background: {btn_hover}; }}
        """)
        
        self.trigger_add_btn.clicked.connect(self._on_add_trigger)
        
        input_row.addWidget(self.trigger_input)
        input_row.addWidget(self.trigger_add_btn)
        layout.addLayout(input_row)
        
        # Display List
        self.trigger_list = QListWidget()
        # 🔑 FIX 2: Force the layout engine to respect a minimum height!
        self.trigger_list.setMinimumHeight(320) 
        
        self.trigger_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.trigger_list.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        layout.addWidget(self.trigger_list)
        
        self._refresh_trigger_list()
        
        return container

    def _refresh_trigger_list(self):
        """Pulls from SQLite and rebuilds the styled list."""
        if not hasattr(self, 'trigger_list'): return
        
        self.trigger_list.clear()
        triggers = self.db.get_rag_triggers()
        
        is_dark = getattr(self.window(), '_is_dark_theme', True)
        text_color = "#cdd6f4" if is_dark else "#1e293b"
        icon_color = "#ef4444" # Danger Red for Trash
        hover_bg = "rgba(239, 68, 68, 0.1)" # Faint red hover
        
        for phrase in triggers:
            item = QListWidgetItem()
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(15, 5, 10, 5)
            
            lbl = QLabel(phrase)
            lbl.setStyleSheet(f"color: {text_color}; font-size: 13px; font-weight: bold;")
            
            del_btn = QPushButton()
            del_btn.setIcon(qta.icon('fa5s.trash-alt', color=icon_color))
            del_btn.setFixedSize(28, 28)
            del_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            del_btn.setStyleSheet(f"""
                QPushButton {{ background: transparent; border: none; border-radius: 4px; }}
                QPushButton:hover {{ background-color: {hover_bg}; }}
            """)
            del_btn.clicked.connect(lambda checked, p=phrase: self._on_delete_trigger(p))
            
            layout.addWidget(lbl)
            layout.addStretch()
            layout.addWidget(del_btn)
            
            item.setSizeHint(QSize(0, 60))
            self.trigger_list.addItem(item)
            self.trigger_list.setItemWidget(item, row)

    def _on_add_trigger(self):
        text = self.trigger_input.text().strip()
        if text:
            # Add to database
            success = self.db.add_rag_trigger(text)
            if success:
                self.trigger_input.clear()
                self._refresh_trigger_list()
                
    def _on_delete_trigger(self, phrase):
        # Remove from database
        self.db.remove_rag_trigger(phrase)
        self._refresh_trigger_list()

    def _on_memory_enrichment_toggled(self, checked: bool):
        set_enable_memory_enrichment(checked)
        self.memory_enrichment_changed.emit(checked)

    # --------------------------------------------------------- #
    #  THE PRESTIGE MENU LOGIC                                  #
    # --------------------------------------------------------- #
    def _build_prestige_menu(self, button, items, callback):
        from PyQt6.QtWidgets import QMenu, QWidgetAction, QListWidget
        from PyQt6.QtCore import Qt

        menu = QMenu(button)
        menu.setObjectName("PrestigeMenu")
        menu.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        
        is_dark = getattr(self.window(), '_is_dark_theme', True)
        self._apply_menu_theme(menu, is_dark)

        list_widget = QListWidget()
        list_widget.setObjectName("PrestigeMenuList")
        list_widget.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        for label, data in items:
            list_widget.addItem(label)
            
        required_height = len(items) * 32 + 10 
        main_win = self.window()
        max_height = int(main_win.height() * 0.5) if main_win else 400
        list_widget.setFixedHeight(min(required_height, max_height))

        def sync_dropdown_width():
            content_w = list_widget.sizeHintForColumn(0) + 40
            list_widget.setFixedWidth(max(button.width() - 8, content_w, 220))

        menu.aboutToShow.connect(sync_dropdown_width)

        def on_item_clicked(item):
            selected_label = item.text()
            matched_data = next((d for l, d in items if l == selected_label), selected_label)
            self._handle_selection(button, selected_label, matched_data, callback)
            menu.hide()

        list_widget.itemClicked.connect(on_item_clicked)

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
            QMenu {{ background-color: {bg.name()}; border: 1px solid {border}; border-radius: 6px; padding: 4px; }}
            QListWidget#PrestigeMenuList {{ background-color: transparent; border: none; outline: none; }}
            QListWidget#PrestigeMenuList::item {{ background-color: transparent; color: {fg.name()}; padding: 8px 25px; border-radius: 4px; min-height: 24px; }}
            QListWidget#PrestigeMenuList::item:selected, QListWidget#PrestigeMenuList::item:hover {{ background-color: {hover}; color: {sel_fg.name()}; }}
            QScrollBar:vertical {{ border: none; background: transparent; width: 6px; margin: 0px; }}
            QScrollBar::handle:vertical {{ background: {border}; border-radius: 3px; min-height: 20px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
        """)

    def refresh_menu_themes(self, is_dark: bool):
        """Standardizes icons and borders when the theme is toggled."""
        buttons = [
            self.mic_selector,
            self.device_selector,
            self.wakeword_selector,
            self.engine_selector,
            self.provider_selector,
            self.voice_selector,
            self.native_chat_format_selector,
        ]
        for btn in buttons:
            if isinstance(btn, SelectorButton):
                btn.apply_theme(is_dark)
            if btn.menu():
                self._apply_menu_theme(btn.menu(), is_dark)

        # Update Section Header Icons
        icon_color = "#8b5cf6" if is_dark else "#4c4f69" 
        
        for icon_lbl in [
            getattr(self, 'audio_icon_label', None),
            getattr(self, 'ai_icon_label', None),
            getattr(self, 'native_lib_icon_label', None),
            getattr(self, 'perf_icon_label', None),
            getattr(self, 'rag_icon_label', None),
        ]:
            if icon_lbl:
                name = icon_lbl.property("icon_name")
                icon_lbl.setPixmap(qta.icon(name, color=icon_color).pixmap(QSize(18, 18)))

        # Update Trigger Add Button
        if hasattr(self, 'trigger_add_btn'):
            btn_bg = "#313244" if is_dark else "#e2e8f0"
            btn_hover = "#45475a" if is_dark else "#cbd5e1"
            self.trigger_add_btn.setIcon(qta.icon('fa5s.plus', color=icon_color))
            self.trigger_add_btn.setStyleSheet(f"""
                QPushButton {{ background: {btn_bg}; border: none; border-radius: 8px; }}
                QPushButton:hover {{ background: {btn_hover}; }}
            """)

        self._apply_spinbox_style(is_dark)
        self._refresh_trigger_list() # Repaints the list fonts & trash icons!
        self._sync_ai_provider_enabled_for_inference(get_engine_mode())

        if self._wakeword_testbed_dialog is not None:
            self._wakeword_testbed_dialog.refresh_theme(is_dark)

    def _handle_selection(self, button, label, data, callback):
        button.setText(label)
        callback(data)

    def _on_wakeword_selection_changed(self, display_name: str) -> None:
        if not self.audio_worker:
            return
        self._wakeword_selected_label = str(display_name)
        self.audio_worker.set_wakeword(display_name)
        if self._wakeword_testbed_dialog is not None:
            self._wakeword_testbed_dialog.on_wakeword_selection_changed()

    def _build_section_header(self, icon_name, title_text):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        icon_label = QLabel()
        icon_label.setProperty("icon_name", icon_name)
        
        is_dark = getattr(self.window(), '_is_dark_theme', True)
        icon_color = "#8b5cf6" if is_dark else "#4c4f69"
        icon_label.setPixmap(qta.icon(icon_name, color=icon_color).pixmap(QSize(18, 18)))
        icon_label.setProperty("class", "SectionHeaderIcon")
        
        if "AUDIO" in title_text:
            self.audio_icon_label = icon_label
        elif "MODELS" in title_text and "ROUTING" in title_text:
            self.ai_icon_label = icon_label
        elif "NATIVE ENGINE" in title_text:
            self.native_lib_icon_label = icon_label
        elif "MEMORY" in title_text:
            self.perf_icon_label = icon_label
        elif "TRIGGERS" in title_text:
            self.rag_icon_label = icon_label
        
        text_label = QLabel(title_text)
        text_label.setProperty("class", "SectionHeaderLabel")
        
        layout.addWidget(icon_label)
        layout.addWidget(text_label)
        layout.addStretch()
        return container

    def _build_divider(self):
        line = QFrame()
        line.setObjectName("SettingsDivider")
        line.setFrameShape(QFrame.Shape.HLine)
        return line

    def _populate_hardware_selectors(self):
        mics = get_input_devices() 
        if mics:
            self._build_prestige_menu(
                self.mic_selector,
                [(name, idx) for idx, name in mics],
                self._on_input_device_selected,
            )
            saved_input_idx = get_audio_input_device_index()
            if saved_input_idx is not None and self.audio_worker:
                self.audio_worker.set_input_device(saved_input_idx)
            active_mic_name = mics[0][1] 
            active_input_idx = saved_input_idx
            if active_input_idx is None and self.audio_worker and hasattr(self.audio_worker, 'input_device_index'):
                active_input_idx = self.audio_worker.input_device_index
            if active_input_idx is not None:
                for idx, name in mics:
                    if idx == active_input_idx:
                        active_mic_name = name; break
            self.mic_selector.setText(active_mic_name)

        outputs = get_output_devices()
        if outputs:
            self._build_prestige_menu(
                self.device_selector,
                [(name, idx) for idx, name in outputs],
                self._on_output_device_selected,
            )
            saved_output_idx = get_audio_output_device_index()
            if saved_output_idx is not None and self.tts_worker:
                self.tts_worker.set_device(saved_output_idx)
            active_output_name = outputs[0][1]
            active_output_idx = saved_output_idx
            if active_output_idx is None and self.tts_worker and hasattr(self.tts_worker, 'current_device_index'):
                active_output_idx = self.tts_worker.current_device_index
            if active_output_idx is not None:
                for idx, name in outputs:
                    if idx == active_output_idx:
                        active_output_name = name; break
            self.device_selector.setText(active_output_name)

        if self.audio_worker:
            self.wakeword_selector.pressed.connect(self._on_wakeword_selector_pressed)
            self._sync_wakeword_catalog(trigger="settings load")

        engine_modes = [
            ("External Server (localhost)", "external"),
            ("Internal Engine (native)", "internal"),
        ]
        self._build_prestige_menu(
            self.engine_selector,
            engine_modes,
            lambda mode: self.engine_mode_changed.emit(str(mode)),
        )
        em = get_engine_mode()
        engine_label = next((lbl for lbl, m in engine_modes if m == em), engine_modes[0][0])
        self.engine_selector.setText(engine_label)

        providers = [("Ollama (Port 11434)", 11434), ("LM Studio (Port 1234)", 1234)]
        self._build_prestige_menu(self.provider_selector, providers, lambda port: self.llm_worker.set_provider(port) if self.llm_worker else None)
        
        if is_port_open(1234): self.provider_selector.setText("LM Studio (Port 1234)")
        elif is_port_open(11434): self.provider_selector.setText("Ollama (Port 11434)")

        self._sync_ai_provider_enabled_for_inference(get_engine_mode())

    def _on_input_device_selected(self, idx: int) -> None:
        set_audio_input_device_index(idx)
        if self.audio_worker:
            self.audio_worker.set_input_device(idx)

    def _on_output_device_selected(self, idx: int) -> None:
        set_audio_output_device_index(idx)
        if self.tts_worker:
            self.tts_worker.set_device(idx)

    def update_voice_dropdown(self, model_name: str, voices: list) -> None:
        if not voices: return
        self._build_prestige_menu(self.voice_selector, [(v, v) for v in voices], lambda v: self.tts_worker.set_voice(v) if self.tts_worker else None)
        self.voice_selector.setText(voices[0])
        if self.tts_worker: self.tts_worker.set_voice(voices[0])