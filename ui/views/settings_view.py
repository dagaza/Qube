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
    set_internal_model_path,
    get_internal_n_gpu_layers,
    set_internal_n_gpu_layers,
    get_internal_n_threads,
    set_internal_n_threads,
    get_llm_models_dir,
)
from core.cpu_threads import max_cpu_threads_for_ui
from core.gpu_layers_cap import max_safe_n_gpu_layers
from ui.components.toggle import PrestigeToggle
from ui.components.prestige_dialog import PrestigeDialog


logger = logging.getLogger("Qube.UI.Settings")


class SettingsView(QWidget):
    audio_pin_toggle = pyqtSignal(bool)
    auto_activator_toggle = pyqtSignal(bool) # 🔑 ADD THIS
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

    def _setup_ui(self):
        from PyQt6.QtWidgets import QMenu 
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 40, 40, 40)

        # Title
        title = QLabel("SYSTEM SETTINGS")
        title.setObjectName("ViewTitle")
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
        
        self.mic_selector = QPushButton("Select Input Device...")
        self.device_selector = QPushButton("Select Output Device...")
        
        for btn in [self.mic_selector, self.device_selector]:
            btn.setObjectName("SettingsMenuButton")
            btn.setMaximumWidth(350)
            btn.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
            btn.setIcon(qta.icon('fa5s.chevron-down', color='#64748b'))
            btn.setMenu(QMenu(btn))

        self.timeout_spinner = QDoubleSpinBox()
        self.timeout_spinner.setFixedWidth(90)
        self.timeout_spinner.setRange(0.5, 5.0)
        self.timeout_spinner.setSingleStep(0.1)
        self.timeout_spinner.setValue(self.audio_worker.silence_timeout if self.audio_worker else 2.0)
        self.timeout_spinner.setSuffix(" sec")
        if self.audio_worker:
            self.timeout_spinner.valueChanged.connect(self.audio_worker.set_silence_timeout)

        self.threshold_spinner = QSpinBox()
        self.threshold_spinner.setFixedWidth(90)
        self.threshold_spinner.setRange(1, 100)
        self.threshold_spinner.setValue(int(self.audio_worker.speech_threshold) if self.audio_worker else 2)
        self.threshold_spinner.setSuffix("%")
        if self.audio_worker:
            self.threshold_spinner.valueChanged.connect(self.audio_worker.set_speech_threshold)

        hw_form.addRow("Audio Input", self.mic_selector)
        hw_form.addRow("Audio Output", self.device_selector)
        hw_form.addRow("Silence Cutoff", self.timeout_spinner)
        hw_form.addRow("Mic Sensitivity", self.threshold_spinner)

        self.pin_audio_cb = QCheckBox("Pin Audio Controls to Toolbar")
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

        self.wakeword_selector = QPushButton("Select Wakeword...")
        self.engine_selector = QPushButton("Select engine...")
        self.provider_selector = QPushButton("Select Provider...")
        self.voice_selector = QPushButton("Select Voice...")

        for btn in [self.wakeword_selector, self.engine_selector, self.provider_selector, self.voice_selector]:
            btn.setObjectName("SettingsMenuButton")
            btn.setMaximumWidth(250)
            btn.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
            btn.setIcon(qta.icon('fa5s.chevron-down', color='#64748b'))
            btn.setMenu(QMenu(btn))

        ai_form.addRow("Active Wakeword", self.wakeword_selector)
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

        self.gpu_layers_slider = QSlider(Qt.Orientation.Horizontal)
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
        _gpu_tip = (
            "How many transformer layers of the built-in llama.cpp model run on the GPU. "
            "Higher values can be faster but use more video memory; 0 runs the model on the CPU only. "
            f"The maximum ({self._gpu_layers_cap} layers) is capped from your detected GPU memory. "
            "On first launch, the default is about 75% of that cap to leave headroom for the system."
        )
        self.gpu_layers_slider.setToolTip(_gpu_tip)
        self.gpu_layers_value_lbl.setToolTip(_gpu_tip)

        self.gpu_layers_slider.valueChanged.connect(self._on_gpu_layers_slider_changed)

        gpu_layers_row_layout.addWidget(self.gpu_layers_slider, stretch=1)
        gpu_layers_row_layout.addWidget(self.gpu_layers_value_lbl)

        self._cpu_threads_max = max_cpu_threads_for_ui()
        cpu_threads_row = QWidget()
        cpu_threads_row_layout = QHBoxLayout(cpu_threads_row)
        cpu_threads_row_layout.setContentsMargins(0, 0, 0, 0)
        cpu_threads_row_layout.setSpacing(12)

        self.cpu_threads_slider = QSlider(Qt.Orientation.Horizontal)
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
        _cpu_tip = (
            "Number of CPU threads llama.cpp uses for the internal engine (matrix ops, batch decode). "
            "Higher can speed up CPU-bound work; lower leaves more cores free for the OS and other apps. "
            f"Range 1–{self._cpu_threads_max} (logical cores). Defaults reserve ~25% of hardware for background tasks."
        )
        self.cpu_threads_slider.setToolTip(_cpu_tip)
        self.cpu_threads_value_lbl.setToolTip(_cpu_tip)

        self.cpu_threads_slider.valueChanged.connect(self._on_cpu_threads_slider_changed)

        cpu_threads_row_layout.addWidget(self.cpu_threads_slider, stretch=1)
        cpu_threads_row_layout.addWidget(self.cpu_threads_value_lbl)

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
        self.use_local_gguf_btn.setToolTip("Activate a downloaded .gguf for the native engine")
        self.use_local_gguf_btn.clicked.connect(self._apply_selected_local_gguf)
        local_btn_col.addWidget(self.use_local_gguf_btn, alignment=Qt.AlignmentFlag.AlignTop)
        self.delete_local_gguf_btn = QPushButton("Delete")
        self.delete_local_gguf_btn.setToolTip("Permanently delete the selected .gguf file from disk")
        self.delete_local_gguf_btn.clicked.connect(self._delete_selected_local_gguf)
        local_btn_col.addWidget(self.delete_local_gguf_btn, alignment=Qt.AlignmentFlag.AlignTop)
        local_row.addLayout(local_btn_col)

        self.active_native_model_lbl = QLabel()

        native_form.addRow("GPU offload layers", gpu_layers_row)
        native_form.addRow("CPU thread pool", cpu_threads_row)
        native_form.addRow("Model storage", self.models_dir_label)
        native_form.addRow("On this device", local_row)
        native_form.addRow("Active model", self.active_native_model_lbl)

        content_layout.addWidget(native_widget)
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
        """QtAwesome icons do not follow QSS; keep chevrons muted when the button is disabled."""
        muted = "#3f3f46" if getattr(self.window(), "_is_dark_theme", True) else "#a1a1aa"
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
            QDoubleSpinBox, QSpinBox {{
                background-color: {bg_color};
                color: {text_color};
                border: 1px solid {border_color};
                border-radius: 8px;
                padding: 5px 10px;
            }}
            QDoubleSpinBox:disabled, QSpinBox:disabled {{
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
            item = QListWidgetItem(p.name)
            item.setData(Qt.ItemDataRole.UserRole, str(p.resolve()))
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
        set_internal_model_path(path)
        self._sync_active_native_model_label()
        llm = self.workers.get("llm")
        if llm:
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
        name = os.path.basename(path)
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        dlg = PrestigeDialog(
            self.window(),
            "Delete model",
            f'Permanently delete "{name}" from this device? This cannot be undone.',
            is_dark=is_dark,
        )
        if not dlg.exec():
            return
        try:
            os.remove(path)
        except OSError as e:
            logger.error("Failed to delete GGUF %s: %s", path, e)
            PrestigeDialog(
                self.window(),
                "Delete failed",
                f"Could not remove the file: {e}",
                is_dark=is_dark,
            ).exec()
            return

        active = get_internal_model_path()
        try:
            was_active = active and Path(active).resolve() == Path(path).resolve()
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
            list_widget.setFixedWidth(button.width() - 8)

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
        ]
        for btn in buttons:
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

    def _handle_selection(self, button, label, data, callback):
        button.setText(label)
        callback(data)

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
            self._build_prestige_menu(self.mic_selector, [(name, idx) for idx, name in mics], lambda idx: self.audio_worker.set_input_device(idx) if self.audio_worker else None)
            active_mic_name = mics[0][1] 
            if self.audio_worker and hasattr(self.audio_worker, 'input_device_index'):
                for idx, name in mics:
                    if idx == self.audio_worker.input_device_index:
                        active_mic_name = name; break
            self.mic_selector.setText(active_mic_name)

        outputs = get_output_devices()
        if outputs:
            self._build_prestige_menu(self.device_selector, [(name, idx) for idx, name in outputs], lambda idx: self.tts_worker.set_device(idx) if self.tts_worker else None)
            active_output_name = outputs[0][1]
            if self.tts_worker and hasattr(self.tts_worker, 'current_device_index'):
                for idx, name in outputs:
                    if idx == self.tts_worker.current_device_index:
                        active_output_name = name; break
            self.device_selector.setText(active_output_name)

        if self.audio_worker:
            wakewords = list(self.audio_worker.available_wakewords.keys())
            if wakewords:
                self._build_prestige_menu(self.wakeword_selector, [(w, w) for w in wakewords], self.audio_worker.set_wakeword)
                if hasattr(self.audio_worker, 'active_wakeword_name') and self.audio_worker.active_wakeword_name:
                    self.wakeword_selector.setText(self.audio_worker.active_wakeword_name)
                else:
                    self.wakeword_selector.setText(wakewords[0])

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

    def update_voice_dropdown(self, model_name: str, voices: list) -> None:
        if not voices: return
        self._build_prestige_menu(self.voice_selector, [(v, v) for v in voices], lambda v: self.tts_worker.set_voice(v) if self.tts_worker else None)
        self.voice_selector.setText(voices[0])
        if self.tts_worker: self.tts_worker.set_voice(voices[0])