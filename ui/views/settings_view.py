import os
import logging
from pathlib import Path

import qtawesome as qta
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QFrame, QPushButton,
    QLabel, QCheckBox, QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox, QScrollArea, QProgressBar,
    QToolButton,
    QStyledItemDelegate, QListView, QMenu, QListWidget, QListWidgetItem, QSlider,
    QButtonGroup, QPlainTextEdit,
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QTimer, QFileSystemWatcher
from PyQt6.QtGui import QShowEvent

from core.audio_utils import get_input_devices, get_output_devices
from core.local_gguf_display import format_local_gguf_display, local_gguf_sort_key
from core.network import is_port_open
from core.settings_store import (
    default_user_settings_path,
    get_settings_store,
)
from core.app_settings import (
    get_enable_memory_enrichment,
    set_enable_memory_enrichment,
    get_enable_memory_promotion,
    set_enable_memory_promotion,
    get_memory_promotion_acknowledged,
    set_memory_promotion_acknowledged,
    get_enable_memory_consolidation,
    set_enable_memory_consolidation,
    get_enable_chat_personality_nudge,
    set_enable_chat_personality_nudge,
    get_memory_promotion_preset,
    set_memory_promotion_preset,
    get_profile_units,
    set_profile_units,
    DEFAULT_ENGINE_MODE,
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
    get_model_manager_hardware_suggestions,
    set_model_manager_hardware_suggestions,
    get_audio_input_device_index,
    set_audio_input_device_index,
    get_audio_output_device_index,
    set_audio_output_device_index,
    get_advanced_engine_unlocked,
    set_advanced_engine_unlocked,
    get_sidecar_model_path,
    set_sidecar_model_path,
    get_sidecar_chat_format,
    set_sidecar_chat_format,
    get_llm_temperature,
    get_llm_context_limit,
    get_llm_chat_history_messages,
    get_llm_top_k,
    get_llm_repeat_penalty,
    get_llm_presence_penalty,
    get_llm_top_p,
    get_llm_min_p,
)
from core.auxiliary_cognition import (
    get_cognition_models_dir,
    is_protected_cognition_model,
    list_selectable_cognition_models,
    resolve_active_cognition_path,
    validate_cognition_model_path,
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
from ui.components.settings_json_editor_dialog import SettingsJsonEditorDialog
from ui.components.selector_button import SelectorButton


logger = logging.getLogger("Qube.UI.Settings")
LOCAL_GGUF_SHARD_PATHS_ROLE = int(Qt.ItemDataRole.UserRole) + 1
COGNITION_ENTRY_DELETABLE_ROLE = int(Qt.ItemDataRole.UserRole) + 2


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
    memory_promotion_changed = pyqtSignal(bool)
    memory_consolidation_changed = pyqtSignal(bool)
    engine_mode_changed = pyqtSignal(str)
    external_settings_reloaded = pyqtSignal(set)
    cognition_model_changed = pyqtSignal()
    def __init__(self, workers: dict, db_manager):
        super().__init__()
        self.workers = workers
        self.db = db_manager
        
        self.audio_worker = workers.get("audio")
        self.tts_worker = workers.get("tts")
        self.llm_worker = workers.get("llm")
        self._template_override_reload_pending = False
        self._auto_reset_reload_pending = False
        self._companion_verbal_test_worker = None

        self._setup_ui()
        self.engine_mode_changed.connect(self._sync_ai_provider_enabled_for_inference)
        self.engine_mode_changed.connect(lambda _mode: self._sync_native_chat_template_label())
        native_engine = self.workers.get("native_engine")
        if native_engine is not None and hasattr(native_engine, "load_finished"):
            native_engine.load_finished.connect(self._on_native_model_load_finished)
        self._populate_hardware_selectors()
        os.makedirs(get_llm_models_dir(), exist_ok=True)
        self._sync_models_dir_label()
        self._sync_active_native_model_label()
        self._sync_native_chat_template_label()
        self._refresh_local_gguf_list()
        self._wakeword_testbed_dialog = None
        self._settings_json_dialog: SettingsJsonEditorDialog | None = None
        self._setup_settings_file_watcher()

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self._sync_active_native_model_label()
        self._sync_native_chat_template_label()
        self._ensure_settings_file_watched()

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
        self.mic_selector.setToolTip(
            "Microphone used for voice input and wakeword detection."
        )
        self.device_selector.setToolTip(
            "Speaker or headset used for text-to-speech playback."
        )

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
        self.engine_selector.setObjectName("SettingsEngineSelector")
        self.provider_selector = SelectorButton("Select Provider...", is_dark=is_dark)
        self.voice_selector = SelectorButton("Select Voice...", is_dark=is_dark)

        self.wakeword_selector.setMenu(QMenu(self.wakeword_selector))
        self.wakeword_selector.setFixedWidth(300)

        for btn in [self.engine_selector, self.provider_selector, self.voice_selector]:
            btn.setMaximumWidth(250)
            btn.setMenu(QMenu(btn))
        self.engine_selector.setToolTip(
            "Internal runs downloaded .gguf models on this device. "
            "External connects to LM Studio or Ollama."
        )
        self.provider_selector.setToolTip(
            "OpenAI-compatible server to use when External inference is selected."
        )
        self.voice_selector.setToolTip("Default text-to-speech voice for spoken responses.")
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
        self.wakeword_test_lab_btn.setToolTip(
            "Test wakeword detection with your microphone before relying on it in conversation."
        )
        self.wakeword_test_lab_btn.clicked.connect(self._open_wakeword_test_lab)
        wakeword_lab_row = QWidget()
        wakeword_lab_layout = QHBoxLayout(wakeword_lab_row)
        wakeword_lab_layout.setContentsMargins(0, 0, 0, 0)
        wakeword_lab_layout.addWidget(self.wakeword_test_lab_btn, 0)
        wakeword_lab_layout.addStretch(1)
        ai_form.addRow("", wakeword_lab_row)
        ai_form.addRow("AI Engine", self.engine_selector)
        ai_form.addRow("External Provider", self.provider_selector)

        self._generation_spinboxes: list = []
        _gen_temp_tip = (
            "Creativity slider: lower values (0.1–0.3) produce strict, factual answers. "
            "Higher values (0.7–1.0) make Qube more creative."
        )
        _gen_ctx_tip = (
            "Memory wall: sets the absolute maximum number of tokens Qube is allowed to "
            "output in a single turn."
        )
        _gen_history_tip = (
            "Short-term memory: how many past messages to send to the AI. Higher values "
            "give the AI better context but consume more system RAM (VRAM). Qube's "
            "long-term memory still remembers important facts even when this is set low."
        )
        _gen_top_k_tip = (
            "Top-K sampling: only the K most likely next tokens are considered. "
            "0 disables top-K filtering."
        )
        _gen_repeat_tip = (
            "Repeat penalty: values above 1.0 discourage the model from repeating recent "
            "words or phrases."
        )
        _gen_presence_tip = (
            "Presence penalty: discourages tokens that have already appeared anywhere in "
            "the current output."
        )
        _gen_top_p_tip = (
            "Top-P (nucleus) sampling: keeps the smallest set of tokens whose cumulative "
            "probability reaches P."
        )
        _gen_min_p_tip = (
            "Min-P sampling: drops tokens below this relative probability floor. "
            "0 disables min-P filtering."
        )

        self.llm_temp_spin = NoScrollDoubleSpinBox()
        self.llm_temp_spin.setRange(0.0, 2.0)
        self.llm_temp_spin.setSingleStep(0.1)
        self.llm_temp_spin.setValue(get_llm_temperature())
        self._add_generation_form_row(ai_form, "Temperature", _gen_temp_tip, self.llm_temp_spin)

        self.llm_ctx_spin = NoScrollSpinBox()
        self.llm_ctx_spin.setRange(1024, 128000)
        self.llm_ctx_spin.setSingleStep(256)
        self.llm_ctx_spin.setValue(get_llm_context_limit())
        self._add_generation_form_row(ai_form, "Context limit", _gen_ctx_tip, self.llm_ctx_spin)

        self.llm_history_spin = NoScrollSpinBox()
        self.llm_history_spin.setRange(2, 100)
        self.llm_history_spin.setSingleStep(2)
        self.llm_history_spin.setValue(get_llm_chat_history_messages())
        self._add_generation_form_row(ai_form, "Chat history", _gen_history_tip, self.llm_history_spin)

        self.llm_top_k_spin = NoScrollSpinBox()
        self.llm_top_k_spin.setRange(0, 200)
        self.llm_top_k_spin.setValue(get_llm_top_k())
        self._add_generation_form_row(ai_form, "Top-K sampling", _gen_top_k_tip, self.llm_top_k_spin)

        self.llm_repeat_penalty_spin = NoScrollDoubleSpinBox()
        self.llm_repeat_penalty_spin.setRange(0.0, 2.0)
        self.llm_repeat_penalty_spin.setSingleStep(0.05)
        self.llm_repeat_penalty_spin.setValue(get_llm_repeat_penalty())
        self._add_generation_form_row(
            ai_form, "Repeat penalty", _gen_repeat_tip, self.llm_repeat_penalty_spin
        )

        self.llm_presence_penalty_spin = NoScrollDoubleSpinBox()
        self.llm_presence_penalty_spin.setRange(0.0, 2.0)
        self.llm_presence_penalty_spin.setSingleStep(0.05)
        self.llm_presence_penalty_spin.setValue(get_llm_presence_penalty())
        self._add_generation_form_row(
            ai_form, "Presence penalty", _gen_presence_tip, self.llm_presence_penalty_spin
        )

        self.llm_top_p_spin = NoScrollDoubleSpinBox()
        self.llm_top_p_spin.setRange(0.0, 1.0)
        self.llm_top_p_spin.setSingleStep(0.01)
        self.llm_top_p_spin.setValue(get_llm_top_p())
        self._add_generation_form_row(ai_form, "Top-P sampling", _gen_top_p_tip, self.llm_top_p_spin)

        self.llm_min_p_spin = NoScrollDoubleSpinBox()
        self.llm_min_p_spin.setRange(0.0, 1.0)
        self.llm_min_p_spin.setSingleStep(0.01)
        self.llm_min_p_spin.setValue(get_llm_min_p())
        self._add_generation_form_row(ai_form, "Min-P sampling", _gen_min_p_tip, self.llm_min_p_spin)

        self._wire_llm_generation_settings()

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
        self._native_chat_format_items = [
            ("Auto (GGUF / library default)", "auto"),
            ("GGUF Jinja (tokenizer.chat_template)", "jinja"),
            ("ChatML", "chatml"),
            ("Llama 3 Instruct", "llama-3"),
            ("Mistral / Mixtral Instruct", "mistral"),
            ("Llama 2 Chat", "llama-2"),
        ]
        self._build_prestige_menu(
            self.native_chat_format_selector,
            self._native_chat_format_items,
            self._on_native_chat_format_changed,
        )
        self.native_chat_format_reset_btn = QPushButton("Reset")
        self.native_chat_format_reset_btn.setToolTip(
            "Reset to automatic template selection for the currently loaded model."
        )
        self.native_chat_format_reset_btn.clicked.connect(
            self._on_reset_native_chat_format_clicked
        )
        chat_template_row = QWidget()
        chat_template_row_layout = QHBoxLayout(chat_template_row)
        chat_template_row_layout.setContentsMargins(0, 0, 0, 0)
        chat_template_row_layout.setSpacing(8)
        chat_template_row_layout.addWidget(self.native_chat_format_selector, stretch=1)
        chat_template_row_layout.addWidget(self.native_chat_format_reset_btn)
        self._sync_native_chat_template_label()

        self.models_dir_label = QLabel()
        self.models_dir_label.setWordWrap(True)

        local_row = QHBoxLayout()
        self.local_gguf_list = QListWidget()
        self.local_gguf_list.setMinimumHeight(100)
        self.local_gguf_list.setMaximumHeight(160)
        self.local_gguf_list.setToolTip(
            "Downloaded .gguf models on this device. Select one, then click Use selected."
        )
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
        native_form.addRow("Chat template (internal)", chat_template_row)
        native_form.addRow("Model storage", self.models_dir_label)
        native_form.addRow("On this device", local_row)
        native_form.addRow("Active model", self.active_native_model_lbl)

        _adv_tip = (
            "Advanced engine controls are not for everyday use. Only enable them if you "
            "have a very powerful machine with plenty of RAM.\n\n"
            "Unlocks optional auxiliary cognition model selection. The cognition model "
            "runs on CPU RAM in parallel with your primary chat model — larger swaps "
            "(e.g. 1.5B+) reduce headroom available for conversation."
        )
        self.advanced_engine_toggle = PrestigeToggle()
        self.advanced_engine_label = QLabel("Show advanced engine settings")
        self.advanced_engine_toggle.setToolTip(_adv_tip)
        self.advanced_engine_label.setToolTip(_adv_tip)
        self.advanced_engine_info_btn = self._make_settings_info_button(_adv_tip)
        label_cluster = QWidget()
        label_cluster_layout = QHBoxLayout(label_cluster)
        label_cluster_layout.setContentsMargins(0, 0, 0, 0)
        label_cluster_layout.setSpacing(6)
        label_cluster_layout.addWidget(self.advanced_engine_label)
        label_cluster_layout.addWidget(self.advanced_engine_info_btn)
        advanced_row = QWidget()
        advanced_row_layout = QHBoxLayout(advanced_row)
        advanced_row_layout.setContentsMargins(0, 0, 0, 0)
        advanced_row_layout.setSpacing(8)
        advanced_row_layout.addWidget(
            self.advanced_engine_toggle, alignment=Qt.AlignmentFlag.AlignLeft
        )
        advanced_row_layout.addWidget(label_cluster)
        advanced_row_layout.addStretch(1)
        self.advanced_engine_toggle.blockSignals(True)
        self.advanced_engine_toggle.setChecked(get_advanced_engine_unlocked())
        self.advanced_engine_toggle.blockSignals(False)
        self.advanced_engine_toggle.toggled.connect(self._on_advanced_engine_toggled)
        native_form.addRow("", advanced_row)

        self.advanced_engine_panel = QWidget()
        adv_panel_layout = QVBoxLayout(self.advanced_engine_panel)
        adv_panel_layout.setContentsMargins(0, 8, 0, 0)
        adv_panel_layout.setSpacing(12)

        cognition_dir = get_cognition_models_dir()
        self.cognition_dir_label = QLabel(cognition_dir)
        self.cognition_dir_label.setWordWrap(True)
        self.cognition_dir_label.setToolTip(
            "Place optional cognition .gguf files here. The bundled Qwen3 1.7B default "
            "also lives in this folder."
        )

        cognition_row = QHBoxLayout()
        self.cognition_gguf_list = QListWidget()
        self.cognition_gguf_list.setMinimumHeight(90)
        self.cognition_gguf_list.setMaximumHeight(140)
        self.cognition_gguf_list.setToolTip(
            "Built-in Qwen3 1.7B default cannot be deleted. Select a custom model and "
            "click Use selected, or Reset to default."
        )
        cognition_row.addWidget(self.cognition_gguf_list, stretch=1)
        cognition_btn_col = QVBoxLayout()
        cognition_btn_col.setSpacing(8)
        self.use_cognition_gguf_btn = QPushButton("Use selected")
        apply_brand_primary(self.use_cognition_gguf_btn)
        self.use_cognition_gguf_btn.clicked.connect(self._apply_selected_cognition_gguf)
        cognition_btn_col.addWidget(
            self.use_cognition_gguf_btn, alignment=Qt.AlignmentFlag.AlignTop
        )
        self.reset_cognition_btn = QPushButton("Reset to default")
        apply_brand_primary(self.reset_cognition_btn, icon_name="fa5s.undo")
        self.reset_cognition_btn.clicked.connect(self._reset_cognition_to_default)
        cognition_btn_col.addWidget(
            self.reset_cognition_btn, alignment=Qt.AlignmentFlag.AlignTop
        )
        self.delete_cognition_gguf_btn = QPushButton("Delete")
        apply_brand_danger(self.delete_cognition_gguf_btn)
        self.delete_cognition_gguf_btn.clicked.connect(self._delete_selected_cognition_gguf)
        cognition_btn_col.addWidget(
            self.delete_cognition_gguf_btn, alignment=Qt.AlignmentFlag.AlignTop
        )
        cognition_row.addLayout(cognition_btn_col)

        self.cognition_chat_format_selector = SelectorButton(
            "Cognition chat template...", is_dark=is_dark
        )
        self.cognition_chat_format_selector.setMaximumWidth(350)
        self.cognition_chat_format_selector.setMenu(
            QMenu(self.cognition_chat_format_selector)
        )
        self.cognition_chat_format_selector.setToolTip(
            "Prompt format for the auxiliary cognition model. Auto infers from filename."
        )
        self._cognition_chat_format_items = [
            ("Auto (from filename)", "auto"),
            ("ChatML", "chatml"),
            ("Llama 3 Instruct", "llama-3"),
            ("Phi-3", "phi"),
            ("Gemma", "gemma"),
        ]
        self._build_prestige_menu(
            self.cognition_chat_format_selector,
            self._cognition_chat_format_items,
            self._on_cognition_chat_format_changed,
        )
        self._sync_cognition_chat_format_label()

        self.active_cognition_model_lbl = QLabel()
        self.active_cognition_model_lbl.setWordWrap(True)

        adv_panel_layout.addWidget(
            QLabel("Optional cognition models directory:")
        )
        adv_panel_layout.addWidget(self.cognition_dir_label)
        adv_panel_layout.addLayout(cognition_row)
        adv_panel_layout.addWidget(
            QLabel("Cognition chat template (advanced)")
        )
        adv_panel_layout.addWidget(self.cognition_chat_format_selector)
        adv_panel_layout.addWidget(self.active_cognition_model_lbl)

        native_form.addRow("", self.advanced_engine_panel)
        self.advanced_engine_panel.setVisible(get_advanced_engine_unlocked())
        self._refresh_cognition_gguf_list()
        self._sync_active_cognition_label()

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

        # --- CHAT ---
        content_layout.addWidget(self._build_section_header("fa5s.comments", "CHAT"))
        chat_widget = QWidget()
        chat_widget.setObjectName("SettingsFormContainer")
        chat_form = QFormLayout(chat_widget)
        chat_form.setSpacing(15)
        chat_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.chat_personality_toggle = PrestigeToggle()
        self.chat_personality_label = QLabel("Encourage brief follow-ups on general chat")
        self.chat_personality_label.setWordWrap(True)
        _chat_personality_tip = (
            "When enabled, plain chat turns (no library or memory sources) "
            "gently invite one optional short follow-up—e.g. after a joke or "
            "story—not on retrieval, web search, or remember-this turns. "
            "On by default."
        )
        self.chat_personality_toggle.setToolTip(_chat_personality_tip)
        self.chat_personality_label.setToolTip(_chat_personality_tip)
        chat_personality_row = QWidget()
        chat_personality_row_layout = QHBoxLayout(chat_personality_row)
        chat_personality_row_layout.setContentsMargins(0, 0, 0, 0)
        chat_personality_row_layout.addWidget(
            self.chat_personality_toggle, alignment=Qt.AlignmentFlag.AlignLeft
        )
        chat_personality_row_layout.addWidget(self.chat_personality_label, stretch=1)
        self.chat_personality_toggle.blockSignals(True)
        self.chat_personality_toggle.setChecked(get_enable_chat_personality_nudge())
        self.chat_personality_toggle.blockSignals(False)
        self.chat_personality_toggle.toggled.connect(self._on_chat_personality_toggled)
        chat_form.addRow("", chat_personality_row)
        content_layout.addWidget(chat_widget)
        content_layout.addWidget(self._build_divider())

        # --- SECTION: MEMORY & PERFORMANCE (Low-end / RAM) ---
        content_layout.addWidget(self._build_section_header("fa5s.memory", "MEMORY & PERFORMANCE"))
        perf_widget = QWidget()
        perf_widget.setObjectName("SettingsFormContainer")
        perf_form = QFormLayout(perf_widget)
        perf_form.setSpacing(15)
        perf_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.memory_enrichment_toggle = PrestigeToggle()
        self.mem_enrichment_label = QLabel("Enable Memory Enrichment & Reflection (Requires more RAM)")
        self.mem_enrichment_label.setWordWrap(True)
        _mem_enrichment_tip = (
            "When enabled, Qube extracts durable facts from chat, summarises sessions "
            "into episodic memories, and runs a periodic LLM audit that flags suspicious "
            "stored memories for review. Uses more RAM and background LLM time. "
            "When disabled, existing memories and retrieval still work; usage counters "
            "and decay maintenance for stored rows continue."
        )
        self.memory_enrichment_toggle.setToolTip(_mem_enrichment_tip)
        self.mem_enrichment_label.setToolTip(_mem_enrichment_tip)
        mem_row = QWidget()
        mem_row_layout = QHBoxLayout(mem_row)
        mem_row_layout.setContentsMargins(0, 0, 0, 0)
        mem_row_layout.addWidget(self.memory_enrichment_toggle, alignment=Qt.AlignmentFlag.AlignLeft)
        mem_row_layout.addWidget(self.mem_enrichment_label, stretch=1)

        self.memory_enrichment_toggle.blockSignals(True)
        self.memory_enrichment_toggle.setChecked(get_enable_memory_enrichment())
        self.memory_enrichment_toggle.blockSignals(False)
        self.memory_enrichment_toggle.toggled.connect(self._on_memory_enrichment_toggled)

        self.memory_promotion_toggle = PrestigeToggle()
        self.mem_promotion_label = QLabel("Promote well-used memories to preferences")
        self.mem_promotion_label.setWordWrap(True)
        _mem_promotion_tip = (
            "When this is on, Qube occasionally upgrades facts you rely on often into "
            "long-term preferences — the kind of thing Qube should remember about you "
            "without being asked each time.\n\n"
            "It looks at how often a memory is recalled in chat, whether answers actually "
            "use it, and whether it comes up in different conversations. Requires "
            "Memory Enrichment above.\n\n"
            "Off by default. Qube never removes memories on its own — you can always "
            "review or edit everything in Memory Manager."
        )
        self.memory_promotion_toggle.setToolTip(_mem_promotion_tip)
        self.mem_promotion_label.setToolTip(_mem_promotion_tip)
        promo_row = QWidget()
        promo_row_layout = QHBoxLayout(promo_row)
        promo_row_layout.setContentsMargins(0, 0, 0, 0)
        promo_row_layout.addWidget(self.memory_promotion_toggle, alignment=Qt.AlignmentFlag.AlignLeft)
        promo_row_layout.addWidget(self.mem_promotion_label, stretch=1)
        self.memory_promotion_toggle.blockSignals(True)
        self.memory_promotion_toggle.setChecked(get_enable_memory_promotion())
        self.memory_promotion_toggle.blockSignals(False)
        self.memory_promotion_toggle.toggled.connect(self._on_memory_promotion_toggled)

        self.memory_promotion_preset_selector = SelectorButton("Standard", is_dark=getattr(self.window(), "_is_dark_theme", True))
        self.memory_promotion_preset_selector.setMinimumWidth(200)
        self.memory_promotion_preset_selector.setMaximumWidth(280)
        self.memory_promotion_preset_selector.setToolTip(
            "How cautious Qube should be before promoting a memory.\n\n"
            "Conservative — waits for more repeated use before upgrading.\n"
            "Standard — balanced default.\n"
            "Aggressive — promotes sooner.\n\n"
            "Only applies when Promote well-used memories is enabled."
        )
        self._build_memory_promotion_preset_menu()

        self.memory_consolidation_toggle = PrestigeToggle()
        self.mem_consolidation_label = QLabel("Highlight memories that keep coming back")
        self.mem_consolidation_label.setWordWrap(True)
        _mem_consolidation_tip = (
            "When this is on, Qube watches for memories that show up again on "
            "different days — a hint they may matter more than one-off notes.\n\n"
            "Those items get a gentle nudge in Memory Manager (marked for your "
            "review). Qube does not rewrite or delete them automatically, and "
            "this runs quietly in the background.\n\n"
            "On by default. Turn off if you prefer to curate memories only yourself."
        )
        self.memory_consolidation_toggle.setToolTip(_mem_consolidation_tip)
        self.mem_consolidation_label.setToolTip(_mem_consolidation_tip)
        consolidate_row = QWidget()
        consolidate_row_layout = QHBoxLayout(consolidate_row)
        consolidate_row_layout.setContentsMargins(0, 0, 0, 0)
        consolidate_row_layout.addWidget(self.memory_consolidation_toggle, alignment=Qt.AlignmentFlag.AlignLeft)
        consolidate_row_layout.addWidget(self.mem_consolidation_label, stretch=1)
        self.memory_consolidation_toggle.blockSignals(True)
        self.memory_consolidation_toggle.setChecked(get_enable_memory_consolidation())
        self.memory_consolidation_toggle.blockSignals(False)
        self.memory_consolidation_toggle.toggled.connect(self._on_memory_consolidation_toggled)

        promo_preset_row = QWidget()
        promo_preset_layout = QHBoxLayout(promo_preset_row)
        promo_preset_layout.setContentsMargins(0, 0, 0, 0)
        self._promo_preset_lbl = QLabel("Promotion preset")
        self._promo_preset_lbl.setToolTip(self.memory_promotion_preset_selector.toolTip())
        promo_preset_layout.addWidget(self._promo_preset_lbl)
        promo_preset_layout.addWidget(self.memory_promotion_preset_selector)
        promo_preset_layout.addStretch(1)

        self._sync_memory_promotion_controls_for_enrichment()

        perf_form.addRow("", mem_row)
        perf_form.addRow("", promo_row)
        perf_form.addRow("", promo_preset_row)
        perf_form.addRow("", consolidate_row)

        self.profile_units_selector = SelectorButton("Use inferred units", is_dark=is_dark)
        self.profile_units_selector.setMinimumWidth(200)
        self.profile_units_selector.setMaximumWidth(280)
        self.profile_units_selector.setToolTip(
            "Default measurement units for weather and other numeric answers. "
            "Unset lets Qube learn units from conversation."
        )
        self._build_profile_units_menu()
        profile_units_row = QWidget()
        profile_units_layout = QHBoxLayout(profile_units_row)
        profile_units_layout.setContentsMargins(0, 0, 0, 0)
        profile_units_lbl = QLabel("Default units")
        profile_units_lbl.setToolTip(self.profile_units_selector.toolTip())
        profile_units_layout.addWidget(profile_units_lbl)
        profile_units_layout.addWidget(self.profile_units_selector)
        profile_units_layout.addStretch(1)
        self._sync_profile_units_selector()
        perf_form.addRow("", profile_units_row)

        content_layout.addWidget(perf_widget)
        content_layout.addWidget(self._build_divider())

        # --- SECTION: NOTIFICATIONS ---
        content_layout.addWidget(self._build_section_header("fa5s.bell", "NOTIFICATIONS"))
        notif_widget = QWidget()
        notif_widget.setObjectName("SettingsFormContainer")
        notif_layout = QVBoxLayout(notif_widget)
        notif_layout.setContentsMargins(15, 0, 15, 10)
        notif_layout.setSpacing(8)

        from core import app_settings as _notif_settings

        self.notifications_enabled_cb = QCheckBox("Enable notifications")
        self.notifications_enabled_cb.setChecked(_notif_settings.get_notifications_enabled())
        self.notifications_enabled_cb.toggled.connect(_notif_settings.set_notifications_enabled)
        notif_layout.addWidget(self.notifications_enabled_cb)

        self.notifications_dnd_cb = QCheckBox("Do Not Disturb (critical only)")
        self.notifications_dnd_cb.setChecked(_notif_settings.get_notifications_dnd())
        self.notifications_dnd_cb.toggled.connect(self._on_notifications_dnd_toggled)
        notif_layout.addWidget(self.notifications_dnd_cb)

        self.notifications_suppress_focus_cb = QCheckBox("Suppress info/success while app is focused")
        self.notifications_suppress_focus_cb.setChecked(
            _notif_settings.get_notifications_suppress_when_focused()
        )
        self.notifications_suppress_focus_cb.toggled.connect(
            _notif_settings.set_notifications_suppress_when_focused
        )
        notif_layout.addWidget(self.notifications_suppress_focus_cb)

        self.notifications_os_hidden_cb = QCheckBox("OS notifications when hidden")
        self.notifications_os_hidden_cb.setChecked(_notif_settings.get_notifications_os_when_hidden())
        self.notifications_os_hidden_cb.toggled.connect(_notif_settings.set_notifications_os_when_hidden)
        notif_layout.addWidget(self.notifications_os_hidden_cb)

        self.notifications_sound_cb = QCheckBox("Play alert sounds")
        self.notifications_sound_cb.setChecked(_notif_settings.get_notifications_sound_enabled())
        self.notifications_sound_cb.toggled.connect(_notif_settings.set_notifications_sound_enabled)
        notif_layout.addWidget(self.notifications_sound_cb)

        self.notifications_preview_cb = QCheckBox("Show message preview in notifications")
        self.notifications_preview_cb.setChecked(_notif_settings.get_notifications_show_preview())
        self.notifications_preview_cb.toggled.connect(_notif_settings.set_notifications_show_preview)
        notif_layout.addWidget(self.notifications_preview_cb)

        self.notifications_memory_cb = QCheckBox("Memory extraction notifications")
        self.notifications_memory_cb.setChecked(_notif_settings.get_notifications_category_memory())
        self.notifications_memory_cb.toggled.connect(_notif_settings.set_notifications_category_memory)
        notif_layout.addWidget(self.notifications_memory_cb)

        clear_history_btn = QPushButton("Clear notification history")
        clear_history_btn.clicked.connect(self._clear_notification_history)
        notif_layout.addWidget(clear_history_btn)

        content_layout.addWidget(notif_widget)
        content_layout.addWidget(self._build_divider())

        # --- SECTION: DESKTOP COMPANION ---
        content_layout.addWidget(self._build_section_header("fa5s.ghost", "DESKTOP COMPANION"))
        companion_widget = QWidget()
        companion_widget.setObjectName("SettingsFormContainer")
        companion_layout = QVBoxLayout(companion_widget)
        companion_layout.setContentsMargins(15, 0, 15, 10)
        companion_layout.setSpacing(8)

        from core import app_settings as _companion_settings
        from core.platform.companion_capabilities import (
            detect_companion_platform_tier,
            tier_display_name,
        )

        tier = detect_companion_platform_tier()
        tier_lbl = QLabel(f"Platform: {tier_display_name(tier)}")
        tier_lbl.setWordWrap(True)
        _companion_tier_tip = (
            "What Qube detected for floating overlay support on this system. "
            "Full tier is typical on Windows and macOS; Linux Wayland is usually degraded "
            "(dock strip or tray fallback recommended)."
        )
        tier_lbl.setToolTip(_companion_tier_tip)
        companion_layout.addWidget(tier_lbl)

        _companion_enabled_tip = (
            "Master switch for the desktop companion orb or dock strip. "
            "When off, chat, voice, tray, and notifications still work."
        )
        self.companion_enabled_cb = QCheckBox("Enable desktop companion")
        self.companion_enabled_cb.setToolTip(_companion_enabled_tip)
        self.companion_enabled_cb.setChecked(_companion_settings.get_companion_enabled())
        self.companion_enabled_cb.toggled.connect(self._on_companion_enabled_toggled)
        companion_layout.addWidget(self.companion_enabled_cb)

        _companion_tray_tip = (
            "Show the companion when the main window is minimized or closed to the tray. "
            "Turn off if you only want the companion while the app window is visible."
        )
        self.companion_tray_hidden_cb = QCheckBox("Show when hidden to tray")
        self.companion_tray_hidden_cb.setToolTip(_companion_tray_tip)
        self.companion_tray_hidden_cb.setChecked(_companion_settings.get_companion_show_when_tray_hidden())
        self.companion_tray_hidden_cb.toggled.connect(self._on_companion_setting_changed)
        companion_layout.addWidget(self.companion_tray_hidden_cb)

        _companion_while_open_tip = (
            "Keep the companion visible even when the main Qube window is open and not minimized. "
            "Uncheck to hide the companion whenever the main window is in the foreground."
        )
        self.companion_while_open_cb = QCheckBox("Show while main window is open")
        self.companion_while_open_cb.setToolTip(_companion_while_open_tip)
        self.companion_while_open_cb.setChecked(_companion_settings.get_companion_show_while_window_open())
        self.companion_while_open_cb.toggled.connect(self._on_companion_setting_changed)
        companion_layout.addWidget(self.companion_while_open_cb)

        _companion_auto_hide_tip = (
            "Fade the companion when Qube has been idle for a while (listening with no speech). "
            "It reappears when you interact or when assistant activity resumes."
        )
        self.companion_auto_hide_cb = QCheckBox("Auto-hide when idle")
        self.companion_auto_hide_cb.setToolTip(_companion_auto_hide_tip)
        self.companion_auto_hide_cb.setChecked(_companion_settings.get_companion_auto_hide_idle())
        self.companion_auto_hide_cb.toggled.connect(self._on_companion_setting_changed)
        companion_layout.addWidget(self.companion_auto_hide_cb)

        self.companion_caption_cb = QCheckBox("Show activity label under companion")
        self.companion_caption_cb.setToolTip(
            "When enabled, a short status chip appears below the companion "
            "(Idle, Listening, Thinking, Writing, Speaking). Uncheck to show only the companion widget."
        )
        self.companion_caption_cb.setChecked(_companion_settings.get_companion_show_caption())
        self.companion_caption_cb.toggled.connect(self._on_companion_setting_changed)
        companion_layout.addWidget(self.companion_caption_cb)

        _companion_verbal_section_tip = (
            "Optional short lines under the companion, generated by the auxiliary cognition model. "
            "Does not change chat replies or TTS."
        )
        verbal_lbl = QLabel("Companion commentary")
        verbal_lbl.setObjectName("SettingsSubsectionLabel")
        verbal_lbl.setToolTip(_companion_verbal_section_tip)
        companion_layout.addWidget(verbal_lbl)

        self.companion_verbal_enabled_cb = QCheckBox("Enable companion commentary")
        self.companion_verbal_enabled_cb.setToolTip(
            "When enabled, the auxiliary cognition model may write short caption lines "
            "under the companion while idle or after ingest/download events. "
            "Does not affect chat replies."
        )
        self.companion_verbal_enabled_cb.setChecked(
            _companion_settings.get_companion_verbal_enabled()
        )
        self.companion_verbal_enabled_cb.toggled.connect(self._on_companion_verbal_setting_changed)
        companion_layout.addWidget(self.companion_verbal_enabled_cb)

        self.companion_cognition_v2_cb = QCheckBox("Companion Cognition v2 (curated + intentional captions)")
        self.companion_cognition_v2_cb.setToolTip(
            "Uses a deterministic observation → thought → expression pipeline with a curated "
            "message library. Sidecar is used only for optional rephrasing on capable models (1.7B+)."
        )
        self.companion_cognition_v2_cb.setChecked(
            _companion_settings.get_companion_cognition_v2_enabled()
        )
        self.companion_cognition_v2_cb.toggled.connect(self._on_companion_verbal_setting_changed)
        companion_layout.addWidget(self.companion_cognition_v2_cb)

        _companion_freedom_tip = (
            "How creative companion commentary may be (Cognition v2).\n\n"
            "Conservative — curated library only; no sidecar rephrasing.\n"
            "Balanced — capability follows your auxiliary model size.\n"
            "Expressive — richer lines plus sidecar rephrasing or generation when supported."
        )
        freedom_row = QHBoxLayout()
        freedom_row.setSpacing(8)
        freedom_lbl = QLabel("Expression freedom")
        freedom_lbl.setToolTip(_companion_freedom_tip)
        freedom_row.addWidget(freedom_lbl)
        self.companion_expression_freedom_selector = SelectorButton("Balanced", is_dark=is_dark)
        self.companion_expression_freedom_selector.setMinimumWidth(180)
        self.companion_expression_freedom_selector.setMaximumWidth(250)
        self.companion_expression_freedom_selector.setToolTip(_companion_freedom_tip)
        self.companion_expression_freedom_selector.setMenu(
            QMenu(self.companion_expression_freedom_selector)
        )
        self._build_companion_expression_freedom_menu()
        freedom_row.addWidget(self.companion_expression_freedom_selector)
        freedom_row.addStretch()
        companion_layout.addLayout(freedom_row)

        self.companion_verbal_prompt = QPlainTextEdit()
        self.companion_verbal_prompt.setPlaceholderText(
            "Optional companion-only style notes (does not affect chat replies)…"
        )
        self.companion_verbal_prompt.setMaximumHeight(90)
        self.companion_verbal_prompt.setToolTip(
            "Appended to the companion commentary prompt only. Max 800 characters."
        )
        self.companion_verbal_prompt.setPlainText(
            _companion_settings.get_companion_verbal_system_prompt()
        )
        self.companion_verbal_prompt.textChanged.connect(self._on_companion_verbal_prompt_changed)
        companion_layout.addWidget(self.companion_verbal_prompt)

        _companion_trait_tip = (
            "Tone preset for companion commentary prompts.\n\n"
            "Neutral — calm and brief.\n"
            "Warm — gently encouraging.\n"
            "Witty / Dry / Light sarcastic — humor variants; never insulting or distracting."
        )
        trait_row = QHBoxLayout()
        trait_row.setSpacing(8)
        trait_lbl = QLabel("Personality")
        trait_lbl.setToolTip(_companion_trait_tip)
        trait_row.addWidget(trait_lbl)
        self.companion_verbal_trait_selector = SelectorButton("Neutral", is_dark=is_dark)
        self.companion_verbal_trait_selector.setMinimumWidth(180)
        self.companion_verbal_trait_selector.setMaximumWidth(250)
        self.companion_verbal_trait_selector.setToolTip(_companion_trait_tip)
        self.companion_verbal_trait_selector.setMenu(QMenu(self.companion_verbal_trait_selector))
        self._build_companion_verbal_trait_menu()
        trait_row.addWidget(self.companion_verbal_trait_selector)
        trait_row.addStretch()
        companion_layout.addLayout(trait_row)

        _companion_freq_tip = (
            "Spacing for proactive idle commentary while the assistant is listening and idle.\n\n"
            "Rare — after 2 min idle, at most one line every ~45 min.\n"
            "Normal — after 1 min idle, at most one line every ~15 min.\n"
            "Chatty — after 30 sec idle, at most one line every ~5 min.\n\n"
            "Requires companion commentary enabled and the companion visible. "
            "With the main window open, idle lines only appear when "
            "'Show while main window is open' is enabled. "
            "Ingest/download reactions use separate cooldowns."
        )
        freq_row = QHBoxLayout()
        freq_row.setSpacing(8)
        freq_lbl = QLabel("How often")
        freq_lbl.setToolTip(_companion_freq_tip)
        freq_row.addWidget(freq_lbl)
        self.companion_verbal_frequency_selector = SelectorButton("Normal", is_dark=is_dark)
        self.companion_verbal_frequency_selector.setMinimumWidth(180)
        self.companion_verbal_frequency_selector.setMaximumWidth(250)
        self.companion_verbal_frequency_selector.setToolTip(_companion_freq_tip)
        self.companion_verbal_frequency_selector.setMenu(
            QMenu(self.companion_verbal_frequency_selector)
        )
        self._build_companion_verbal_frequency_menu()
        freq_row.addWidget(self.companion_verbal_frequency_selector)
        freq_row.addStretch()
        companion_layout.addLayout(freq_row)

        self.companion_verbal_react_ingest_cb = QCheckBox("Comment when library ingest completes")
        self.companion_verbal_react_ingest_cb.setToolTip(
            "After a document finishes indexing in the Library, the companion may show a "
            "short acknowledgment line (subject to commentary being enabled and rate limits)."
        )
        self.companion_verbal_react_ingest_cb.setChecked(
            _companion_settings.get_companion_verbal_react_ingest()
        )
        self.companion_verbal_react_ingest_cb.toggled.connect(
            self._on_companion_verbal_setting_changed
        )
        companion_layout.addWidget(self.companion_verbal_react_ingest_cb)

        self.companion_verbal_react_download_cb = QCheckBox("Comment when a model download completes")
        self.companion_verbal_react_download_cb.setToolTip(
            "After a Model Manager download finishes, the companion may show a brief line "
            "celebrating or noting the new model (rate-limited like other commentary)."
        )
        self.companion_verbal_react_download_cb.setChecked(
            _companion_settings.get_companion_verbal_react_download()
        )
        self.companion_verbal_react_download_cb.toggled.connect(
            self._on_companion_verbal_setting_changed
        )
        companion_layout.addWidget(self.companion_verbal_react_download_cb)

        test_row = QHBoxLayout()
        test_row.setSpacing(8)
        self.companion_verbal_test_btn = QPushButton("Test commentary")
        self.companion_verbal_test_btn.setToolTip(
            "Generate a sample caption using the auxiliary cognition model and your "
            "current personality / prompt settings."
        )
        apply_brand_primary(self.companion_verbal_test_btn, icon_name="fa5s.comment-dots")
        self.companion_verbal_test_btn.clicked.connect(self._on_companion_verbal_test_clicked)
        test_row.addWidget(self.companion_verbal_test_btn)
        test_row.addStretch()
        companion_layout.addLayout(test_row)

        self.companion_verbal_test_result = QLabel(
            "Run Test to preview a sample companion caption here."
        )
        self.companion_verbal_test_result.setWordWrap(True)
        self.companion_verbal_test_result.setObjectName("CompanionVerbalTestResult")
        self.companion_verbal_test_result.setToolTip(
            "Shows the last Test commentary preview from this settings page."
        )
        companion_layout.addWidget(self.companion_verbal_test_result)

        verbal_info = QLabel(
            "Uses the auxiliary cognition model on CPU (bundled Qwen3 1.7B). For lighter "
            "CPU use, place Qwen2 0.5B or Qwen2-1.5B-Instruct in models/cognition/ under "
            "Advanced engine settings."
        )
        verbal_info.setWordWrap(True)
        verbal_info.setToolTip(
            "Companion commentary runs on the auxiliary cognition sidecar (CPU GGUF), not your "
            "main chat model. Swap a smaller GGUF under Advanced engine settings to reduce load."
        )
        companion_layout.addWidget(verbal_info)

        self._sync_companion_verbal_controls_enabled()

        self.companion_fullscreen_cb = QCheckBox("Hide during fullscreen apps")
        self.companion_fullscreen_cb.setToolTip(
            "Hide the companion while another app is fullscreen, unless Qube needs your "
            "attention (listening, thinking, speaking, or an error)."
        )
        self.companion_fullscreen_cb.setChecked(_companion_settings.get_companion_suppress_on_fullscreen())
        self.companion_fullscreen_cb.toggled.connect(self._on_companion_setting_changed)
        companion_layout.addWidget(self.companion_fullscreen_cb)

        self.companion_wayland_cb = QCheckBox("Try floating overlay on Wayland (experimental)")
        self.companion_wayland_cb.setToolTip(
            "On Linux Wayland, global always-on-top overlays are often blocked. Enable to "
            "attempt the floating orb anyway; if it fails, use edge dock strip mode instead."
        )
        self.companion_wayland_cb.setChecked(_companion_settings.get_companion_try_on_wayland())
        self.companion_wayland_cb.toggled.connect(self._on_companion_setting_changed)
        companion_layout.addWidget(self.companion_wayland_cb)

        self.companion_dock_cb = QCheckBox("Use edge dock strip mode (better on Wayland)")
        self.companion_dock_cb.setToolTip(
            "Shows a thin dock strip along the screen edge instead of a floating orb. "
            "Usually works better on Wayland than the experimental overlay."
        )
        self.companion_dock_cb.setChecked(_companion_settings.get_companion_dock_mode())
        self.companion_dock_cb.toggled.connect(self._on_companion_setting_changed)
        companion_layout.addWidget(self.companion_dock_cb)

        _companion_appearance_tip = (
            "Visual style for the companion widget and live preview below."
        )
        appearance_lbl = QLabel("Companion shape")
        appearance_lbl.setObjectName("SettingsSubsectionLabel")
        appearance_lbl.setToolTip(_companion_appearance_tip)
        companion_layout.addWidget(appearance_lbl)

        from core.companion_personas import (
            CompanionPersonaId,
            PERSONA_DESCRIPTIONS,
            PERSONA_LABELS,
        )
        from core.companion_idle_color import (
            CompanionIdleColor,
            IDLE_COLOR_DESCRIPTIONS,
            IDLE_COLOR_LABELS,
        )
        from ui.companion.companion_preview import CompanionPreviewWidget

        persona_row = QHBoxLayout()
        persona_row.setSpacing(16)
        self.companion_persona_group = QButtonGroup(self)
        self.companion_persona_group.setExclusive(True)
        current_persona = _companion_settings.get_companion_persona()
        self.companion_persona_cbs: dict[CompanionPersonaId, QCheckBox] = {}
        for persona_id in (CompanionPersonaId.SPHERE, CompanionPersonaId.QUBE):
            cb = QCheckBox(PERSONA_LABELS[persona_id])
            cb.setToolTip(PERSONA_DESCRIPTIONS[persona_id])
            cb.setProperty("companion_persona_id", persona_id.value)
            cb.setChecked(persona_id == current_persona)
            self.companion_persona_group.addButton(cb)
            self.companion_persona_cbs[persona_id] = cb
            persona_row.addWidget(cb)
        self.companion_persona_group.buttonToggled.connect(self._on_companion_persona_toggled)
        persona_row.addStretch()
        companion_layout.addLayout(persona_row)

        _companion_idle_color_tip = (
            "Accent color for the companion glow while idle. "
            "Does not change colors during listening, thinking, or speaking states."
        )
        idle_color_lbl = QLabel("Companion idle glow color")
        idle_color_lbl.setObjectName("SettingsSubsectionLabel")
        idle_color_lbl.setToolTip(_companion_idle_color_tip)
        companion_layout.addWidget(idle_color_lbl)

        self.companion_idle_color_group = QButtonGroup(self)
        self.companion_idle_color_group.setExclusive(True)
        current_idle_color = _companion_settings.get_companion_idle_color()
        self.companion_idle_color_cbs: dict[CompanionIdleColor, QCheckBox] = {}
        for color_id in (CompanionIdleColor.PURPLE, CompanionIdleColor.BLUE):
            cb = QCheckBox(IDLE_COLOR_LABELS[color_id])
            cb.setToolTip(IDLE_COLOR_DESCRIPTIONS[color_id])
            cb.setProperty("companion_idle_color_id", color_id.value)
            cb.setChecked(color_id == current_idle_color)
            self.companion_idle_color_group.addButton(cb)
            self.companion_idle_color_cbs[color_id] = cb
            companion_layout.addWidget(cb)
        self.companion_idle_color_group.buttonToggled.connect(self._on_companion_idle_color_toggled)

        _companion_demo_tip = (
            "Pick an assistant activity to preview animations and caption styling "
            "in the companion preview below (does not affect the live companion)."
        )
        demo_row = QHBoxLayout()
        demo_row.setSpacing(8)
        demo_lbl = QLabel("Preview state:")
        demo_lbl.setToolTip(_companion_demo_tip)
        demo_row.addWidget(demo_lbl)
        self.companion_demo_selector = SelectorButton("", is_dark=is_dark)
        self.companion_demo_selector.setMinimumWidth(180)
        self.companion_demo_selector.setMaximumWidth(250)
        self.companion_demo_selector.setToolTip(_companion_demo_tip)
        self.companion_demo_selector.setMenu(QMenu(self.companion_demo_selector))
        self._companion_demo_items = [
            ("Idle", "idle"),
            ("Listening", "capturing"),
            ("Thinking", "working"),
            ("Writing", "writing"),
            ("Speaking", "speaking"),
        ]
        self._build_prestige_menu(
            self.companion_demo_selector,
            self._companion_demo_items,
            self._on_companion_demo_state_selected,
        )
        self._sync_companion_demo_selector_label("idle")
        demo_row.addWidget(self.companion_demo_selector)
        demo_row.addStretch()
        companion_layout.addLayout(demo_row)

        self.companion_preview = CompanionPreviewWidget()
        self.companion_preview.apply_theme(is_dark)
        self.companion_preview.setToolTip(
            "Live preview of the selected persona, idle glow color, and preview activity state."
        )
        companion_layout.addWidget(self.companion_preview)

        self.companion_preview.set_persona(current_persona)
        self._on_companion_demo_state_selected("idle")

        content_layout.addWidget(companion_widget)
        content_layout.addWidget(self._build_divider())
        
        # --- 🔑 SECTION 3: NLP RAG TRIGGERS ---
        content_layout.addWidget(self._build_section_header("fa5s.bolt", "NLP RAG TRIGGERS"))
        content_layout.addWidget(self._build_triggers_manager())
        content_layout.addWidget(self._build_divider())

        # --- HELP & GUIDANCE ---
        content_layout.addWidget(
            self._build_section_header("fa5s.route", "HELP & GUIDANCE")
        )
        help_widget = QWidget()
        help_widget.setObjectName("SettingsFormContainer")
        help_layout = QVBoxLayout(help_widget)
        help_layout.setContentsMargins(15, 0, 15, 10)
        help_layout.setSpacing(8)
        help_hint = QLabel(
            "Replay the guided tour for choosing Internal Engine and loading a local .gguf model. "
            "The tour includes model picks matched to your hardware."
        )
        help_hint.setWordWrap(True)
        help_hint.setProperty("class", "ToolsPaneControl")
        self.local_llm_tour_hint_lbl = help_hint
        self.model_manager_hardware_suggestions_cb = QCheckBox(
            "Suggest models for my hardware in Model Manager"
        )
        self.model_manager_hardware_suggestions_cb.setToolTip(
            "When enabled, Model Manager ranks Qube Verified models and shows Good fit badges "
            "based on detected RAM and VRAM. The setup tour always includes personalized picks."
        )
        self.model_manager_hardware_suggestions_cb.setChecked(get_model_manager_hardware_suggestions())
        self.model_manager_hardware_suggestions_cb.toggled.connect(
            self._on_model_manager_hardware_suggestions_toggled
        )
        self.replay_local_llm_tour_btn = QPushButton("Replay Local LLM Setup Tour")
        apply_brand_primary(self.replay_local_llm_tour_btn, icon_name="fa5s.play-circle")
        self.replay_local_llm_tour_btn.setToolTip(
            "Walk through Settings, AI Engine, Select AI Model, and Model Manager with "
            "spotlight hints."
        )
        self.replay_local_llm_tour_btn.clicked.connect(self._on_replay_local_llm_tour_clicked)
        help_layout.addWidget(help_hint)
        help_layout.addWidget(self.model_manager_hardware_suggestions_cb)
        help_layout.addWidget(
            self.replay_local_llm_tour_btn,
            alignment=Qt.AlignmentFlag.AlignLeft,
        )
        content_layout.addWidget(help_widget)
        content_layout.addWidget(self._build_divider())

        # --- JSON SETTINGS ---
        content_layout.addWidget(
            self._build_section_header("fa5s.file-code", "JSON SETTINGS")
        )
        json_settings_widget = QWidget()
        json_settings_widget.setObjectName("SettingsFormContainer")
        json_settings_layout = QVBoxLayout(json_settings_widget)
        json_settings_layout.setContentsMargins(15, 0, 15, 10)
        json_settings_layout.setSpacing(8)
        self.settings_json_hint_lbl = QLabel(
            f"Edit preferences in {default_user_settings_path()} "
            "(schema: assets/config/settings.schema.json). "
            "Use the built-in editor to format, validate, and save — "
            "or reload when the file changes on disk."
        )
        self.settings_json_hint_lbl.setWordWrap(True)
        self.settings_json_hint_lbl.setProperty("class", "ToolsPaneControl")
        self.open_settings_json_btn = QPushButton("Edit settings.json")
        apply_brand_primary(self.open_settings_json_btn, icon_name="fa5s.code")
        self.open_settings_json_btn.setToolTip(
            "Open the built-in JSON editor for user settings. "
            "Format, validate, and save — or reload when the file changes on disk."
        )
        self.open_settings_json_btn.clicked.connect(self._on_open_settings_json_clicked)
        self.settings_file_status_lbl = QLabel("")
        self.settings_file_status_lbl.setProperty("class", "ToolsPaneControl")
        json_settings_layout.addWidget(self.settings_json_hint_lbl)
        json_settings_layout.addWidget(
            self.open_settings_json_btn,
            alignment=Qt.AlignmentFlag.AlignLeft,
        )
        json_settings_layout.addWidget(self.settings_file_status_lbl)
        content_layout.addWidget(json_settings_widget)

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

    def _add_generation_form_row(
        self,
        form: QFormLayout,
        label: str,
        tooltip: str,
        spinbox,
        *,
        width: int = 120,
    ) -> None:
        spinbox.setFixedWidth(width)
        spinbox.setToolTip(tooltip)
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        row_layout.addWidget(spinbox)
        row_layout.addWidget(self._make_settings_info_button(tooltip))
        row_layout.addStretch(1)
        form.addRow(label, row)
        self._generation_spinboxes.append(spinbox)

    def _wire_llm_generation_settings(self) -> None:
        llm = self.llm_worker
        if llm is None:
            return
        self.llm_temp_spin.valueChanged.connect(llm.set_temperature)
        self.llm_ctx_spin.valueChanged.connect(llm.set_context_window)
        self.llm_history_spin.valueChanged.connect(llm.set_max_history_messages)
        self.llm_top_k_spin.valueChanged.connect(llm.set_top_k)
        self.llm_repeat_penalty_spin.valueChanged.connect(llm.set_repeat_penalty)
        self.llm_presence_penalty_spin.valueChanged.connect(llm.set_presence_penalty)
        self.llm_top_p_spin.valueChanged.connect(llm.set_top_p)
        self.llm_min_p_spin.valueChanged.connect(llm.set_min_p)

    def _make_settings_info_button(self, tooltip_text: str) -> QToolButton:
        btn = QToolButton()
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setToolTip(tooltip_text)
        btn.setIcon(qta.icon("fa5s.info-circle", color="#64748b"))
        btn.setIconSize(QSize(14, 14))
        btn.setAutoRaise(True)
        btn.setStyleSheet(
            "QToolButton { border: none; padding: 0px; background: transparent; }"
        )
        return btn

    def _iter_settings_checkboxes(self):
        """All Settings-page QCheckBox widgets that share the Prestige indicator style."""
        for name in (
            "pin_audio_cb",
            "auto_load_last_model_cb",
            "auto_activator_cb",
            "model_manager_hardware_suggestions_cb",
            "notifications_enabled_cb",
            "notifications_dnd_cb",
            "notifications_suppress_focus_cb",
            "notifications_os_hidden_cb",
            "notifications_sound_cb",
            "notifications_preview_cb",
            "notifications_memory_cb",
            "companion_enabled_cb",
            "companion_tray_hidden_cb",
            "companion_while_open_cb",
            "companion_auto_hide_cb",
            "companion_caption_cb",
            "companion_fullscreen_cb",
            "companion_wayland_cb",
            "companion_dock_cb",
        ):
            cb = getattr(self, name, None)
            if cb is not None:
                yield cb
        for choice_cbs in (
            getattr(self, "companion_persona_cbs", {}),
            getattr(self, "companion_idle_color_cbs", {}),
        ):
            if isinstance(choice_cbs, dict):
                yield from choice_cbs.values()

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
        """
        checkbox_style = f"""
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
        for spinbox in getattr(self, "_generation_spinboxes", ()):
            spinbox.setStyleSheet(style)
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
        for cb in self._iter_settings_checkboxes():
            cb.setStyleSheet(checkbox_style)
        if hasattr(self, 'mem_enrichment_label'):
            self.mem_enrichment_label.setStyleSheet(f"color: {text_color}; font-size: 13px;")
        if hasattr(self, 'mem_promotion_label'):
            self.mem_promotion_label.setStyleSheet(f"color: {text_color}; font-size: 13px;")
        if hasattr(self, "local_llm_tour_hint_lbl"):
            self.local_llm_tour_hint_lbl.setStyleSheet(
                f"color: {text_color}; font-size: 13px;"
            )
        if hasattr(self, "settings_json_hint_lbl"):
            self.settings_json_hint_lbl.setStyleSheet(
                f"color: {text_color}; font-size: 13px;"
            )
        if hasattr(self, "settings_file_status_lbl"):
            self.settings_file_status_lbl.setStyleSheet(
                f"color: {text_color}; font-size: 12px;"
            )
        
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
        self._sync_active_native_model_label()

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

    def _on_replay_local_llm_tour_clicked(self) -> None:
        win = self.window()
        if win is not None and hasattr(win, "start_local_llm_onboarding_tour"):
            win.start_local_llm_onboarding_tour()
            return
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        PrestigeDialog(
            self.window(),
            "Tour unavailable",
            "The local LLM setup tour could not be started.",
            is_dark=is_dark,
        ).exec()

    def _on_model_manager_hardware_suggestions_toggled(self, enabled: bool) -> None:
        set_model_manager_hardware_suggestions(enabled)
        win = self.window()
        mm = getattr(win, "model_manager_view", None) if win is not None else None
        if mm is not None and hasattr(mm, "refresh_hardware_suggestions"):
            mm.refresh_hardware_suggestions()

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
        """Show the native model actually loaded in-process (telemetry), not only QSettings path."""
        if not hasattr(self, "active_native_model_lbl"):
            return
        mode = get_engine_mode()
        p = get_internal_model_path()
        path_name = os.path.basename(p) if p else ""

        if mode != "internal":
            if path_name:
                self.active_native_model_lbl.setText(f"{path_name} (inactive — external mode)")
            else:
                self.active_native_model_lbl.setText("(none — external mode)")
            return

        mw = self.window()
        ne = getattr(mw, "_native_engine", None) if mw else None
        snap = ne.get_model_reasoning_telemetry() if ne else None
        loaded = bool((snap or {}).get("loaded"))
        loaded_name = str((snap or {}).get("model_basename") or "").strip()

        if loaded and loaded_name:
            self.active_native_model_lbl.setText(loaded_name)
            return

        if path_name:
            self.active_native_model_lbl.setText(f"{path_name} (not loaded)")
        else:
            self.active_native_model_lbl.setText("(none)")

    def sync_active_native_model_label(self) -> None:
        """Public hook for MainWindow when the toolbar/native load state changes."""
        self._sync_active_native_model_label()

    def _on_gpu_layers_slider_changed(self, v: int) -> None:
        self.gpu_layers_value_lbl.setText(str(int(v)))
        self._on_native_gpu_layers_changed(int(v))

    def _on_cpu_threads_slider_changed(self, v: int) -> None:
        self.cpu_threads_value_lbl.setText(str(int(v)))
        set_internal_n_threads(int(v))
        llm = self.workers.get("llm")
        if llm and getattr(llm, "engine_mode", DEFAULT_ENGINE_MODE) == "internal":
            llm.refresh_native_model_from_settings()

    def _on_native_chat_format_changed(self, mode: str) -> None:
        if mode is not None:
            set_internal_native_chat_format(str(mode))
        self._sync_native_chat_template_label()
        llm = self.workers.get("llm")
        if llm and getattr(llm, "engine_mode", DEFAULT_ENGINE_MODE) == "internal":
            self._template_override_reload_pending = (
                str(mode or "").strip().lower() != "auto"
            )
            llm.refresh_native_model_from_settings()

    def _on_native_model_load_finished(self, ok: bool, _message: str) -> None:
        _ = ok
        if self._template_override_reload_pending:
            # Completion belongs to a user-requested manual template override reload.
            self._template_override_reload_pending = False
            self._sync_active_native_model_label()
            self._sync_native_chat_template_label()
            return

        if self._auto_reset_reload_pending:
            # Completion belongs to reset->reload sequence.
            self._auto_reset_reload_pending = False
            self._sync_active_native_model_label()
            self._sync_native_chat_template_label()
            return

        if get_internal_native_chat_format() != "auto":
            # Any normal model load clears persistent forcing and returns to auto template selection.
            set_internal_native_chat_format("auto")
            llm = self.workers.get("llm")
            mw = self.window()
            ne = getattr(mw, "_native_engine", None) if mw else self.workers.get("native_engine")
            snap = ne.get_model_reasoning_telemetry() if ne else None
            loaded = bool((snap or {}).get("loaded"))
            if llm and getattr(llm, "engine_mode", DEFAULT_ENGINE_MODE) == "internal" and loaded:
                self._auto_reset_reload_pending = True
                llm.refresh_native_model_from_settings()
        self._sync_active_native_model_label()
        self._sync_native_chat_template_label()

    def _saved_native_chat_format_label(self, mode: str) -> str:
        items = getattr(self, "_native_chat_format_items", None) or []
        if not items:
            return "Auto (GGUF / library default)"
        return next((label for label, data in items if data == mode), items[0][0])

    def _effective_chat_format_label(self, chat_format: str | None) -> str:
        cf = str(chat_format or "").strip().lower()
        mapping = {
            "chat_template.default": "GGUF Jinja (tokenizer.chat_template)",
            "chatml": "ChatML",
            "llama-3": "Llama 3 Instruct",
            "mistral-instruct": "Mistral / Mixtral Instruct",
            "llama-2": "Llama 2 Chat",
        }
        return mapping.get(cf, str(chat_format or "").strip())

    def _sync_native_chat_template_label(self) -> None:
        if not hasattr(self, "native_chat_format_selector"):
            return
        preferred_mode = get_internal_native_chat_format()
        preferred_label = self._saved_native_chat_format_label(preferred_mode)

        mode = get_engine_mode()
        mw = self.window()
        ne = getattr(mw, "_native_engine", None) if mw else self.workers.get("native_engine")
        snap = ne.get_model_reasoning_telemetry() if ne else None
        loaded = bool((snap or {}).get("loaded"))
        active_cf = (
            ((snap or {}).get("chat_contract") or {}).get("effective_chat_format")
            or (snap or {}).get("prompt_contract_chat_format")
            or ""
        )
        active_label = self._effective_chat_format_label(active_cf)

        if mode == "internal" and loaded and active_label:
            self.native_chat_format_selector.setText(f"{preferred_label} (active: {active_label})")
        else:
            self.native_chat_format_selector.setText(preferred_label)
        if hasattr(self, "native_chat_format_reset_btn"):
            self.native_chat_format_reset_btn.setEnabled(preferred_mode != "auto")

    def _on_reset_native_chat_format_clicked(self) -> None:
        if get_internal_native_chat_format() == "auto":
            self._sync_native_chat_template_label()
            return
        set_internal_native_chat_format("auto")
        self._sync_native_chat_template_label()
        llm = self.workers.get("llm")
        if llm and getattr(llm, "engine_mode", DEFAULT_ENGINE_MODE) == "internal":
            self._auto_reset_reload_pending = True
            llm.refresh_native_model_from_settings()

    def _on_native_gpu_layers_changed(self, v: int) -> None:
        set_internal_n_gpu_layers(int(v))
        llm = self.workers.get("llm")
        if llm and getattr(llm, "engine_mode", DEFAULT_ENGINE_MODE) == "internal":
            llm.refresh_native_model_from_settings()

    def _refresh_local_gguf_list(self) -> None:
        if not hasattr(self, "local_gguf_list"):
            return
        self.local_gguf_list.clear()
        root = Path(get_llm_models_dir())
        if not root.is_dir():
            return
        for p in sorted(
            (fp for fp in root.glob("*.gguf") if not is_secondary_gguf_shard(str(fp))),
            key=local_gguf_sort_key,
        ):
            resolved_primary = str(p.resolve())
            shard_paths: list[str] = [resolved_primary]
            display_name = format_local_gguf_display(
                str(p), models_dir=root
            ).menu_label
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

    def _reload_sidecar_from_settings(self) -> None:
        sw = self.workers.get("sidecar_worker") if getattr(self, "workers", None) else None
        if sw is not None and hasattr(sw, "reload_from_settings"):
            sw.reload_from_settings()
        self.cognition_model_changed.emit()

    def _on_advanced_engine_toggled(self, checked: bool) -> None:
        if checked:
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            dlg = PrestigeDialog(
                self.window(),
                "Advanced engine settings",
                "The auxiliary cognition model uses additional CPU RAM while your primary "
                "chat model is loaded. Swapping to a larger model (1.5B+) can reduce "
                "headroom and slow background tasks.\n\n"
                "The bundled Qwen3 1.7B default cannot be deleted — you may only load an "
                "alternate model from models/cognition/.\n\nContinue?",
                is_dark=is_dark,
                tone="danger",
                dialog_width=450,
            )
            if not dlg.exec():
                self.advanced_engine_toggle.blockSignals(True)
                self.advanced_engine_toggle.setChecked(False)
                self.advanced_engine_toggle.blockSignals(False)
                return
        set_advanced_engine_unlocked(bool(checked))
        if hasattr(self, "advanced_engine_panel"):
            self.advanced_engine_panel.setVisible(bool(checked))

    def _refresh_cognition_gguf_list(self) -> None:
        if not hasattr(self, "cognition_gguf_list"):
            return
        self.cognition_gguf_list.clear()
        active = resolve_active_cognition_path()
        try:
            active_norm = str(Path(active).resolve()) if active else ""
        except OSError:
            active_norm = active or ""

        for entry in list_selectable_cognition_models():
            item = QListWidgetItem(entry.display_name)
            item.setData(Qt.ItemDataRole.UserRole, entry.path)
            item.setData(COGNITION_ENTRY_DELETABLE_ROLE, entry.is_deletable)
            self.cognition_gguf_list.addItem(item)
            try:
                if active_norm and str(Path(entry.path).resolve()) == active_norm:
                    self.cognition_gguf_list.setCurrentItem(item)
            except OSError:
                if entry.path == active:
                    self.cognition_gguf_list.setCurrentItem(item)

    def _sync_active_cognition_label(self) -> None:
        if not hasattr(self, "active_cognition_model_lbl"):
            return
        path = resolve_active_cognition_path()
        if not path or not os.path.isfile(path):
            self.active_cognition_model_lbl.setText("— (bundled default missing)")
            return
        base = os.path.basename(path)
        if is_protected_cognition_model(path):
            self.active_cognition_model_lbl.setText(f"{base} (bundled default)")
        else:
            self.active_cognition_model_lbl.setText(f"{base} (custom)")

    def _sync_cognition_chat_format_label(self) -> None:
        if not hasattr(self, "cognition_chat_format_selector"):
            return
        fmt = get_sidecar_chat_format()
        labels = {v: k for k, v in self._cognition_chat_format_items}
        self.cognition_chat_format_selector.setText(
            labels.get(fmt, "Auto (from filename)")
        )

    def _on_cognition_chat_format_changed(self, mode: str) -> None:
        set_sidecar_chat_format(str(mode))
        self._sync_cognition_chat_format_label()
        self._reload_sidecar_from_settings()

    def _apply_selected_cognition_gguf(self) -> None:
        item = self.cognition_gguf_list.currentItem()
        if not item:
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            PrestigeDialog(
                self.window(),
                "No model",
                "Select a cognition model from the list.",
                is_dark=is_dark,
            ).exec()
            return
        path = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if is_protected_cognition_model(path):
            set_sidecar_model_path("")
        else:
            ok, msg = validate_cognition_model_path(path)
            if not ok:
                is_dark = getattr(self.window(), "_is_dark_theme", True)
                PrestigeDialog(
                    self.window(),
                    "Invalid cognition model",
                    msg or "That file cannot be used as the cognition model.",
                    is_dark=is_dark,
                ).exec()
                return
            set_sidecar_model_path(path)
        self._sync_active_cognition_label()
        self._reload_sidecar_from_settings()

    def _reset_cognition_to_default(self) -> None:
        set_sidecar_model_path("")
        self._refresh_cognition_gguf_list()
        self._sync_active_cognition_label()
        self._reload_sidecar_from_settings()

    def _delete_selected_cognition_gguf(self) -> None:
        item = self.cognition_gguf_list.currentItem()
        if not item:
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            PrestigeDialog(
                self.window(),
                "No model",
                "Select a cognition model to delete.",
                is_dark=is_dark,
            ).exec()
            return
        path = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if is_protected_cognition_model(path):
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            PrestigeDialog(
                self.window(),
                "Protected model",
                "The bundled Qwen3 1.7B default cannot be deleted. Use Reset to default "
                "to stop using a custom cognition model.",
                is_dark=is_dark,
            ).exec()
            return
        if not item.data(COGNITION_ENTRY_DELETABLE_ROLE):
            return
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
            "Delete cognition model",
            f'Permanently delete "{name}" from models/cognition/? This cannot be undone.',
            is_dark=is_dark,
        )
        if not dlg.exec():
            return
        try:
            os.remove(path)
        except OSError as e:
            logger.error("Failed to delete cognition GGUF %s: %s", path, e)
            PrestigeDialog(
                self.window(),
                "Delete failed",
                str(e),
                is_dark=is_dark,
            ).exec()
            return
        override = get_sidecar_model_path()
        try:
            was_active = str(Path(override).resolve()) == str(Path(path).resolve())
        except OSError:
            was_active = override == path
        if was_active:
            set_sidecar_model_path("")
            self._reload_sidecar_from_settings()
        self._refresh_cognition_gguf_list()
        self._sync_active_cognition_label()

    def refresh_native_local_library(self) -> None:
        """Call when a .gguf is saved elsewhere (e.g. Model Manager download)."""
        self._sync_models_dir_label()
        self._sync_active_native_model_label()
        self._refresh_local_gguf_list()
        if hasattr(self, "_refresh_cognition_gguf_list"):
            self._refresh_cognition_gguf_list()

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
        self.auto_activator_cb.setToolTip(
            "When enabled, custom trigger phrases can search your Knowledge Base for a single turn, "
            "even if the master RAG switch is off. Add magic words below."
        )
        self.auto_activator_cb.toggled.connect(self.auto_activator_toggle.emit)
        layout.addWidget(self.auto_activator_cb)
        
        # Input Row
        input_row = QHBoxLayout()
        self.trigger_input = QLineEdit()
        self.trigger_input.setPlaceholderText("e.g. 'search my notes for...'")
        self.trigger_input.setToolTip(
            "Type a phrase that should trigger a Knowledge Base search, then press Enter or +."
        )
        self.trigger_input.returnPressed.connect(self._on_add_trigger)
        
        self.trigger_add_btn = QPushButton()
        self.trigger_add_btn.setFixedSize(36, 36)
        self.trigger_add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.trigger_add_btn.setToolTip("Add trigger phrase")
        
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
            del_btn.setToolTip("Remove this trigger phrase")
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

    def _refresh_llm_rag_triggers(self) -> None:
        if self.llm_worker is not None and hasattr(self.llm_worker, "refresh_rag_triggers"):
            self.llm_worker.refresh_rag_triggers()

    def _on_add_trigger(self):
        text = self.trigger_input.text().strip()
        if text:
            success = self.db.add_rag_trigger(text)
            if success:
                self.trigger_input.clear()
                self._refresh_trigger_list()
                self._refresh_llm_rag_triggers()

    def _on_delete_trigger(self, phrase):
        self.db.remove_rag_trigger(phrase)
        self._refresh_trigger_list()
        self._refresh_llm_rag_triggers()

    def _sync_memory_promotion_controls_for_enrichment(self) -> None:
        """Enable promotion controls only when enrichment is on; worker uses effective AND."""
        enrichment_on = get_enable_memory_enrichment()
        for widget in (
            getattr(self, "memory_promotion_toggle", None),
            getattr(self, "mem_promotion_label", None),
            getattr(self, "memory_promotion_preset_selector", None),
            getattr(self, "_promo_preset_lbl", None),
        ):
            if widget is not None:
                widget.setEnabled(enrichment_on)
        if hasattr(self, "memory_promotion_toggle"):
            self.memory_promotion_changed.emit(
                enrichment_on and get_enable_memory_promotion()
            )

    def _on_memory_enrichment_toggled(self, checked: bool):
        set_enable_memory_enrichment(checked)
        self.memory_enrichment_changed.emit(checked)
        self._sync_memory_promotion_controls_for_enrichment()

    def _confirm_memory_promotion_enable(self) -> bool:
        """One-time PrestigeDialog before first enable; returns True if user confirms."""
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        dlg = PrestigeDialog(
            self.window(),
            "Enable memory promotion?",
            "When this is on, Qube may upgrade facts you rely on often from "
            "context or knowledge into long-term preferences — the kind of "
            "thing Qube should remember about you without being asked each time.\n\n"
            "Preferences are weighted more strongly in future recall. Runs quietly "
            "in the background (about every 6 hours). Qube never deletes memories "
            "on its own.\n\n"
            "Review promoted rows in Memory Manager. Use the Conservative preset "
            "below if you want stricter gates before anything is upgraded.\n\n"
            "Requires Memory Enrichment & Reflection to be enabled.",
            is_dark=is_dark,
            dialog_width=480,
        )
        return bool(dlg.exec())

    def _on_memory_promotion_toggled(self, checked: bool):
        if checked and not get_memory_promotion_acknowledged():
            if not self._confirm_memory_promotion_enable():
                self.memory_promotion_toggle.blockSignals(True)
                self.memory_promotion_toggle.setChecked(False)
                self.memory_promotion_toggle.blockSignals(False)
                return
            set_memory_promotion_acknowledged(True)
        set_enable_memory_promotion(checked)
        self.memory_promotion_changed.emit(
            get_enable_memory_enrichment() and checked
        )

    def _build_profile_units_menu(self) -> None:
        if not hasattr(self, "profile_units_selector"):
            return
        menu = QMenu(self)
        options = [
            ("", "Use inferred units"),
            ("metric", "Metric"),
            ("imperial", "Imperial"),
        ]

        def _pick(value: str, label: str) -> None:
            set_profile_units(value or None)
            self.profile_units_selector.setText(label)

        for value, label in options:
            act = menu.addAction(label)
            act.triggered.connect(lambda _checked=False, v=value, l=label: _pick(v, l))
        self.profile_units_selector.setMenu(menu)

    def _sync_profile_units_selector(self) -> None:
        if not hasattr(self, "profile_units_selector"):
            return
        units = get_profile_units()
        labels = {"metric": "Metric", "imperial": "Imperial"}
        self.profile_units_selector.setText(labels.get(units or "", "Use inferred units"))

    def _build_memory_promotion_preset_menu(self) -> None:
        if not hasattr(self, "memory_promotion_preset_selector"):
            return
        menu = QMenu(self)
        labels = {
            "conservative": "Conservative",
            "standard": "Standard",
            "aggressive": "Aggressive",
        }
        current = get_memory_promotion_preset()

        def _pick(key: str, label: str) -> None:
            set_memory_promotion_preset(key)
            self.memory_promotion_preset_selector.setText(label)

        for key, label in labels.items():
            act = menu.addAction(label)
            act.triggered.connect(lambda _checked=False, k=key, l=label: _pick(k, l))
        self.memory_promotion_preset_selector.setMenu(menu)
        self.memory_promotion_preset_selector.setText(labels.get(current, "Standard"))

    def _on_memory_consolidation_toggled(self, checked: bool):
        set_enable_memory_consolidation(checked)
        self.memory_consolidation_changed.emit(checked)

    def _on_chat_personality_toggled(self, checked: bool) -> None:
        set_enable_chat_personality_nudge(checked)

    def _on_notifications_dnd_toggled(self, checked: bool) -> None:
        from core.app_settings import set_notifications_dnd

        set_notifications_dnd(checked)
        win = self.window()
        if win is not None and hasattr(win, "tray_controller") and win.tray_controller is not None:
            win.tray_controller.sync_dnd_toggle()
        if win is not None and hasattr(win, "_presence_service"):
            win._presence_service.set_dnd(checked)
            if hasattr(win, "_companion_controller") and win._companion_controller is not None:
                win._companion_controller.on_settings_changed()

    def _on_companion_enabled_toggled(self, checked: bool) -> None:
        win = self.window()
        if win is not None and hasattr(win, "_companion_controller") and win._companion_controller is not None:
            win._companion_controller.set_user_enabled(checked)
        else:
            from core.app_settings import set_companion_enabled

            set_companion_enabled(checked)
            self._on_companion_setting_changed()
        self._sync_companion_verbal_controls_enabled()
        if win is not None and hasattr(win, "tray_controller") and win.tray_controller is not None:
            win.tray_controller.sync_companion_toggle()

    def _sync_companion_verbal_controls_enabled(self) -> None:
        companion_on = (
            hasattr(self, "companion_enabled_cb")
            and self.companion_enabled_cb.isChecked()
        )
        for name in (
            "companion_verbal_enabled_cb",
            "companion_cognition_v2_cb",
            "companion_expression_freedom_selector",
            "companion_verbal_prompt",
            "companion_verbal_trait_selector",
            "companion_verbal_frequency_selector",
            "companion_verbal_react_ingest_cb",
            "companion_verbal_react_download_cb",
            "companion_verbal_test_btn",
        ):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(companion_on)
        result_lbl = getattr(self, "companion_verbal_test_result", None)
        if result_lbl is not None:
            result_lbl.setEnabled(True)

    def _build_companion_verbal_trait_menu(self) -> None:
        if not hasattr(self, "companion_verbal_trait_selector"):
            return
        from core import app_settings as _cs
        from core.companion_verbal_traits import (
            CompanionVerbalTraitPreset,
            TRAIT_LABELS,
            normalize_companion_verbal_trait,
        )

        menu = QMenu(self)
        current = normalize_companion_verbal_trait(_cs.get_companion_verbal_trait_preset())

        def _pick(preset: CompanionVerbalTraitPreset) -> None:
            _cs.set_companion_verbal_trait_preset(preset.value)
            self.companion_verbal_trait_selector.setText(TRAIT_LABELS[preset])
            self._on_companion_verbal_setting_changed()

        trait_tips = {
            CompanionVerbalTraitPreset.NEUTRAL: "Calm, brief companion lines.",
            CompanionVerbalTraitPreset.WARM: "Gently encouraging tone.",
            CompanionVerbalTraitPreset.WITTY: "Light humor; never distracting or insulting.",
            CompanionVerbalTraitPreset.DRY: "Understated, deadpan humor.",
            CompanionVerbalTraitPreset.SARCASTIC: "Mild sarcasm; still friendly.",
        }
        for preset in CompanionVerbalTraitPreset:
            act = menu.addAction(TRAIT_LABELS[preset])
            act.setToolTip(trait_tips.get(preset, ""))
            act.triggered.connect(lambda _checked=False, p=preset: _pick(p))
        self.companion_verbal_trait_selector.setMenu(menu)
        self.companion_verbal_trait_selector.setText(TRAIT_LABELS[current])

    def _build_companion_verbal_frequency_menu(self) -> None:
        if not hasattr(self, "companion_verbal_frequency_selector"):
            return
        from core import app_settings as _cs
        from core.companion_verbal_policy import (
            CompanionVerbalFrequency,
            frequency_idle_label,
            normalize_companion_verbal_frequency,
        )

        labels = {
            CompanionVerbalFrequency.RARE: "Rare",
            CompanionVerbalFrequency.NORMAL: "Normal",
            CompanionVerbalFrequency.CHATTY: "Chatty",
        }
        menu = QMenu(self)
        current = normalize_companion_verbal_frequency(_cs.get_companion_verbal_frequency())

        def _pick(freq: CompanionVerbalFrequency) -> None:
            _cs.set_companion_verbal_frequency(freq.value)
            self.companion_verbal_frequency_selector.setText(labels[freq])
            self._on_companion_verbal_setting_changed()

        for freq in CompanionVerbalFrequency:
            act = menu.addAction(labels[freq])
            act.setToolTip(frequency_idle_label(freq))
            act.triggered.connect(lambda _checked=False, f=freq: _pick(f))
        self.companion_verbal_frequency_selector.setMenu(menu)
        self.companion_verbal_frequency_selector.setText(labels[current])

    def _build_companion_expression_freedom_menu(self) -> None:
        if not hasattr(self, "companion_expression_freedom_selector"):
            return
        from core import app_settings as _cs

        labels = {
            "conservative": "Conservative",
            "balanced": "Balanced",
            "expressive": "Expressive",
        }
        freedom_tips = {
            "conservative": (
                "Curated message library only — templates at most. "
                "No sidecar rephrasing or full generation."
            ),
            "balanced": (
                "Expression depth follows your auxiliary cognition model "
                "(small models: templates; larger models: optional rephrasing)."
            ),
            "expressive": (
                "Allows the richest local lines plus sidecar rephrasing or "
                "full generation when the auxiliary model supports it."
            ),
        }
        menu = QMenu(self)
        current = _cs.get_companion_expression_freedom()

        def _pick(mode: str) -> None:
            _cs.set_companion_expression_freedom(mode)
            self.companion_expression_freedom_selector.setText(labels[mode])
            self._on_companion_verbal_setting_changed()

        for mode in ("conservative", "balanced", "expressive"):
            act = menu.addAction(labels[mode])
            act.setToolTip(freedom_tips[mode])
            act.triggered.connect(lambda _checked=False, m=mode: _pick(m))
        self.companion_expression_freedom_selector.setMenu(menu)
        self.companion_expression_freedom_selector.setText(labels.get(current, "Balanced"))

    def _on_companion_verbal_prompt_changed(self) -> None:
        from core.app_settings import set_companion_verbal_system_prompt

        if not hasattr(self, "companion_verbal_prompt"):
            return
        set_companion_verbal_system_prompt(self.companion_verbal_prompt.toPlainText())
        self._on_companion_verbal_setting_changed()

    def _on_companion_verbal_setting_changed(self, *_args) -> None:
        from core import app_settings as _cs

        if hasattr(self, "companion_verbal_enabled_cb"):
            _cs.set_companion_verbal_enabled(self.companion_verbal_enabled_cb.isChecked())
        if hasattr(self, "companion_cognition_v2_cb"):
            _cs.set_companion_cognition_v2_enabled(self.companion_cognition_v2_cb.isChecked())
        if hasattr(self, "companion_verbal_react_ingest_cb"):
            _cs.set_companion_verbal_react_ingest(
                self.companion_verbal_react_ingest_cb.isChecked()
            )
        if hasattr(self, "companion_verbal_react_download_cb"):
            _cs.set_companion_verbal_react_download(
                self.companion_verbal_react_download_cb.isChecked()
            )
        win = self.window()
        if win is not None and hasattr(win, "_companion_controller") and win._companion_controller is not None:
            win._companion_controller.on_settings_changed()

    def _on_companion_verbal_test_clicked(self) -> None:
        from core import app_settings as _cs
        from ui.companion.companion_verbal_test_worker import CompanionVerbalTestWorker

        if (
            self._companion_verbal_test_worker is not None
            and self._companion_verbal_test_worker.isRunning()
        ):
            return

        if hasattr(self, "companion_verbal_prompt"):
            _cs.set_companion_verbal_system_prompt(self.companion_verbal_prompt.toPlainText())

        if _cs.get_companion_cognition_v2_enabled():
            win = self.window()
            sched = None
            if win is not None and hasattr(win, "_companion_controller"):
                ctrl = win._companion_controller
                if ctrl is not None:
                    sched = getattr(ctrl, "_verbal_scheduler", None)
            if sched is not None:
                line, _kind = sched.process_test_preview()
                if line:
                    self.companion_verbal_test_result.setText(f'Preview: "{line}"')
                    if hasattr(self, "companion_preview"):
                        self.companion_preview.show_sample_caption(line, ttl_sec=12.0)
                    win = self.window()
                    controller = getattr(win, "_companion_controller", None) if win is not None else None
                    if controller is not None and getattr(controller, "is_visible_for_policy", False):
                        controller.window.show_banter_caption(line, ttl_sec=12.0)
                    return
                self.companion_verbal_test_result.setText(
                    "Cognition v2 returned no line — try a different personality."
                )
                return

        from core.sidecar_llm import sidecar_model_available

        sidecar = self.workers.get("sidecar")
        if sidecar is None or not sidecar_model_available():
            PrestigeDialog(
                self,
                "Cognition model unavailable",
                "The auxiliary cognition model file is missing. "
                "Ensure the bundled sidecar model is present, or select one under "
                "Advanced engine settings.",
                is_dark=getattr(self.window(), "_is_dark_theme", True),
            ).exec()
            return

        if hasattr(self, "companion_verbal_prompt"):
            _cs.set_companion_verbal_system_prompt(self.companion_verbal_prompt.toPlainText())

        self.companion_verbal_test_btn.setEnabled(False)
        self.companion_verbal_test_result.setText("Generating preview…")

        payload = {
            "trigger": "test",
            "trait_preset": _cs.get_companion_verbal_trait_preset(),
            "user_system_prompt": _cs.get_companion_verbal_system_prompt(),
        }
        worker = CompanionVerbalTestWorker(sidecar, payload, self)
        self._companion_verbal_test_worker = worker
        worker.finished.connect(self._on_companion_verbal_test_finished)
        worker.start()

    def _on_companion_verbal_test_finished(self, result: object) -> None:
        from core.sidecar_types import SidecarResult

        self._sync_companion_verbal_controls_enabled()
        if not isinstance(result, SidecarResult):
            self.companion_verbal_test_result.setText("Preview failed (unexpected response).")
            return

        if not result.ok:
            if result.error == "model_unavailable":
                msg = "Cognition model is not available yet. Wait a few seconds after launch and try again."
            elif result.error == "timeout":
                msg = "Preview timed out — the sidecar queue may be busy. Try again shortly."
            elif result.error == "skip":
                msg = "Model returned no line for this configuration. Try a different personality or prompt."
            elif result.error == "parse_fail" and (result.text or "").strip():
                snippet = (result.text or "").strip()
                if len(snippet) > 80:
                    snippet = snippet[:77] + "…"
                msg = (
                    "The cognition model returned tutorial-style text instead of a short "
                    "JSON caption. Try again, switch personality (e.g. Witty), or add a "
                    "custom prompt like 'one short casual sentence only'. "
                    f'Raw: "{snippet}"'
                )
            else:
                msg = f"Preview failed ({result.error or 'unknown'})."
            self.companion_verbal_test_result.setText(msg)
            if hasattr(self, "companion_preview"):
                self.companion_preview._clear_sample_caption()
            return

        line = (result.text or "").strip()
        self.companion_verbal_test_result.setText(f'Preview: "{line}"')

        if hasattr(self, "companion_preview"):
            self.companion_preview.show_sample_caption(line, ttl_sec=12.0)

        win = self.window()
        controller = getattr(win, "_companion_controller", None) if win is not None else None
        if controller is not None and getattr(controller, "is_visible_for_policy", False):
            controller.window.show_banter_caption(line, ttl_sec=12.0)
        elif controller is not None:
            hint = (
                ' Preview is shown above the orb sample. To see it on the desktop orb, '
                "enable the companion and either hide the main window to the tray or turn on "
                '"Show companion while main window is open".'
            )
            self.companion_verbal_test_result.setText(
                f'Preview: "{line}"' + hint
            )

    def _on_companion_setting_changed(self, *_args) -> None:
        from core import app_settings as _cs

        if hasattr(self, "companion_tray_hidden_cb"):
            _cs.set_companion_show_when_tray_hidden(self.companion_tray_hidden_cb.isChecked())
        if hasattr(self, "companion_while_open_cb"):
            _cs.set_companion_show_while_window_open(self.companion_while_open_cb.isChecked())
        if hasattr(self, "companion_auto_hide_cb"):
            _cs.set_companion_auto_hide_idle(self.companion_auto_hide_cb.isChecked())
        if hasattr(self, "companion_caption_cb"):
            _cs.set_companion_show_caption(self.companion_caption_cb.isChecked())
        if hasattr(self, "companion_fullscreen_cb"):
            _cs.set_companion_suppress_on_fullscreen(self.companion_fullscreen_cb.isChecked())
        if hasattr(self, "companion_wayland_cb"):
            _cs.set_companion_try_on_wayland(self.companion_wayland_cb.isChecked())
        if hasattr(self, "companion_dock_cb"):
            _cs.set_companion_dock_mode(self.companion_dock_cb.isChecked())

        win = self.window()
        if win is not None and hasattr(win, "_companion_controller") and win._companion_controller is not None:
            win._companion_controller.on_settings_changed()

    def _on_companion_persona_toggled(self, button, checked: bool) -> None:
        if not checked:
            if not any(cb.isChecked() for cb in self.companion_persona_cbs.values()):
                button.blockSignals(True)
                button.setChecked(True)
                button.blockSignals(False)
            return
        from core import app_settings as _cs
        from core.companion_personas import normalize_companion_persona

        persona_id = normalize_companion_persona(button.property("companion_persona_id"))
        _cs.set_companion_persona(persona_id.value)
        if hasattr(self, "companion_preview"):
            self.companion_preview.set_persona(persona_id)
        win = self.window()
        if win is not None and hasattr(win, "_companion_controller") and win._companion_controller is not None:
            win._companion_controller.on_settings_changed()

    def _on_companion_idle_color_toggled(self, button, checked: bool) -> None:
        if not checked:
            if not any(cb.isChecked() for cb in self.companion_idle_color_cbs.values()):
                button.blockSignals(True)
                button.setChecked(True)
                button.blockSignals(False)
            return
        from core import app_settings as _cs
        from core.companion_idle_color import normalize_companion_idle_color

        color_id = normalize_companion_idle_color(button.property("companion_idle_color_id"))
        _cs.set_companion_idle_color(color_id.value)
        if hasattr(self, "companion_preview"):
            self.companion_preview.update()
        win = self.window()
        if win is not None and hasattr(win, "_companion_controller") and win._companion_controller is not None:
            win._companion_controller.on_settings_changed()

    def _sync_companion_demo_selector_label(self, key: str = "idle") -> None:
        if not hasattr(self, "companion_demo_selector"):
            return
        label = next(
            (lbl for lbl, data in getattr(self, "_companion_demo_items", []) if data == key),
            "Idle",
        )
        self.companion_demo_selector.setText(label)
        self.companion_demo_selector.update()

    def _on_companion_demo_state_selected(self, key: str) -> None:
        self._sync_companion_demo_selector_label(key)
        if not hasattr(self, "companion_preview"):
            return
        from core.assistant_activity import AssistantActivity

        mapping = {
            "idle": AssistantActivity.IDLE_LISTEN,
            "working": AssistantActivity.WORKING,
            "writing": AssistantActivity.WORKING,
            "capturing": AssistantActivity.CAPTURING,
            "speaking": AssistantActivity.SPEAKING,
        }
        self.companion_preview.set_demo_activity(mapping.get(str(key), AssistantActivity.IDLE_LISTEN))

    def _clear_notification_history(self) -> None:
        win = self.window()
        if win is not None and hasattr(win, "notification_service"):
            win.notification_service.history.clear()
            if hasattr(win, "tray_controller") and win.tray_controller is not None:
                win.tray_controller.update_recent_notifications([])
        self.settings_file_status_lbl.setText("Notification history cleared.")

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
            getattr(self, "companion_demo_selector", None),
            getattr(self, "companion_verbal_trait_selector", None),
            getattr(self, "companion_verbal_frequency_selector", None),
            getattr(self, "companion_expression_freedom_selector", None),
            getattr(self, "memory_promotion_preset_selector", None),
        ]
        for btn in buttons:
            if btn is None:
                continue
            if isinstance(btn, SelectorButton):
                btn.apply_theme(is_dark)
            if btn.menu():
                self._apply_menu_theme(btn.menu(), is_dark)

        info_btn = getattr(self, "advanced_engine_info_btn", None)
        if info_btn is not None:
            info_color = "#94a3b8" if is_dark else "#64748b"
            info_btn.setIcon(qta.icon("fa5s.info-circle", color=info_color))

        # Update Section Header Icons
        icon_color = "#8b5cf6" if is_dark else "#4c4f69" 
        
        for icon_lbl in [
            getattr(self, 'audio_icon_label', None),
            getattr(self, 'ai_icon_label', None),
            getattr(self, 'native_lib_icon_label', None),
            getattr(self, 'perf_icon_label', None),
            getattr(self, 'rag_icon_label', None),
            getattr(self, 'json_settings_icon_label', None),
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

        if self._settings_json_dialog is not None:
            self._settings_json_dialog.refresh_theme(is_dark)

        if hasattr(self, "companion_preview"):
            self.companion_preview.apply_theme(is_dark)

    def _handle_selection(self, button, label, data, callback):
        button.setText(label)
        if hasattr(button, "update"):
            button.update()
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
        elif "JSON SETTINGS" in title_text:
            self.json_settings_icon_label = icon_label
        
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
            ("Internal Engine (native)", "internal"),
            ("External Server (localhost)", "external"),
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

    def _setup_settings_file_watcher(self) -> None:
        self._settings_reload_timer = QTimer(self)
        self._settings_reload_timer.setSingleShot(True)
        self._settings_reload_timer.setInterval(400)
        self._settings_reload_timer.timeout.connect(self._reload_settings_from_disk)
        self._settings_watcher = QFileSystemWatcher(self)
        self._settings_watcher.fileChanged.connect(self._on_settings_file_changed)

    def _ensure_settings_file_watched(self) -> None:
        path = str(default_user_settings_path())
        watched = set(self._settings_watcher.files())
        if path not in watched:
            if not default_user_settings_path().is_file():
                get_settings_store().ensure_user_settings_file()
            self._settings_watcher.addPath(path)
        parent = str(default_user_settings_path().parent)
        if parent not in self._settings_watcher.directories():
            self._settings_watcher.addPath(parent)

    def _on_settings_file_changed(self, _path: str) -> None:
        if self._settings_json_dialog is not None and self._settings_json_dialog.isVisible():
            return
        self._settings_reload_timer.start()

    def _on_open_settings_json_clicked(self) -> None:
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        if self._settings_json_dialog is None:
            self._settings_json_dialog = SettingsJsonEditorDialog(self, is_dark=is_dark)
            self._settings_json_dialog.settings_applied.connect(
                self._on_settings_editor_applied
            )
        else:
            self._settings_json_dialog.refresh_theme(is_dark)
        self._settings_json_dialog.load_from_disk()
        self._settings_json_dialog.show()
        self._settings_json_dialog.raise_()
        self._settings_json_dialog.activateWindow()
        self.settings_file_status_lbl.setText("Editing settings.json in the built-in editor.")

    def _on_settings_editor_applied(self, changed: set) -> None:
        if not changed:
            return
        self._sync_ui_from_persisted_settings()
        self.settings_file_status_lbl.setText(
            f"Applied {len(changed)} setting(s) from settings.json."
        )
        self.external_settings_reloaded.emit(changed)

    def _reload_settings_from_disk(self) -> None:
        store = get_settings_store()
        result = store.reload_if_disk_changed()
        if result is None:
            return
        if not result.ok:
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            PrestigeDialog(
                self,
                "Invalid settings.json",
                result.parse_error or "The file could not be parsed.",
                is_dark=is_dark,
            ).exec()
            self.settings_file_status_lbl.setText("settings.json has errors — fix JSON and save again.")
            return
        if result.skipped_keys:
            skipped = ", ".join(result.skipped_keys[:5])
            if len(result.skipped_keys) > 5:
                skipped += ", …"
            logger.info("Ignored unknown settings keys: %s", skipped)
        if not result.changed_keys:
            return
        self._sync_ui_from_persisted_settings()
        self.settings_file_status_lbl.setText(
            f"Reloaded {len(result.changed_keys)} setting(s) from settings.json."
        )
        self.external_settings_reloaded.emit(set(result.changed_keys))

    def _sync_ui_from_persisted_settings(self) -> None:
        engine_modes = [
            ("Internal Engine (native)", "internal"),
            ("External Server (localhost)", "external"),
        ]
        em = get_engine_mode()
        engine_label = next((lbl for lbl, m in engine_modes if m == em), engine_modes[0][0])
        self.engine_selector.blockSignals(True)
        self.engine_selector.setText(engine_label)
        self.engine_selector.blockSignals(False)

        self.memory_enrichment_toggle.blockSignals(True)
        self.memory_enrichment_toggle.setChecked(get_enable_memory_enrichment())
        self.memory_enrichment_toggle.blockSignals(False)
        if hasattr(self, "chat_personality_toggle"):
            self.chat_personality_toggle.blockSignals(True)
            self.chat_personality_toggle.setChecked(get_enable_chat_personality_nudge())
            self.chat_personality_toggle.blockSignals(False)
        if hasattr(self, "memory_promotion_toggle"):
            self.memory_promotion_toggle.blockSignals(True)
            self.memory_promotion_toggle.setChecked(get_enable_memory_promotion())
            self.memory_promotion_toggle.blockSignals(False)
        if hasattr(self, "memory_consolidation_toggle"):
            self.memory_consolidation_toggle.blockSignals(True)
            self.memory_consolidation_toggle.setChecked(get_enable_memory_consolidation())
            self.memory_consolidation_toggle.blockSignals(False)
        if hasattr(self, "memory_promotion_preset_selector"):
            labels = {
                "conservative": "Conservative",
                "standard": "Standard",
                "aggressive": "Aggressive",
            }
            preset = get_memory_promotion_preset()
            self.memory_promotion_preset_selector.setText(labels.get(preset, "Standard"))
        if hasattr(self, "memory_promotion_toggle"):
            self._sync_memory_promotion_controls_for_enrichment()
        if hasattr(self, "profile_units_selector"):
            self._sync_profile_units_selector()

        if hasattr(self, "notifications_enabled_cb"):
            from core import app_settings as _ns

            self.notifications_enabled_cb.blockSignals(True)
            self.notifications_enabled_cb.setChecked(_ns.get_notifications_enabled())
            self.notifications_enabled_cb.blockSignals(False)
            self.notifications_dnd_cb.blockSignals(True)
            self.notifications_dnd_cb.setChecked(_ns.get_notifications_dnd())
            self.notifications_dnd_cb.blockSignals(False)
            self.notifications_suppress_focus_cb.blockSignals(True)
            self.notifications_suppress_focus_cb.setChecked(_ns.get_notifications_suppress_when_focused())
            self.notifications_suppress_focus_cb.blockSignals(False)
            self.notifications_os_hidden_cb.blockSignals(True)
            self.notifications_os_hidden_cb.setChecked(_ns.get_notifications_os_when_hidden())
            self.notifications_os_hidden_cb.blockSignals(False)
            self.notifications_sound_cb.blockSignals(True)
            self.notifications_sound_cb.setChecked(_ns.get_notifications_sound_enabled())
            self.notifications_sound_cb.blockSignals(False)
            self.notifications_preview_cb.blockSignals(True)
            self.notifications_preview_cb.setChecked(_ns.get_notifications_show_preview())
            self.notifications_preview_cb.blockSignals(False)
            self.notifications_memory_cb.blockSignals(True)
            self.notifications_memory_cb.setChecked(_ns.get_notifications_category_memory())
            self.notifications_memory_cb.blockSignals(False)

        if hasattr(self, "companion_enabled_cb"):
            from core import app_settings as _cs

            self.companion_enabled_cb.blockSignals(True)
            self.companion_enabled_cb.setChecked(_cs.get_companion_enabled())
            self.companion_enabled_cb.blockSignals(False)
            win = self.window()
            if win is not None and hasattr(win, "tray_controller") and win.tray_controller is not None:
                win.tray_controller.sync_companion_toggle()

        if hasattr(self, "companion_persona_cbs"):
            from core import app_settings as _cs

            current = _cs.get_companion_persona()
            for persona_id, cb in self.companion_persona_cbs.items():
                cb.blockSignals(True)
                cb.setChecked(persona_id == current)
                cb.blockSignals(False)
            if hasattr(self, "companion_preview"):
                self.companion_preview.set_persona(current)

        if hasattr(self, "companion_idle_color_cbs"):
            from core import app_settings as _cs

            current_idle = _cs.get_companion_idle_color()
            for color_id, cb in self.companion_idle_color_cbs.items():
                cb.blockSignals(True)
                cb.setChecked(color_id == current_idle)
                cb.blockSignals(False)
            if hasattr(self, "companion_preview"):
                self.companion_preview.update()

        if hasattr(self, "advanced_engine_toggle"):
            self.advanced_engine_toggle.blockSignals(True)
            self.advanced_engine_toggle.setChecked(get_advanced_engine_unlocked())
            self.advanced_engine_toggle.blockSignals(False)
            if hasattr(self, "advanced_engine_panel"):
                self.advanced_engine_panel.setVisible(get_advanced_engine_unlocked())

        self.auto_load_last_model_cb.blockSignals(True)
        checked = get_auto_load_last_model_on_startup()
        self.auto_load_last_model_cb.setChecked(checked)
        self.auto_load_last_model_cb.blockSignals(False)
        self.auto_load_last_model_changed.emit(checked)

        self.model_manager_hardware_suggestions_cb.blockSignals(True)
        self.model_manager_hardware_suggestions_cb.setChecked(
            get_model_manager_hardware_suggestions()
        )
        self.model_manager_hardware_suggestions_cb.blockSignals(False)

        gpu_val = get_internal_n_gpu_layers()
        self.gpu_layers_slider.blockSignals(True)
        self.gpu_layers_slider.setValue(gpu_val)
        self.gpu_layers_slider.blockSignals(False)
        self.gpu_layers_value_lbl.setText(str(gpu_val))

        cpu_val = get_internal_n_threads()
        self.cpu_threads_slider.blockSignals(True)
        self.cpu_threads_slider.setValue(cpu_val)
        self.cpu_threads_slider.blockSignals(False)
        self.cpu_threads_value_lbl.setText(str(cpu_val))

        preferred = get_internal_native_chat_format()
        label = next(
            (lbl for lbl, mode in self._native_chat_format_items if mode == preferred),
            self._native_chat_format_items[0][0],
        )
        self.native_chat_format_selector.blockSignals(True)
        self.native_chat_format_selector.setText(label)
        self.native_chat_format_selector.blockSignals(False)
        self._sync_native_chat_template_label()

        self._sync_models_dir_label()
        self._sync_active_native_model_label()
        self._refresh_local_gguf_list()
        self._sync_ai_provider_enabled_for_inference(em)

        saved_input = get_audio_input_device_index()
        if saved_input is not None:
            mics = get_input_devices()
            for idx, name in mics:
                if idx == saved_input:
                    self.mic_selector.setText(name)
                    if self.audio_worker:
                        self.audio_worker.set_input_device(idx)
                    break

        saved_output = get_audio_output_device_index()
        if saved_output is not None:
            outputs = get_output_devices()
            for idx, name in outputs:
                if idx == saved_output:
                    self.device_selector.setText(name)
                    if self.tts_worker:
                        self.tts_worker.set_device(idx)
                    break

        if self.audio_worker:
            self._sync_wakeword_catalog(trigger="settings reload")