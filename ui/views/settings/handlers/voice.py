"""Settings handler mixin: VoiceHandlersMixin."""

from __future__ import annotations

# Shared imports from settings shell (handlers use ``self`` as SettingsView).
import os
import logging
from pathlib import Path
import qtawesome as qta
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QFrame, QPushButton,
    QLabel, QCheckBox, QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox, QScrollArea, QProgressBar,
    QToolButton,
    QStyledItemDelegate, QListView, QMenu, QListWidget, QListWidgetItem, QSlider,
    QButtonGroup, QPlainTextEdit, QGraphicsOpacityEffect, QStackedWidget, QSizePolicy,
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QTimer, QFileSystemWatcher, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFontMetrics, QResizeEvent, QShowEvent
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
    get_llm_output_token_limit,
    get_llm_output_token_limit_enabled,
    get_llm_chat_history_messages,
    get_llm_top_k,
    get_llm_repeat_penalty,
    get_llm_presence_penalty,
    get_llm_top_p,
    get_llm_min_p,
    get_mcp_rag_auto_activator_enabled,
)
from core.output_token_budget import describe_output_token_budget
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
from ui.components.sidebar_list_qss import apply_sidebar_row_title_colors
from ui.sidebar_dimensions import LEFT_NAV_LIST_SIDEBAR_WIDTH
from ui.views.settings.controls import (
    NoScrollComboBox,
    NoScrollDoubleSpinBox,
    NoScrollSlider,
    NoScrollSpinBox,
)
from ui.views.settings.registry import SETTINGS_SECTIONS, resolve_section_id
from ui.views.settings.widgets import update_ai_status_strip
from ui.views.settings.sections import (
    advanced,
    ai_models,
    desktop_companion,
    memory_knowledge,
    notifications,
    voice_audio,
)
logger = logging.getLogger("Qube.UI.Settings")
LOCAL_GGUF_SHARD_PATHS_ROLE = int(Qt.ItemDataRole.UserRole) + 1
COGNITION_ENTRY_DELETABLE_ROLE = int(Qt.ItemDataRole.UserRole) + 2
_SETTINGS_STATUS_BASE_HOLD_MS = 1800
_SETTINGS_STATUS_MS_PER_CHAR = 75
_SETTINGS_STATUS_MIN_HOLD_MS = 2500
_SETTINGS_STATUS_MAX_HOLD_MS = 8000
_SETTINGS_STATUS_FADE_MS = 500
_SECTION_BUILDERS = {
    "voice.audio": voice_audio.build_section,
    "ai.models": ai_models.build_section,
    "memory.knowledge": memory_knowledge.build_section,
    "companion.desktop": desktop_companion.build_section,
    "notifications": notifications.build_section,
    "advanced": advanced.build_section,
}


class VoiceHandlersMixin:
    """Behavior extracted from SettingsView."""

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

    def _on_wakeword_selection_changed(self, display_name: str) -> None:
        if not self.audio_worker:
            return
        self._wakeword_selected_label = str(display_name)
        self.audio_worker.set_wakeword(display_name)
        if self._wakeword_testbed_dialog is not None:
            self._wakeword_testbed_dialog.on_wakeword_selection_changed()

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
