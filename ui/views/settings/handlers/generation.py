"""Settings handler mixin: GenerationMixin."""

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
from ui.views.settings.sections import (
    advanced,
    ai_models,
    desktop_companion,
    help,
    knowledge,
    memory,
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
    "memory": memory.build_section,
    "knowledge": knowledge.build_section,
    "companion.desktop": desktop_companion.build_section,
    "notifications": notifications.build_section,
    "help": help.build_section,
    "advanced": advanced.build_section,
}


class GenerationMixin:
    """Behavior extracted from SettingsView."""

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

    def _refresh_output_token_limit_hint(self) -> None:
        hint = getattr(self, "llm_output_limit_hint", None)
        if hint is None:
            return
        ctx_spin = getattr(self, "llm_ctx_spin", None)
        ctx = int(ctx_spin.value()) if ctx_spin is not None else get_llm_context_limit()
        limit_cb = getattr(self, "llm_output_limit_cb", None)
        limit_enabled = (
            bool(limit_cb.isChecked())
            if limit_cb is not None
            else get_llm_output_token_limit_enabled()
        )
        limit_spin = getattr(self, "llm_output_limit_spin", None)
        user_limit = (
            int(limit_spin.value())
            if limit_spin is not None
            else get_llm_output_token_limit()
        )
        history_spin = getattr(self, "llm_history_spin", None)
        chat_history = (
            int(history_spin.value())
            if history_spin is not None
            else get_llm_chat_history_messages()
        )
        hint.setText(
            describe_output_token_budget(
                context_window=ctx,
                limit_enabled=limit_enabled,
                user_limit=user_limit,
                chat_history_messages=chat_history,
            )
        )

    def _sync_output_limit_controls(self) -> None:
        enabled = bool(
            getattr(self, "llm_output_limit_cb", None).isChecked()
            if hasattr(self, "llm_output_limit_cb")
            else True
        )
        spin = getattr(self, "llm_output_limit_spin", None)
        if spin is not None:
            spin.setEnabled(enabled)
        self._refresh_output_token_limit_hint()

    def _wire_llm_generation_settings(self) -> None:
        llm = self.llm_worker
        if llm is None:
            return
        self.llm_temp_spin.valueChanged.connect(llm.set_temperature)
        self.llm_ctx_spin.valueChanged.connect(llm.set_context_window)
        self.llm_ctx_spin.valueChanged.connect(
            lambda _v: self._refresh_output_token_limit_hint()
        )
        self.llm_output_limit_cb.toggled.connect(llm.set_output_token_limit_enabled)
        self.llm_output_limit_cb.toggled.connect(self._sync_output_limit_controls)
        self.llm_output_limit_spin.valueChanged.connect(llm.set_output_token_limit)
        self.llm_output_limit_spin.valueChanged.connect(
            lambda _v: self._refresh_output_token_limit_hint()
        )
        self._sync_output_limit_controls()
        self.llm_history_spin.valueChanged.connect(llm.set_max_history_messages)
        self.llm_top_k_spin.valueChanged.connect(llm.set_top_k)
        self.llm_repeat_penalty_spin.valueChanged.connect(llm.set_repeat_penalty)
        self.llm_presence_penalty_spin.valueChanged.connect(llm.set_presence_penalty)
        self.llm_top_p_spin.valueChanged.connect(llm.set_top_p)
        self.llm_min_p_spin.valueChanged.connect(llm.set_min_p)
