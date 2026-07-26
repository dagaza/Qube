"""Settings handler mixin: StylingMixin."""

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
from ui.components.sidebar_list_qss import (
    apply_nav_list_sidebar_surface,
    apply_sidebar_row_title_colors,
)
from core.theme.view_theme import view_resolved_theme
from core.theme.widget_styles import (
    SETTINGS_BORDERED_LIST,
    SETTINGS_CHECKBOX,
    SETTINGS_FORM_CONTROLS,
    SETTINGS_LABEL,
    SETTINGS_LINE_EDIT,
    SETTINGS_SLIDER,
)
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
    general,
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
    "general": general.build_section,
    "companion.desktop": desktop_companion.build_section,
    "notifications": notifications.build_section,
    "help": help.build_section,
    "advanced": advanced.build_section,
}


class StylingMixin:
    """Behavior extracted from SettingsView."""

    def _iter_settings_checkboxes(self):
        """All Settings-page QCheckBox widgets that share the Prestige indicator style."""
        for name in (
            "pin_audio_cb",
            "pin_tts_voice_cb",
            "auto_load_last_model_cb",
            "auto_activator_cb",
            "rag_kb_cb",
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
            "companion_verbal_enabled_cb",
            "companion_cognition_v2_cb",
            "companion_verbal_react_ingest_cb",
            "companion_verbal_react_download_cb",
            "themes_auto_adjust_cb",
            "themes_assistant_message_background_cb",
            "themes_library_transcript_background_cb",
        ):
            cb = getattr(self, name, None)
            if cb is not None:
                yield cb
        for choice_cbs in (
            getattr(self, "companion_persona_cbs", {}),
            getattr(self, "companion_cube_style_cbs", {}),
            getattr(self, "companion_idle_color_cbs", {}),
            getattr(self, "ui_language_cbs", {}),
            getattr(self, "themes_appearance_cbs", {}),
            getattr(self, "themes_variant_cbs", {}),
        ):
            if isinstance(choice_cbs, dict):
                yield from choice_cbs.values()
        knowledge_cbs = getattr(self, "knowledge_source_checkboxes", None)
        if isinstance(knowledge_cbs, dict):
            for cb_list in knowledge_cbs.values():
                if isinstance(cb_list, list):
                    yield from cb_list

    def _apply_settings_checkbox_style(self, checkbox: QCheckBox | None) -> None:
        """Apply Prestige indicator styling to a single settings checkbox."""
        if checkbox is None:
            return
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        theme = view_resolved_theme(self, is_dark=is_dark)
        checkbox.setStyleSheet(theme.style(SETTINGS_CHECKBOX))

    def _iter_settings_line_edits(self):
        """Settings form text fields that need explicit light/dark input styling."""
        skip = {getattr(self, "settings_search_input", None)}
        for name in (
            "trigger_input",
            "discovery_searxng_url_field",
            "custom_source_id_input",
            "custom_source_label_input",
            "custom_source_base_url_input",
            "custom_source_search_path_input",
            "knowledge_preset_id_input",
            "knowledge_preset_label_input",
            "knowledge_preset_adapters_input",
            "knowledge_preset_site_bias_input",
            "knowledge_preset_fetch_count_input",
        ):
            field = getattr(self, name, None)
            if field is not None and field not in skip:
                yield field
        key_fields = getattr(self, "knowledge_provider_key_fields", None)
        if isinstance(key_fields, dict):
            for field in key_fields.values():
                if field is not None and field not in skip:
                    yield field

    def _apply_spinbox_style(self, is_dark: bool):
        """Forces borders to be visible on inputs, checkboxes, and the custom trigger elements."""
        theme = view_resolved_theme(self, is_dark=is_dark)
        style = theme.style(SETTINGS_FORM_CONTROLS)
        slider_css = theme.style(SETTINGS_SLIDER)
        label_style = theme.style(SETTINGS_LABEL, min_width="44px")
        line_edit_style = theme.style(SETTINGS_LINE_EDIT)

        self.timeout_spinner.setStyleSheet(style)
        self.threshold_spinner.setStyleSheet(style)
        for spinbox in getattr(self, "_generation_spinboxes", ()):
            spinbox.setStyleSheet(style)
        if hasattr(self, "native_chat_format_selector"):
            self._apply_settings_menu_button_chevron_state(self.native_chat_format_selector)
        if hasattr(self, "embedding_mode_selector"):
            self._apply_settings_menu_button_chevron_state(self.embedding_mode_selector)
        if hasattr(self, "gpu_layers_slider"):
            self.gpu_layers_slider.setStyleSheet(slider_css)
            self.gpu_layers_value_lbl.setStyleSheet(label_style)
            if hasattr(self, "cpu_threads_slider"):
                self.cpu_threads_slider.setStyleSheet(slider_css)
                self.cpu_threads_value_lbl.setStyleSheet(label_style)
        for cb in self._iter_settings_checkboxes():
            self._apply_settings_checkbox_style(cb)
        for name in (
            "mem_enrichment_label",
            "mem_promotion_label",
            "discovery_pacing_label",
            "active_native_model_lbl",
        ):
            lbl = getattr(self, name, None)
            if lbl is not None:
                lbl.setStyleSheet(theme.style(SETTINGS_LABEL))
        for field in self._iter_settings_line_edits():
            field.setStyleSheet(line_edit_style)

        for name in ("themes_chat_wallpaper", "themes_library_wallpaper"):
            editor = getattr(self, name, None)
            apply_theme = getattr(editor, "apply_theme", None)
            if callable(apply_theme):
                apply_theme(is_dark)

        update_themes_actions = getattr(self, "_update_themes_action_buttons", None)
        if callable(update_themes_actions):
            update_themes_actions()

        if hasattr(self, "trigger_list"):
            self.trigger_list.setStyleSheet(
                theme.style(
                    SETTINGS_BORDERED_LIST,
                    object_name="SettingsTriggerList",
                    widget_type="QListWidget",
                    item_padding="0px",
                )
            )

        if hasattr(self, "local_gguf_list"):
            self.local_gguf_list.setStyleSheet(
                theme.style(
                    SETTINGS_BORDERED_LIST,
                    widget_type="QListWidget",
                    item_padding="2px 12px",
                )
            )

    def _apply_settings_sidebar_surface(self, is_dark: bool) -> None:
        """Match Model Manager: tint only the left sidebar frame and section list."""
        hub_host = getattr(self, "_settings_hub_container", None)
        if hub_host is not None:
            hub_host.setAutoFillBackground(False)
            hub_host.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        apply_nav_list_sidebar_surface(
            is_dark=is_dark,
            sidebar_frame=getattr(self, "settings_sidebar", None),
            list_widget=getattr(self, "settings_section_list", None),
        )
