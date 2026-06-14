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
