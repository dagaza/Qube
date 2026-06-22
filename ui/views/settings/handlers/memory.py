"""Settings handler mixin: MemoryHandlersMixin."""

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
    get_mcp_rag_enabled,
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


class MemoryHandlersMixin:
    """Behavior extracted from SettingsView."""

    def _build_triggers_manager(self) -> QWidget:
        """Builds the input box and list UI for custom RAG triggers."""
        container = QWidget()
        container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout = QVBoxLayout(container)
        layout.setContentsMargins(15, 0, 15, 10)
        layout.setSpacing(15)

        self.rag_kb_cb = QCheckBox("Enable Local Knowledge Base")
        self.rag_kb_cb.setChecked(get_mcp_rag_enabled())
        self.rag_kb_cb.setToolTip(
            "Master switch: grants Qube permission to read and cite your local library "
            "during chat. When off, library search still runs for custom trigger phrases "
            "if NLP Auto-Activator is enabled."
        )
        self.rag_kb_cb.toggled.connect(self._on_rag_kb_settings_toggled)
        layout.addWidget(self.rag_kb_cb)
        
        # Instruction Label
        instruction = QLabel("Add custom phrases that will trigger a semantic search of your Knowledge Base:")
        instruction.setStyleSheet("color: #64748b; font-size: 12px; font-style: italic;")
        layout.addWidget(instruction)

        # 🔑 NEW: Master Checkbox
        self.auto_activator_cb = QCheckBox("Enable NLP Auto-Activator")
        self.auto_activator_cb.setChecked(get_mcp_rag_auto_activator_enabled())
        self.auto_activator_cb.setToolTip(
            "When enabled, custom trigger phrases can search your Knowledge Base for a single turn, "
            "even if the master RAG switch is off. Add magic words below."
        )
        self.auto_activator_cb.toggled.connect(self._on_auto_activator_settings_toggled)
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
        self.trigger_list.setMinimumHeight(180)
        self.trigger_list.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.trigger_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.trigger_list.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.trigger_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        layout.addWidget(self.trigger_list, stretch=1)
        
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
        
        list_width = max(0, self.trigger_list.viewport().width())
        if list_width <= 0:
            list_width = max(0, self.trigger_list.width())
        for phrase in triggers:
            item = QListWidgetItem()
            row = QWidget()
            row.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
            )
            layout = QHBoxLayout(row)
            layout.setContentsMargins(15, 5, 10, 5)
            
            lbl = QLabel(phrase)
            lbl.setObjectName("SettingsTriggerPhraseLabel")
            lbl.setWordWrap(True)
            lbl.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
            )
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
            
            layout.addWidget(lbl, stretch=1)
            layout.addWidget(del_btn, stretch=0)
            
            row_h = self._trigger_row_height(phrase, list_width)
            item.setSizeHint(QSize(list_width, row_h))
            self.trigger_list.addItem(item)
            self.trigger_list.setItemWidget(item, row)

        self.trigger_list.doItemsLayout()

    def _trigger_row_text_width(self, list_width: int) -> int:
        """Usable text width inside a trigger list row (margins + delete button)."""
        return max(80, list_width - 68)

    def _trigger_row_height(self, phrase: str, list_width: int) -> int:
        fm = QFontMetrics(self.trigger_list.font())
        text_w = self._trigger_row_text_width(list_width)
        rect = fm.boundingRect(
            0,
            0,
            text_w,
            10000,
            int(Qt.TextFlag.TextWordWrap),
            phrase,
        )
        return max(44, rect.height() + 14)

    def _relayout_trigger_list_rows(self) -> None:
        if not hasattr(self, "trigger_list") or self.trigger_list.count() == 0:
            return
        list_width = self.trigger_list.viewport().width()
        if list_width <= 0:
            return
        for i in range(self.trigger_list.count()):
            item = self.trigger_list.item(i)
            row = self.trigger_list.itemWidget(item)
            if row is None:
                continue
            lbl = row.findChild(QLabel, "SettingsTriggerPhraseLabel")
            if lbl is None:
                continue
            phrase = lbl.text()
            item.setSizeHint(QSize(list_width, self._trigger_row_height(phrase, list_width)))
        self.trigger_list.doItemsLayout()

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

    def _on_rag_kb_settings_toggled(self, checked: bool) -> None:
        if checked:
            win = self.window()
            from ui.bootstrap_feature_prompts import ensure_search_models_for_feature

            if not ensure_search_models_for_feature(
                win,
                feature_label="Library knowledge base",
            ):
                self.rag_kb_cb.blockSignals(True)
                self.rag_kb_cb.setChecked(False)
                self.rag_kb_cb.blockSignals(False)
                return
        self.rag_kb_toggle.emit(checked)

    def _on_auto_activator_settings_toggled(self, checked: bool) -> None:
        if checked:
            win = self.window()
            from ui.bootstrap_feature_prompts import ensure_search_models_for_feature

            if not ensure_search_models_for_feature(
                win,
                feature_label="Library search phrases",
            ):
                self.auto_activator_cb.blockSignals(True)
                self.auto_activator_cb.setChecked(False)
                self.auto_activator_cb.blockSignals(False)
                return
        self.auto_activator_toggle.emit(checked)

    def _on_memory_enrichment_toggled(self, checked: bool):
        if checked:
            win = self.window()
            from core.bootstrap_manifest import BootstrapModelId
            from ui.bootstrap_feature_prompts import (
                ensure_bootstrap_model_downloaded,
                ensure_search_models_for_feature,
            )

            if not ensure_bootstrap_model_downloaded(
                win,
                BootstrapModelId.SIDECAR_QWEN17,
                feature_label="Memory enrichment",
            ):
                self.memory_enrichment_toggle.blockSignals(True)
                self.memory_enrichment_toggle.setChecked(False)
                self.memory_enrichment_toggle.blockSignals(False)
                return
            if not ensure_search_models_for_feature(
                win,
                feature_label="Memory enrichment",
            ):
                self.memory_enrichment_toggle.blockSignals(True)
                self.memory_enrichment_toggle.setChecked(False)
                self.memory_enrichment_toggle.blockSignals(False)
                return
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
