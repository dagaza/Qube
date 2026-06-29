"""Settings handler mixin: PersistenceHandlersMixin."""

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
from core.settings_section_reset import (
    SECTION_RESET_LABELS,
    reset_settings_section,
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
    get_skills_enabled,
    get_deep_research_enabled,
    get_external_knowledge_v2_enabled,
    get_internal_corpus_knowledge_enabled,
    get_research_map_enabled,
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
    get_advanced_embedding_unlocked,
    get_advanced_stt_unlocked,
    get_advanced_hardware_unlocked,
    get_advanced_chat_template_unlocked,
    get_advanced_tts_unlocked,
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


class PersistenceHandlersMixin:
    """Behavior extracted from SettingsView."""

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

    def _settings_file_status_hold_ms(self, message: str) -> int:
        chars = len(message.strip())
        ms = _SETTINGS_STATUS_BASE_HOLD_MS + chars * _SETTINGS_STATUS_MS_PER_CHAR
        return min(
            _SETTINGS_STATUS_MAX_HOLD_MS,
            max(_SETTINGS_STATUS_MIN_HOLD_MS, ms),
        )

    def _cancel_settings_file_status_fade(self) -> None:
        self._settings_file_status_sequence += 1
        if self._settings_file_status_fade_anim is not None:
            self._settings_file_status_fade_anim.stop()
            self._settings_file_status_fade_anim = None
        if hasattr(self, "settings_file_status_lbl"):
            self.settings_file_status_lbl.setGraphicsEffect(None)

    def _show_settings_file_status(self, message: str, *, persistent: bool = False) -> None:
        self._cancel_settings_file_status_fade()
        self.settings_file_status_lbl.setText(message)
        if persistent or not message.strip():
            return
        seq = self._settings_file_status_sequence
        QTimer.singleShot(
            self._settings_file_status_hold_ms(message),
            lambda: self._begin_settings_file_status_fade(seq),
        )

    def _begin_settings_file_status_fade(self, seq: int) -> None:
        if seq != self._settings_file_status_sequence:
            return
        lbl = self.settings_file_status_lbl
        if not lbl.text().strip():
            return
        eff = lbl.graphicsEffect()
        if not isinstance(eff, QGraphicsOpacityEffect):
            eff = QGraphicsOpacityEffect(lbl)
            lbl.setGraphicsEffect(eff)
        eff.setOpacity(1.0)
        anim = QPropertyAnimation(eff, b"opacity", self)
        anim.setDuration(_SETTINGS_STATUS_FADE_MS)
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.finished.connect(lambda: self._finish_settings_file_status_fade(seq))
        self._settings_file_status_fade_anim = anim
        anim.start()

    def _finish_settings_file_status_fade(self, seq: int) -> None:
        if seq != self._settings_file_status_sequence:
            return
        self.settings_file_status_lbl.clear()
        self.settings_file_status_lbl.setGraphicsEffect(None)
        self._settings_file_status_fade_anim = None

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
        self._show_settings_file_status(
            "Editing settings.json in the built-in editor.",
            persistent=True,
        )

    def _on_settings_editor_applied(self, changed: set) -> None:
        if not changed:
            return
        self._sync_ui_from_persisted_settings()
        self._apply_external_ui_language_change(changed)
        self._show_settings_file_status(
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
            self._show_settings_file_status(
                "settings.json has errors — fix JSON and save again."
            )
            return
        if result.skipped_keys:
            skipped = ", ".join(result.skipped_keys[:5])
            if len(result.skipped_keys) > 5:
                skipped += ", …"
            logger.info("Ignored unknown settings keys: %s", skipped)
        if not result.changed_keys:
            return
        self._sync_ui_from_persisted_settings()
        changed = set(result.changed_keys)
        self._apply_external_ui_language_change(changed)
        self._show_settings_file_status(
            f"Reloaded {len(changed)} setting(s) from settings.json."
        )
        self.external_settings_reloaded.emit(changed)

    def _apply_external_ui_language_change(self, changed: set) -> None:
        from core.app_settings import KEY_UI_LANGUAGE

        if KEY_UI_LANGUAGE not in changed:
            return
        if hasattr(self, "_rebuild_settings_sections_for_ui_language"):
            self._rebuild_settings_sections_for_ui_language()
        if hasattr(self, "ui_language_changed"):
            self.ui_language_changed.emit()
        win = self.window()
        if win is not None and hasattr(win, "_apply_ui_language"):
            win._apply_ui_language()

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
        if hasattr(self, "skills_enabled_toggle"):
            self.skills_enabled_toggle.blockSignals(True)
            self.skills_enabled_toggle.setChecked(get_skills_enabled())
            self.skills_enabled_toggle.blockSignals(False)
        if hasattr(self, "external_knowledge_v2_toggle"):
            self.external_knowledge_v2_toggle.blockSignals(True)
            self.external_knowledge_v2_toggle.setChecked(get_external_knowledge_v2_enabled())
            self.external_knowledge_v2_toggle.blockSignals(False)
        if hasattr(self, "internal_corpus_toggle"):
            self.internal_corpus_toggle.blockSignals(True)
            self.internal_corpus_toggle.setChecked(get_internal_corpus_knowledge_enabled())
            self.internal_corpus_toggle.blockSignals(False)
        if hasattr(self, "research_map_toggle"):
            self.research_map_toggle.blockSignals(True)
            self.research_map_toggle.setChecked(get_research_map_enabled())
            self.research_map_toggle.blockSignals(False)
        if hasattr(self, "deep_research_toggle"):
            self.deep_research_toggle.blockSignals(True)
            self.deep_research_toggle.setChecked(get_deep_research_enabled())
            self.deep_research_toggle.blockSignals(False)
        if hasattr(self, "knowledge_source_checkboxes"):
            from ui.views.settings.sections.knowledge_sources import sync_knowledge_source_checkboxes

            sync_knowledge_source_checkboxes(self)
        if hasattr(self, "memory_promotion_toggle"):
            self.memory_promotion_toggle.blockSignals(True)
            self.memory_promotion_toggle.setChecked(get_enable_memory_promotion())
            self.memory_promotion_toggle.blockSignals(False)
        if hasattr(self, "memory_consolidation_toggle"):
            self.memory_consolidation_toggle.blockSignals(True)
            self.memory_consolidation_toggle.setChecked(get_enable_memory_consolidation())
            self.memory_consolidation_toggle.blockSignals(False)
        if hasattr(self, "memory_promotion_preset_selector"):
            self._sync_memory_promotion_preset_selector()
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

        if hasattr(self, "ui_language_cbs"):
            from core import app_settings as _lang_settings

            current_language = _lang_settings.get_ui_language()
            for language_id, cb in self.ui_language_cbs.items():
                cb.blockSignals(True)
                cb.setChecked(language_id == current_language)
                cb.blockSignals(False)

        if hasattr(self, "companion_persona_cbs"):
            from core import app_settings as _cs

            current = _cs.get_companion_persona()
            for persona_id, cb in self.companion_persona_cbs.items():
                cb.blockSignals(True)
                cb.setChecked(persona_id == current)
                cb.blockSignals(False)
            if hasattr(self, "companion_preview"):
                self.companion_preview.set_persona(current)

        if hasattr(self, "companion_cube_style_cbs"):
            from core import app_settings as _cs

            current_style = _cs.get_companion_cube_style()
            for style_id, cb in self.companion_cube_style_cbs.items():
                cb.blockSignals(True)
                cb.setChecked(style_id == current_style)
                cb.blockSignals(False)
            if hasattr(self, "companion_preview"):
                self.companion_preview.set_persona(_cs.get_companion_persona())
            if hasattr(self, "_sync_companion_cube_style_enabled"):
                self._sync_companion_cube_style_enabled()

        if hasattr(self, "companion_idle_color_cbs"):
            from core import app_settings as _cs

            current_idle = _cs.get_companion_idle_color()
            for color_id, cb in self.companion_idle_color_cbs.items():
                cb.blockSignals(True)
                cb.setChecked(color_id == current_idle)
                cb.blockSignals(False)
            if hasattr(self, "companion_preview"):
                self.companion_preview.update()

        if hasattr(self, "_sync_companion_snap_compass"):
            self._sync_companion_snap_compass()

        if hasattr(self, "advanced_engine_toggle"):
            self.advanced_engine_toggle.blockSignals(True)
            self.advanced_engine_toggle.setChecked(get_advanced_engine_unlocked())
            self.advanced_engine_toggle.blockSignals(False)
            if hasattr(self, "advanced_engine_panel"):
                self.advanced_engine_panel.setVisible(get_advanced_engine_unlocked())

        if hasattr(self, "advanced_embedding_toggle"):
            self.advanced_embedding_toggle.blockSignals(True)
            self.advanced_embedding_toggle.setChecked(get_advanced_embedding_unlocked())
            self.advanced_embedding_toggle.blockSignals(False)
            if hasattr(self, "advanced_embedding_panel"):
                self.advanced_embedding_panel.setVisible(get_advanced_embedding_unlocked())

        self._sync_embedding_models_dir_label()
        self._refresh_embedding_gguf_list()
        self._sync_active_embedding_label()
        if hasattr(self, "_sync_embedding_mode_selector"):
            self._sync_embedding_mode_selector()

        if hasattr(self, "advanced_stt_toggle"):
            self.advanced_stt_toggle.blockSignals(True)
            self.advanced_stt_toggle.setChecked(get_advanced_stt_unlocked())
            self.advanced_stt_toggle.blockSignals(False)
            if hasattr(self, "advanced_stt_panel"):
                self.advanced_stt_panel.setVisible(get_advanced_stt_unlocked())

        if hasattr(self, "advanced_tts_toggle"):
            self.advanced_tts_toggle.blockSignals(True)
            self.advanced_tts_toggle.setChecked(get_advanced_tts_unlocked())
            self.advanced_tts_toggle.blockSignals(False)
            if hasattr(self, "advanced_tts_panel"):
                self.advanced_tts_panel.setVisible(get_advanced_tts_unlocked())

        if hasattr(self, "advanced_hardware_toggle"):
            self.advanced_hardware_toggle.blockSignals(True)
            self.advanced_hardware_toggle.setChecked(get_advanced_hardware_unlocked())
            self.advanced_hardware_toggle.blockSignals(False)

        if hasattr(self, "advanced_chat_template_toggle"):
            self.advanced_chat_template_toggle.blockSignals(True)
            self.advanced_chat_template_toggle.setChecked(
                get_advanced_chat_template_unlocked()
            )
            self.advanced_chat_template_toggle.blockSignals(False)

        if hasattr(self, "_sync_hardware_chat_template_panels"):
            self._sync_hardware_chat_template_panels()

        self._sync_stt_models_dir_label()
        self._sync_tts_models_dir_label()
        self._refresh_stt_model_list()
        self._refresh_tts_model_list()
        self._sync_active_stt_label()
        self._sync_active_tts_label()

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

    def _on_reset_section_defaults(self, section_id: str) -> None:
        title = SECTION_RESET_LABELS.get(section_id, "Settings")
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        dlg = PrestigeDialog(
            self.window(),
            f"Reset {title}",
            (
                f"This will restore every setting on the {title} page to its "
                "default configuration.\n\n"
                "Custom overrides on this page will be removed. This cannot be undone."
            ),
            is_dark=is_dark,
            tone="danger",
            confirm_text="RESET",
        )
        if not dlg.exec():
            return

        changed = set(reset_settings_section(section_id))
        if section_id == "knowledge" and hasattr(self, "db") and self.db is not None:
            self.db.reset_rag_triggers_to_defaults()
            if hasattr(self, "_refresh_trigger_list"):
                self._refresh_trigger_list()
            if hasattr(self, "_refresh_llm_rag_triggers"):
                self._refresh_llm_rag_triggers()

        self._apply_section_defaults_to_ui(section_id)
        if section_id == "companion.desktop":
            win = self.window()
            ctrl = getattr(win, "_companion_controller", None) if win is not None else None
            if ctrl is not None:
                ctrl.reset_position_to_default()
            if hasattr(self, "_sync_companion_snap_compass"):
                self._sync_companion_snap_compass()
        self._show_settings_file_status(f"{title} settings restored to defaults.")
        if changed:
            self.external_settings_reloaded.emit(changed)

    def _apply_section_defaults_to_ui(self, section_id: str) -> None:
        self._sync_ui_from_persisted_settings()

        if section_id == "voice.audio":
            self._apply_voice_audio_defaults_to_ui()
        elif section_id == "ai.models":
            self._apply_ai_models_defaults_to_ui()
        elif section_id == "memory":
            if hasattr(self, "_sync_memory_promotion_controls_for_enrichment"):
                self._sync_memory_promotion_controls_for_enrichment()
        elif section_id == "knowledge":
            if hasattr(self, "rag_kb_cb"):
                self.rag_kb_cb.blockSignals(True)
                self.rag_kb_cb.setChecked(get_mcp_rag_enabled())
                self.rag_kb_cb.blockSignals(False)
                self.rag_kb_toggle.emit(self.rag_kb_cb.isChecked())
            if hasattr(self, "auto_activator_cb"):
                self.auto_activator_cb.blockSignals(True)
                self.auto_activator_cb.setChecked(get_mcp_rag_auto_activator_enabled())
                self.auto_activator_cb.blockSignals(False)
                self.auto_activator_toggle.emit(self.auto_activator_cb.isChecked())
        elif section_id == "companion.desktop":
            self._apply_companion_defaults_to_ui()
        elif section_id == "notifications":
            pass

    def _apply_voice_audio_defaults_to_ui(self) -> None:
        if hasattr(self, "timeout_spinner"):
            self.timeout_spinner.blockSignals(True)
            self.timeout_spinner.setValue(2.0)
            self.timeout_spinner.blockSignals(False)
            if self.audio_worker:
                self.audio_worker.set_silence_timeout(2.0)
        if hasattr(self, "threshold_spinner"):
            self.threshold_spinner.blockSignals(True)
            self.threshold_spinner.setValue(2)
            self.threshold_spinner.blockSignals(False)
            if self.audio_worker:
                self.audio_worker.set_speech_threshold(2)
        for cb_name, signal in (
            ("pin_audio_cb", self.audio_pin_toggle),
            ("pin_tts_voice_cb", self.tts_voice_pin_toggle),
        ):
            cb = getattr(self, cb_name, None)
            if cb is None:
                continue
            cb.blockSignals(True)
            cb.setChecked(True)
            cb.blockSignals(False)
            signal.emit(True)
        mics = get_input_devices()
        if mics and hasattr(self, "mic_selector"):
            self.mic_selector.setText(mics[0][1])
        outputs = get_output_devices()
        if outputs and hasattr(self, "device_selector"):
            self.device_selector.setText(outputs[0][1])
        if self.audio_worker:
            self._sync_wakeword_catalog(trigger="section reset")
        if hasattr(self, "advanced_stt_panel"):
            self.advanced_stt_panel.setVisible(get_advanced_stt_unlocked())
        if hasattr(self, "advanced_tts_panel"):
            self.advanced_tts_panel.setVisible(get_advanced_tts_unlocked())
        if hasattr(self, "stt_model_changed"):
            self.stt_model_changed.emit()
        if hasattr(self, "tts_model_changed"):
            self.tts_model_changed.emit()

    def _apply_ai_models_defaults_to_ui(self) -> None:
        from core import app_settings as _as

        spin_map = (
            ("llm_temp_spin", _as.get_llm_temperature),
            ("llm_ctx_spin", _as.get_llm_context_limit),
            ("llm_output_limit_spin", _as.get_llm_output_token_limit),
            ("llm_history_spin", _as.get_llm_chat_history_messages),
            ("llm_top_k_spin", _as.get_llm_top_k),
            ("llm_top_p_spin", _as.get_llm_top_p),
            ("llm_min_p_spin", _as.get_llm_min_p),
            ("llm_repeat_penalty_spin", _as.get_llm_repeat_penalty),
            ("llm_presence_penalty_spin", _as.get_llm_presence_penalty),
        )
        for attr, getter in spin_map:
            spin = getattr(self, attr, None)
            if spin is None:
                continue
            spin.blockSignals(True)
            spin.setValue(getter())
            spin.blockSignals(False)
        if hasattr(self, "llm_output_limit_cb"):
            self.llm_output_limit_cb.blockSignals(True)
            self.llm_output_limit_cb.setChecked(_as.get_llm_output_token_limit_enabled())
            self.llm_output_limit_cb.blockSignals(False)
        if hasattr(self, "_sync_output_limit_controls"):
            self._sync_output_limit_controls()
        if hasattr(self, "generation_advanced_toggle"):
            self.generation_advanced_toggle.blockSignals(True)
            self.generation_advanced_toggle.setChecked(False)
            self.generation_advanced_toggle.blockSignals(False)
        if hasattr(self, "generation_advanced_panel"):
            self.generation_advanced_panel.setVisible(False)
        if hasattr(self, "advanced_hardware_toggle"):
            self.advanced_hardware_toggle.blockSignals(True)
            self.advanced_hardware_toggle.setChecked(get_advanced_hardware_unlocked())
            self.advanced_hardware_toggle.blockSignals(False)
        if hasattr(self, "advanced_hardware_panel"):
            self.advanced_hardware_panel.setVisible(get_advanced_hardware_unlocked())
        if hasattr(self, "advanced_chat_template_toggle"):
            self.advanced_chat_template_toggle.blockSignals(True)
            self.advanced_chat_template_toggle.setChecked(
                get_advanced_chat_template_unlocked()
            )
            self.advanced_chat_template_toggle.blockSignals(False)
        if hasattr(self, "_sync_hardware_chat_template_panels"):
            self._sync_hardware_chat_template_panels()
        if hasattr(self, "advanced_engine_toggle"):
            self.advanced_engine_toggle.blockSignals(True)
            self.advanced_engine_toggle.setChecked(get_advanced_engine_unlocked())
            self.advanced_engine_toggle.blockSignals(False)
        if hasattr(self, "advanced_engine_panel"):
            self.advanced_engine_panel.setVisible(get_advanced_engine_unlocked())
        if hasattr(self, "_reload_sidecar_from_settings"):
            self._reload_sidecar_from_settings()
        from core.logging_bootstrap import sync_diagnostic_file_sinks_from_settings

        sync_diagnostic_file_sinks_from_settings()
        if hasattr(self, "_sync_all_diagnostic_log_recording_toggles"):
            self._sync_all_diagnostic_log_recording_toggles()
        if hasattr(self, "cognition_model_changed"):
            self.cognition_model_changed.emit()
        llm = getattr(self, "llm_worker", None)
        if llm is not None:
            llm.set_temperature(_as.get_llm_temperature())
            llm.set_context_window(_as.get_llm_context_limit())
            llm.set_output_token_limit_enabled(_as.get_llm_output_token_limit_enabled())
            llm.set_output_token_limit(_as.get_llm_output_token_limit())
            llm.set_max_history_messages(_as.get_llm_chat_history_messages())
            llm.set_top_k(_as.get_llm_top_k())
            llm.set_repeat_penalty(_as.get_llm_repeat_penalty())
            llm.set_presence_penalty(_as.get_llm_presence_penalty())
            llm.set_top_p(_as.get_llm_top_p())
            llm.set_min_p(_as.get_llm_min_p())
        em = get_engine_mode()
        self.engine_mode_changed.emit(em)
        win = self.window()
        if win is not None and hasattr(win, "refresh_toolbar_native_model_dropdown"):
            win.refresh_toolbar_native_model_dropdown()

    def _apply_companion_defaults_to_ui(self) -> None:
        from core import app_settings as _cs

        checkbox_map = (
            ("companion_tray_hidden_cb", _cs.get_companion_show_when_tray_hidden),
            ("companion_while_open_cb", _cs.get_companion_show_while_window_open),
            ("companion_auto_hide_cb", _cs.get_companion_auto_hide_idle),
            ("companion_caption_cb", _cs.get_companion_show_caption),
            ("companion_fullscreen_cb", _cs.get_companion_suppress_on_fullscreen),
            ("companion_wayland_cb", _cs.get_companion_try_on_wayland),
            ("companion_dock_cb", _cs.get_companion_dock_mode),
            ("companion_verbal_enabled_cb", _cs.get_companion_verbal_enabled),
            ("companion_cognition_v2_cb", _cs.get_companion_cognition_v2_enabled),
            ("companion_verbal_react_ingest_cb", _cs.get_companion_verbal_react_ingest),
            ("companion_verbal_react_download_cb", _cs.get_companion_verbal_react_download),
        )
        for attr, getter in checkbox_map:
            cb = getattr(self, attr, None)
            if cb is None:
                continue
            cb.blockSignals(True)
            cb.setChecked(getter())
            cb.blockSignals(False)
        if hasattr(self, "companion_verbal_prompt"):
            self.companion_verbal_prompt.blockSignals(True)
            self.companion_verbal_prompt.setPlainText(_cs.get_companion_verbal_system_prompt())
            self.companion_verbal_prompt.blockSignals(False)
        if hasattr(self, "_build_companion_verbal_trait_menu"):
            self._build_companion_verbal_trait_menu()
        if hasattr(self, "_build_companion_verbal_frequency_menu"):
            self._build_companion_verbal_frequency_menu()
        if hasattr(self, "_build_companion_expression_freedom_menu"):
            self._build_companion_expression_freedom_menu()
        if hasattr(self, "_sync_companion_verbal_controls_enabled"):
            self._sync_companion_verbal_controls_enabled()
        win = self.window()
        if win is not None and hasattr(win, "_companion_controller") and win._companion_controller is not None:
            win._companion_controller.on_settings_changed()
        if win is not None and hasattr(win, "tray_controller") and win.tray_controller is not None:
            win.tray_controller.sync_companion_toggle()