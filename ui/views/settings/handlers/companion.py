"""Settings handler mixin: CompanionHandlersMixin."""

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


class CompanionHandlersMixin:
    """Behavior extracted from SettingsView."""

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
        self._show_settings_file_status("Notification history cleared.")
