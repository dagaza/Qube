"""Settings handler mixin: AiModelsHandlersMixin."""

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
    get_skills_enabled,
    set_skills_enabled,
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
    get_advanced_hardware_unlocked,
    set_advanced_hardware_unlocked,
    get_advanced_chat_template_unlocked,
    set_advanced_chat_template_unlocked,
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
    general,
    help,
    integrations,
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
    "integrations": integrations.build_section,
    "general": general.build_section,
    "companion.desktop": desktop_companion.build_section,
    "notifications": notifications.build_section,
    "help": help.build_section,
    "advanced": advanced.build_section,
}


class AiModelsHandlersMixin:
    """Behavior extracted from SettingsView."""

    def _sync_ai_provider_enabled_for_inference(self, mode: str) -> None:
        """LM Studio / Ollama only applies when routing to an external OpenAI-compatible server."""
        if not hasattr(self, "provider_selector"):
            return
        m = str(mode).lower().strip()
        self.provider_selector.setEnabled(m == "external")
        self._apply_settings_menu_button_chevron_state(self.provider_selector)
        self._sync_active_native_model_label()
        self._sync_internal_engine_subsections(m)

    def _sync_models_dir_label(self) -> None:
        if not hasattr(self, "models_dir_label"):
            return
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
        else:
            mw = self.window()
            ne = getattr(mw, "_native_engine", None) if mw else None
            snap = ne.get_model_reasoning_telemetry() if ne else None
            loaded = bool((snap or {}).get("loaded"))
            loaded_name = str((snap or {}).get("model_basename") or "").strip()

            if loaded and loaded_name:
                self.active_native_model_lbl.setText(loaded_name)
            elif path_name:
                self.active_native_model_lbl.setText(f"{path_name} (not loaded)")
            else:
                self.active_native_model_lbl.setText("(none)")

    def _refresh_inference_transparency_panel(self, *, is_dark: bool | None = None) -> None:
        table = getattr(self, "inference_transparency_table", None)
        if table is None:
            return
        from core.inference_transparency import (
            aggregate_app_transparency,
            format_transparency_rows,
        )
        from ui.views.settings.knowledge_access_badge import coalesce_settings_is_dark
        from ui.views.settings.knowledge_list_table import populate_table_rows

        if is_dark is None:
            is_dark = coalesce_settings_is_dark(self)
        mw = self.window()
        ne = getattr(mw, "_native_engine", None) if mw else self.workers.get("native_engine")
        try:
            snap = aggregate_app_transparency(
                native_engine=ne,
                embedder=self.workers.get("embedder") if getattr(self, "workers", None) else None,
                sidecar_worker=self.workers.get("sidecar_worker") if getattr(self, "workers", None) else None,
            )
            rows = format_transparency_rows(snap)
            populate_table_rows(
                table,
                rows=rows,
                placeholder="Inference stack details unavailable.",
                is_dark=is_dark,
            )
        except Exception as e:
            logger.debug("Inference transparency panel refresh failed: %s", e)
            populate_table_rows(
                table,
                rows=[],
                placeholder="Inference stack details unavailable.",
                is_dark=is_dark,
            )

    def _on_gpu_layers_slider_changed(self, v: int) -> None:
        self.gpu_layers_value_lbl.setText(str(int(v)))
        self._on_native_gpu_layers_changed(int(v))

    def _on_cpu_threads_slider_changed(self, v: int) -> None:
        self.cpu_threads_value_lbl.setText(str(int(v)))
        set_internal_n_threads(int(v))
        llm = self.workers.get("llm")
        if llm and getattr(llm, "engine_mode", DEFAULT_ENGINE_MODE) == "internal":
            llm.refresh_native_model_from_settings(notify_hardware_reload=True)

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
            self._refresh_inference_transparency_panel()
            return

        if self._auto_reset_reload_pending:
            # Completion belongs to reset->reload sequence.
            self._auto_reset_reload_pending = False
            self._sync_active_native_model_label()
            self._sync_native_chat_template_label()
            self._refresh_inference_transparency_panel()
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
        self._refresh_inference_transparency_panel()

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
        self._refresh_inference_transparency_panel()
        llm = self.workers.get("llm")
        if llm and getattr(llm, "engine_mode", DEFAULT_ENGINE_MODE) == "internal":
            llm.refresh_native_model_from_settings(notify_hardware_reload=True)

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

    def _on_refresh_local_gguf_clicked(self) -> None:
        self.refresh_native_local_library()
        self._refresh_toolbar_native_model_after_model_change()

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

    def _on_advanced_hardware_toggled(self, checked: bool) -> None:
        if checked:
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            dlg = PrestigeDialog(
                self.window(),
                "Advanced hardware settings",
                "GPU offload layers and CPU thread counts directly affect native engine "
                "performance and stability.\n\n"
                "Setting GPU layers too high can exhaust video memory and crash the app. "
                "Using nearly all CPU cores may slow other applications.\n\nContinue?",
                is_dark=is_dark,
                tone="danger",
                dialog_width=450,
            )
            if not dlg.exec():
                self.advanced_hardware_toggle.blockSignals(True)
                self.advanced_hardware_toggle.setChecked(False)
                self.advanced_hardware_toggle.blockSignals(False)
                return
        set_advanced_hardware_unlocked(bool(checked))
        self._sync_hardware_chat_template_panels()

    def _on_advanced_chat_template_toggled(self, checked: bool) -> None:
        if checked:
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            dlg = PrestigeDialog(
                self.window(),
                "Advanced chat template settings",
                "Manual chat template overrides change how prompts are formatted for the "
                "native engine.\n\n"
                "Auto usually matches the loaded model. An incorrect template can cause "
                "hallucinations or the model talking to itself.\n\nContinue?",
                is_dark=is_dark,
                tone="danger",
                dialog_width=450,
            )
            if not dlg.exec():
                self.advanced_chat_template_toggle.blockSignals(True)
                self.advanced_chat_template_toggle.setChecked(False)
                self.advanced_chat_template_toggle.blockSignals(False)
                return
        set_advanced_chat_template_unlocked(bool(checked))
        self._sync_hardware_chat_template_panels()

    def _show_ai_models_subsection_labels(self, *anchors: str) -> None:
        anchor_set = set(anchors)
        for lbl in getattr(self, "_ai_internal_subsection_labels", []):
            anchor = lbl.property("settings_anchor")
            if anchor in anchor_set:
                lbl.setVisible(True)

    def _sync_hardware_chat_template_panels(self) -> None:
        internal = str(get_engine_mode()).lower().strip() == "internal"
        tour_hw = getattr(self, "_tour_hardware_preview_active", False)
        tour_hw_row = getattr(self, "_tour_hardware_row_preview_active", False)
        tour_ct = getattr(self, "_tour_chat_template_preview_active", False)
        tour_ct_row = getattr(self, "_tour_chat_template_row_preview_active", False)
        hw_unlocked = get_advanced_hardware_unlocked()
        ct_unlocked = get_advanced_chat_template_unlocked()

        hw_row = getattr(self, "advanced_hardware_row", None)
        if hw_row is not None:
            hw_row.setVisible(internal or tour_hw_row or tour_hw)
        ct_row = getattr(self, "advanced_chat_template_row", None)
        if ct_row is not None:
            ct_row.setVisible(internal or tour_ct_row or tour_ct)

        hw_panel = getattr(self, "advanced_hardware_panel", None)
        if hw_panel is not None:
            hw_panel.setVisible((internal and hw_unlocked) or tour_hw)
        ct_panel = getattr(self, "advanced_chat_template_panel", None)
        if ct_panel is not None:
            ct_panel.setVisible((internal and ct_unlocked) or tour_ct)

        if hasattr(self, "advanced_hardware_toggle"):
            self.advanced_hardware_toggle.blockSignals(True)
            self.advanced_hardware_toggle.setChecked(
                True if (tour_hw or tour_hw_row) else hw_unlocked
            )
            self.advanced_hardware_toggle.blockSignals(False)
        if hasattr(self, "advanced_chat_template_toggle"):
            self.advanced_chat_template_toggle.blockSignals(True)
            self.advanced_chat_template_toggle.setChecked(
                True if (tour_ct or tour_ct_row) else ct_unlocked
            )
            self.advanced_chat_template_toggle.blockSignals(False)

    def begin_ai_models_hardware_tutorial_preview(
        self, *, reveal_panel: bool = True
    ) -> None:
        """Reveal advanced hardware controls during the AI & Models guided tour."""
        self._tour_hardware_row_preview_active = True
        if reveal_panel:
            self._tour_hardware_preview_active = True
        self._show_ai_models_subsection_labels("hardware", "inference_stack")
        self._sync_hardware_chat_template_panels()

    def end_ai_models_hardware_tutorial_preview(self) -> None:
        """Restore advanced hardware panel visibility after the guided tour."""
        if not (
            getattr(self, "_tour_hardware_preview_active", False)
            or getattr(self, "_tour_hardware_row_preview_active", False)
        ):
            return
        self._tour_hardware_preview_active = False
        self._tour_hardware_row_preview_active = False
        self._sync_internal_engine_subsections(get_engine_mode())

    def begin_ai_models_chat_template_tutorial_preview(
        self, *, reveal_panel: bool = True
    ) -> None:
        """Reveal advanced chat template controls during the AI & Models guided tour."""
        self._tour_chat_template_row_preview_active = True
        if reveal_panel:
            self._tour_chat_template_preview_active = True
        self._show_ai_models_subsection_labels("chat_template")
        self._sync_hardware_chat_template_panels()

    def end_ai_models_chat_template_tutorial_preview(self) -> None:
        """Restore advanced chat template panel visibility after the guided tour."""
        if not (
            getattr(self, "_tour_chat_template_preview_active", False)
            or getattr(self, "_tour_chat_template_row_preview_active", False)
        ):
            return
        self._tour_chat_template_preview_active = False
        self._tour_chat_template_row_preview_active = False
        self._sync_internal_engine_subsections(get_engine_mode())

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

    def _on_open_composer_mention_guide_clicked(self) -> None:
        from ui.components.composer_mention_guide_dialog import show_composer_mention_guide

        is_dark = getattr(self.window(), "_is_dark_theme", True)
        show_composer_mention_guide(self.window(), is_dark=is_dark)

    def _on_model_manager_hardware_suggestions_toggled(self, enabled: bool) -> None:
        set_model_manager_hardware_suggestions(enabled)
        win = self.window()
        mm = getattr(win, "_model_manager_view", None) if win is not None else None
        if mm is not None and hasattr(mm, "refresh_hardware_suggestions"):
            mm.refresh_hardware_suggestions()

    def _on_composer_bare_mention_routing_toggled(self, enabled: bool) -> None:
        from core.app_settings import set_composer_bare_mention_routing_enabled

        set_composer_bare_mention_routing_enabled(enabled)

    def _on_chat_personality_toggled(self, checked: bool) -> None:
        set_enable_chat_personality_nudge(checked)

    def _on_skills_enabled_toggled(self, checked: bool) -> None:
        set_skills_enabled(checked)
        if checked:
            message = (
                "Reasoning skills are now on. Auto-detected skills inject guidance "
                "after routing; use @ in the composer to force a skill."
            )
        else:
            message = (
                "Reasoning skills auto-detection is off. "
                "@[skill:…] tokens in the composer still work."
            )
        self._show_settings_file_status(message, persistent=True)
