"""Settings handler mixin: VoiceHandlersMixin."""

from __future__ import annotations

# Shared imports from settings shell (handlers use ``self`` as SettingsView).
import os
import logging
import shutil
from collections.abc import Callable
from pathlib import Path
import qtawesome as qta
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QFrame, QPushButton,
    QLabel, QCheckBox, QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox, QScrollArea, QProgressBar,
    QToolButton,
    QStyledItemDelegate, QListView, QMenu, QListWidget, QListWidgetItem, QSlider,
    QButtonGroup, QPlainTextEdit, QGraphicsOpacityEffect, QStackedWidget, QSizePolicy,
    QWidgetAction,
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QTimer, QFileSystemWatcher, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFontMetrics, QResizeEvent, QShowEvent
from core.audio_utils import get_input_devices, get_output_devices, build_audio_device_menu_rows
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
    get_advanced_stt_unlocked,
    set_advanced_stt_unlocked,
    get_advanced_tts_unlocked,
    set_advanced_tts_unlocked,
    set_stt_model_path,
    set_tts_model_path,
)
from core.model_paths_pro_features import (
    LICENSE_REQUIRED_MESSAGE as CUSTOM_MODEL_PATHS_LICENSE_MESSAGE,
    effective_advanced_stt_unlocked,
    effective_advanced_tts_unlocked,
    sync_custom_model_paths_pro_features,
    user_has_pro_custom_model_paths,
)
from core.wakeword_pro_features import (
    LICENSE_REQUIRED_MESSAGE as WAKEWORD_LIBRARY_LICENSE_MESSAGE,
    build_wakeword_menu_items,
    revoke_unlicensed_wakeword_selection,
    sync_wakeword_pro_features,
    user_has_pro_wakeword_library,
    wakeword_selection_allowed,
)
from core.tts_voice_preview import next_tts_voice_preview_phrase
from core.stt_models import (
    BUNDLED_STT_MODEL_ID,
    get_stt_models_dir,
    is_protected_stt_model,
    list_selectable_stt_models,
    resolve_active_stt_model_spec,
    validate_stt_model_path,
)
from core.tts_models import (
    bundled_default_path,
    get_tts_models_dir,
    is_protected_tts_model,
    list_selectable_tts_models,
    migrate_legacy_tts_layout,
    migrate_stale_tts_override,
    any_supported_tts_model_on_disk,
    describe_tts_model_disk_state,
    resolve_active_tts_path,
    validate_tts_model_path,
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
SPEECH_ENTRY_DELETABLE_ROLE = int(Qt.ItemDataRole.UserRole) + 4
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


class VoiceHandlersMixin:
    """Behavior extracted from SettingsView."""

    def _show_wakeword_library_license_dialog(self) -> None:
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        PrestigeDialog(
            self.window(),
            "Pro license required",
            WAKEWORD_LIBRARY_LICENSE_MESSAGE,
            is_dark=is_dark,
        ).exec()

    def _sync_wakeword_pro_features(self) -> None:
        sync_wakeword_pro_features(self)

    def _sync_wakeword_catalog(self, trigger: str = "manual") -> None:
        _ = trigger
        if not self.audio_worker:
            return
        from core.wakeword_manager import WakewordManager

        manager = getattr(self.audio_worker, "wakeword_manager", None)
        if not isinstance(manager, WakewordManager):
            return
        try:
            self.audio_worker.refresh_wakewords(include_remote=False)
            revoke_unlicensed_wakeword_selection(self.audio_worker)
            wakeword_items = build_wakeword_menu_items(self.audio_worker.wakeword_manager)
            if not wakeword_items:
                self.wakeword_selector.setEnabled(False)
                self.wakeword_selector.setText("No model available")
                self.wakeword_selector.setMenu(QMenu(self.wakeword_selector))
                return

            licensed = user_has_pro_wakeword_library()
            self.wakeword_selector.setEnabled(True)
            if licensed and len(wakeword_items) > 1:
                self._build_prestige_menu(
                    self.wakeword_selector,
                    wakeword_items,
                    self._on_wakeword_selection_changed,
                )
            else:
                self.wakeword_selector.setMenu(QMenu(self.wakeword_selector))

            active_name = getattr(self.audio_worker, "active_wakeword_name", "") or wakeword_items[0][1]
            matching_label = next(
                (label for label, data in wakeword_items if data == active_name),
                wakeword_items[0][0],
            )
            self.wakeword_selector.setText(matching_label)
        except Exception as exc:
            logger.exception("Wakeword catalog sync failed: %s", exc)
            if hasattr(self, "wakeword_selector"):
                self.wakeword_selector.setEnabled(False)
                self.wakeword_selector.setText("No model available")
                self.wakeword_selector.setMenu(QMenu(self.wakeword_selector))

    def _on_wakeword_selector_pressed(self) -> None:
        if not user_has_pro_wakeword_library():
            self._show_wakeword_library_license_dialog()
            self._sync_wakeword_catalog(trigger="dropdown")
            return
        self._sync_wakeword_catalog(trigger="dropdown")

    def _set_wakeword_download_buttons_enabled(self, enabled: bool) -> None:
        for btn_name in ("wakeword_download_open_btn", "wakeword_download_community_btn"):
            btn = getattr(self, btn_name, None)
            if btn is not None:
                btn.setEnabled(bool(enabled))

    def _start_wakeword_download(self, kind: str) -> None:
        if not user_has_pro_wakeword_library():
            self._show_wakeword_library_license_dialog()
            return
        if getattr(self, "_wakeword_download_worker", None) is not None and self._wakeword_download_worker.isRunning():
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            PrestigeDialog(
                self.window(),
                "Download busy",
                "A wakeword download is already in progress.",
                is_dark=is_dark,
            ).exec()
            return

        self._set_wakeword_download_buttons_enabled(False)

        from workers.wakeword_models_download_worker import WakewordModelsDownloadWorker

        worker = WakewordModelsDownloadWorker(kind=kind)
        self._wakeword_download_worker = worker

        is_dark = getattr(self.window(), "_is_dark_theme", True)
        self._download_dialog = PrestigeDialog(
            self.window(),
            "Downloading wakewords",
            "Working… this may take a while on first run.",
            is_dark=is_dark,
            show_cancel=False,
        )
        self._download_dialog.show()

        def _on_ok() -> None:
            try:
                if getattr(self, "_download_dialog", None) is not None:
                    self._download_dialog.accept()
            except Exception:
                pass

            self._set_wakeword_download_buttons_enabled(True)
            if self.audio_worker:
                self._sync_wakeword_catalog(trigger="wakeword download")
            if getattr(self, "_wakeword_testbed_dialog", None) is not None:
                self._wakeword_testbed_dialog.on_wakeword_selection_changed(sync_catalog=True)

            PrestigeDialog(
                self.window(),
                "Download complete",
                "Wakeword models are ready. Open the Test Lab or select a wakeword in Settings.",
                is_dark=is_dark,
            ).exec()

        def _on_failed(err: str) -> None:
            try:
                if getattr(self, "_download_dialog", None) is not None:
                    self._download_dialog.reject()
            except Exception:
                pass

            self._set_wakeword_download_buttons_enabled(True)

            # Also raise a non-modal in-app notification for visibility (especially
            # when the OpenWakeWord install directory is read-only).
            try:
                from core.notification_types import NotificationEvent, NotificationSeverity

                err_norm = str(err or "").lower()
                is_readonly = "not writable" in err_norm or "read-only" in err_norm
                title = "Wakeword download"
                if is_readonly:
                    body = (
                        "OpenWakeWord model install directory is read-only, so built-in "
                        "wakewords could not be downloaded automatically.\n\n"
                        f"{err}"
                    )
                else:
                    body = str(err or "Wakeword download failed.")

                self.window().emit_notification(
                    NotificationEvent(
                        title=title,
                        body=body,
                        severity=NotificationSeverity.WARNING,
                        category="system",
                        action_label="Help",
                        action_id="open_help_wakeword_models",
                        auto_dismiss_ms=8000,
                    )
                )
            except Exception:
                # Fallback to the blocking dialog only.
                pass

            PrestigeDialog(
                self.window(),
                "Wakeword download failed",
                f"{err}",
                is_dark=is_dark,
                tone="danger",
            ).exec()

        worker.finished_ok.connect(_on_ok)
        worker.failed.connect(_on_failed)
        worker.start()

    def _open_wakeword_test_lab(self) -> None:
        if not user_has_pro_wakeword_library():
            self._show_wakeword_library_license_dialog()
            return
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

    def _download_openwakeword_models(self) -> None:
        self._start_wakeword_download(kind="openwakeword")

    def _download_community_wakeword_models(self) -> None:
        self._start_wakeword_download(kind="community")

    def _on_wakeword_selection_changed(self, display_name: str) -> None:
        if not self.audio_worker:
            return
        spec = self.audio_worker.catalog_by_ui_name.get(display_name)
        if spec is not None and not wakeword_selection_allowed(
            spec, self.audio_worker.wakeword_manager
        ):
            self._show_wakeword_library_license_dialog()
            self._sync_wakeword_catalog(trigger="selection blocked")
            return
        self._wakeword_selected_label = str(display_name)
        self.audio_worker.set_wakeword(display_name)
        if self._wakeword_testbed_dialog is not None:
            self._wakeword_testbed_dialog.on_wakeword_selection_changed()

    def _resolve_active_input_device_index(self) -> int | None:
        saved = get_audio_input_device_index()
        if saved is not None:
            return saved
        worker = getattr(self, "audio_worker", None)
        if worker is not None:
            worker_idx = getattr(worker, "input_device_index", None)
            if worker_idx is not None:
                return int(worker_idx)
        return None

    def _resolve_active_output_device_index(self) -> int | None:
        saved = get_audio_output_device_index()
        if saved is not None:
            return saved
        worker = getattr(self, "tts_worker", None)
        if worker is not None:
            worker_idx = getattr(worker, "current_device_index", None)
            if worker_idx is not None:
                return int(worker_idx)
        return None

    @staticmethod
    def _device_name_for_index(
        devices: list[tuple[int, str]],
        device_index: int | None,
    ) -> str | None:
        if device_index is None:
            return None
        for idx, name in devices:
            if idx == device_index:
                return name
        return None

    def _build_refreshing_audio_device_menu(
        self,
        button,
        *,
        list_devices: Callable[[], list[tuple[int, str]]],
        resolve_active_index: Callable[[], int | None],
        on_selected: Callable[[int], None],
        empty_label: str,
    ) -> None:
        menu = QMenu(button)
        menu.setObjectName("PrestigeMenu")
        menu.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        is_dark = getattr(self.window(), "_is_dark_theme", True)
        self._apply_menu_theme(menu, is_dark)

        list_widget = QListWidget()
        list_widget.setObjectName("PrestigeMenuList")
        list_widget.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        def refresh_menu() -> None:
            devices = list_devices()
            active_idx = resolve_active_index()
            list_widget.clear()
            for idx, label in build_audio_device_menu_rows(devices, active_idx):
                row = QListWidgetItem(label)
                row.setData(Qt.ItemDataRole.UserRole, idx)
                list_widget.addItem(row)

            if not devices:
                row = QListWidgetItem(empty_label)
                row.setFlags(Qt.ItemFlag.NoItemFlags)
                list_widget.addItem(row)

            required_height = max(1, list_widget.count()) * 32 + 10
            main_win = self.window()
            max_height = int(main_win.height() * 0.5) if main_win else 400
            list_widget.setFixedHeight(min(required_height, max_height))

        def sync_dropdown_width() -> None:
            content_w = list_widget.sizeHintForColumn(0) + 40
            list_widget.setFixedWidth(max(button.width() - 8, content_w, 220))

        def on_item_clicked(item: QListWidgetItem) -> None:
            idx = item.data(Qt.ItemDataRole.UserRole)
            if idx is None:
                return
            device_index = int(idx)
            name = self._device_name_for_index(list_devices(), device_index)
            on_selected(device_index)
            if name:
                from ui.views.settings.widgets import refit_settings_selector_width

                button.setText(name)
                refit_settings_selector_width(button)
                button.update()
            menu.hide()

        menu.aboutToShow.connect(refresh_menu)
        menu.aboutToShow.connect(sync_dropdown_width)
        list_widget.itemClicked.connect(on_item_clicked)

        action = QWidgetAction(menu)
        action.setDefaultWidget(list_widget)
        menu.addAction(action)
        button.setMenu(menu)

    def _sync_input_device_selector_label(self) -> None:
        if not hasattr(self, "mic_selector"):
            return
        mics = get_input_devices()
        active_name = self._device_name_for_index(
            mics,
            self._resolve_active_input_device_index(),
        )
        if active_name:
            self.mic_selector.setText(active_name)
        elif mics:
            self.mic_selector.setText(mics[0][1])
        else:
            self.mic_selector.setText("Select Input Device...")

    def _sync_output_device_selector_label(self) -> None:
        if not hasattr(self, "device_selector"):
            return
        outputs = get_output_devices()
        active_name = self._device_name_for_index(
            outputs,
            self._resolve_active_output_device_index(),
        )
        if active_name:
            self.device_selector.setText(active_name)
        elif outputs:
            self.device_selector.setText(outputs[0][1])
        else:
            self.device_selector.setText("Select Output Device...")

    def _populate_audio_device_selectors(self) -> None:
        if getattr(self, "_audio_device_selectors_populated", False):
            return
        if not hasattr(self, "mic_selector"):
            return

        self._build_refreshing_audio_device_menu(
            self.mic_selector,
            list_devices=get_input_devices,
            resolve_active_index=self._resolve_active_input_device_index,
            on_selected=self._on_input_device_selected,
            empty_label="No microphones found",
        )
        saved_input_idx = get_audio_input_device_index()
        if saved_input_idx is not None and self.audio_worker:
            self.audio_worker.set_input_device(saved_input_idx)
        self._sync_input_device_selector_label()

        self._build_refreshing_audio_device_menu(
            self.device_selector,
            list_devices=get_output_devices,
            resolve_active_index=self._resolve_active_output_device_index,
            on_selected=self._on_output_device_selected,
            empty_label="No output devices found",
        )
        saved_output_idx = get_audio_output_device_index()
        if saved_output_idx is not None and self.tts_worker:
            self.tts_worker.set_device(saved_output_idx)
        self._sync_output_device_selector_label()

        if self.audio_worker and hasattr(self, "wakeword_selector"):
            self.wakeword_selector.pressed.connect(self._on_wakeword_selector_pressed)
            self._sync_wakeword_catalog(trigger="settings load")

        self._sync_tts_voice_controls_state()
        self._audio_device_selectors_populated = True

    def _populate_engine_selectors(self) -> None:
        if getattr(self, "_engine_selectors_populated", False):
            return
        if not hasattr(self, "engine_selector"):
            return
        from ui.views.settings.widgets import register_settings_selector_width, refit_settings_selector_width

        engine_modes = [
            ("Internal Engine (native)", "internal"),
            ("External Server (localhost)", "external"),
        ]
        register_settings_selector_width(
            self.engine_selector,
            *(label for label, _mode in engine_modes),
        )
        self._build_prestige_menu(
            self.engine_selector,
            engine_modes,
            lambda mode: self.engine_mode_changed.emit(str(mode)),
        )
        em = get_engine_mode()
        engine_label = next((lbl for lbl, m in engine_modes if m == em), engine_modes[0][0])
        self.engine_selector.setText(engine_label)

        if hasattr(self, "provider_selector"):
            providers = [("Ollama (Port 11434)", 11434), ("LM Studio (Port 1234)", 1234)]
            register_settings_selector_width(
                self.provider_selector,
                *(label for label, _port in providers),
            )
            self._build_prestige_menu(
                self.provider_selector,
                providers,
                lambda port: self.llm_worker.set_provider(port) if self.llm_worker else None,
            )

            if is_port_open(1234):
                self.provider_selector.setText("LM Studio (Port 1234)")
            elif is_port_open(11434):
                self.provider_selector.setText("Ollama (Port 11434)")

        refit_settings_selector_width(self.engine_selector)
        refit_settings_selector_width(self.provider_selector)
        self._sync_ai_provider_enabled_for_inference(get_engine_mode())
        self._engine_selectors_populated = True

    def _populate_hardware_selectors(self):
        self._populate_audio_device_selectors()
        self._populate_engine_selectors()

    def _on_input_device_selected(self, idx: int) -> None:
        set_audio_input_device_index(idx)
        if self.audio_worker:
            self.audio_worker.set_input_device(idx)

    def _on_output_device_selected(self, idx: int) -> None:
        set_audio_output_device_index(idx)
        if self.tts_worker:
            self.tts_worker.set_device(idx)

    def _on_audio_input_hint_clicked(self) -> None:
        self.mic_vu_hint_requested.emit()

    def _tts_engine_ready(self) -> bool:
        return bool(
            self.tts_worker and getattr(self.tts_worker, "active_adapter", None)
        )

    def _voice_preview_unavailable_message(self) -> str:
        worker = getattr(self, "tts_worker", None)
        last_error = getattr(worker, "_last_load_error", None) if worker else None
        if last_error:
            return (
                "Qube could not load the text-to-speech model:\n\n"
                f"{last_error}"
            )

        on_disk, detail = describe_tts_model_disk_state()
        if on_disk:
            return (
                "A text-to-speech model is on disk but has not been loaded yet.\n\n"
                "Use Refresh under Text-to-speech in this settings page, or reopen "
                "Voice & Audio, then try preview again."
            )
        if detail:
            return detail
        return "Load a text-to-speech model before previewing voices."

    def _sync_tts_voice_controls_state(self) -> None:
        ready = self._tts_engine_ready()
        placeholder = "Select Voice..." if ready else "Load TTS model first..."

        if hasattr(self, "voice_selector"):
            self.voice_selector.setEnabled(ready)
            if not ready:
                self.voice_selector.setText(placeholder)
                self.voice_selector.setMenu(QMenu(self.voice_selector))
            elif not self.voice_selector.text() or self.voice_selector.text() == placeholder:
                from core.tts_models import resolve_default_tts_voice

                voices = getattr(self.tts_worker, "active_adapter", None)
                if voices is not None:
                    available = getattr(voices, "available_voices", [])
                    if available:
                        active = getattr(
                            self.tts_worker,
                            "active_voice_name",
                            resolve_default_tts_voice(available),
                        )
                        self.voice_selector.setText(active)

        for btn_name in ("tts_voice_preview_btn", "audio_output_preview_btn"):
            btn = getattr(self, btn_name, None)
            if btn is not None:
                btn.setEnabled(ready)

    def _attempt_tts_model_load_if_needed(self) -> None:
        if self._tts_engine_ready():
            return
        if not any_supported_tts_model_on_disk():
            self._sync_tts_voice_controls_state()
            return
        self._reload_tts_from_settings()
        self._sync_tts_voice_controls_state()

    def _play_tts_voice_preview(self) -> None:
        if not self._tts_engine_ready():
            self._attempt_tts_model_load_if_needed()
        if not self._tts_engine_ready():
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            PrestigeDialog(
                self.window(),
                "Voice preview unavailable",
                self._voice_preview_unavailable_message(),
                is_dark=is_dark,
            ).exec()
            return
        phrase, next_index = next_tts_voice_preview_phrase(
            getattr(self, "_tts_voice_preview_phrase_index", 0)
        )
        self._tts_voice_preview_phrase_index = next_index
        self.tts_worker.queue_voice_preview(phrase)

    def _show_custom_model_paths_license_dialog(self) -> None:
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        PrestigeDialog(
            self.window(),
            "Pro license required",
            CUSTOM_MODEL_PATHS_LICENSE_MESSAGE,
            is_dark=is_dark,
        ).exec()

    def _sync_custom_model_paths_pro_features(self) -> None:
        changed = sync_custom_model_paths_pro_features(self)
        if not changed:
            return
        if hasattr(self, "_reload_stt_from_settings"):
            self._reload_stt_from_settings()
        if hasattr(self, "_reload_tts_from_settings"):
            self._reload_tts_from_settings()
        if hasattr(self, "_reload_embedder_from_settings"):
            self._reload_embedder_from_settings()

    def _on_advanced_stt_toggled(self, checked: bool) -> None:
        if checked and not user_has_pro_custom_model_paths():
            self._show_custom_model_paths_license_dialog()
            self._sync_custom_model_paths_pro_features()
            return
        if checked:
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            dlg = PrestigeDialog(
                self.window(),
                "Advanced STT settings",
                "Swapping the speech-to-text model affects voice input transcription.\n\n"
                "Place CTranslate2 Whisper folders (with model.bin) under models/stt/. "
                "The bundled Whisper small default cannot be deleted.\n\nContinue?",
                is_dark=is_dark,
                tone="danger",
                dialog_width=450,
            )
            if not dlg.exec():
                self.advanced_stt_toggle.blockSignals(True)
                self.advanced_stt_toggle.setChecked(False)
                self.advanced_stt_toggle.blockSignals(False)
                return
        set_advanced_stt_unlocked(bool(checked and user_has_pro_custom_model_paths()))
        self._apply_advanced_stt_panel_visibility()

    def _on_advanced_tts_toggled(self, checked: bool) -> None:
        if checked and not user_has_pro_custom_model_paths():
            self._show_custom_model_paths_license_dialog()
            self._sync_custom_model_paths_pro_features()
            return
        if checked:
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            dlg = PrestigeDialog(
                self.window(),
                "Advanced TTS settings",
                "Swapping the text-to-speech model affects spoken replies.\n\n"
                "Supported engines: Kokoro ONNX (default) and Piper ONNX. Place .onnx files "
                "under models/tts/ — Kokoro also needs voices-v1.0.bin in the same folder; "
                "Piper needs a sibling .onnx.json config (or \"piper\" in the filename).\n\n"
                "The bundled Kokoro v1.0 default cannot be deleted.\n\n"
                "Continue?",
                is_dark=is_dark,
                tone="danger",
                dialog_width=450,
            )
            if not dlg.exec():
                self.advanced_tts_toggle.blockSignals(True)
                self.advanced_tts_toggle.setChecked(False)
                self.advanced_tts_toggle.blockSignals(False)
                return
        set_advanced_tts_unlocked(bool(checked and user_has_pro_custom_model_paths()))
        self._apply_advanced_tts_panel_visibility()

    def _apply_advanced_stt_panel_visibility(self) -> None:
        unlocked = effective_advanced_stt_unlocked()
        visible = unlocked or getattr(self, "_tour_stt_preview_active", False)
        if hasattr(self, "advanced_stt_panel"):
            self.advanced_stt_panel.setVisible(visible)
        if hasattr(self, "advanced_stt_toggle"):
            self.advanced_stt_toggle.blockSignals(True)
            self.advanced_stt_toggle.setChecked(
                True if getattr(self, "_tour_stt_preview_active", False) else unlocked
            )
            self.advanced_stt_toggle.blockSignals(False)

    def _apply_advanced_tts_panel_visibility(self) -> None:
        unlocked = effective_advanced_tts_unlocked()
        visible = unlocked or getattr(self, "_tour_tts_preview_active", False)
        if hasattr(self, "advanced_tts_panel"):
            self.advanced_tts_panel.setVisible(visible)
        if hasattr(self, "advanced_tts_toggle"):
            self.advanced_tts_toggle.blockSignals(True)
            self.advanced_tts_toggle.setChecked(
                True if getattr(self, "_tour_tts_preview_active", False) else unlocked
            )
            self.advanced_tts_toggle.blockSignals(False)

    def begin_voice_audio_stt_tutorial_preview(self) -> None:
        """Reveal advanced STT controls during the Voice & Audio guided tour."""
        self._tour_stt_preview_active = True
        self._apply_advanced_stt_panel_visibility()

    def end_voice_audio_stt_tutorial_preview(self) -> None:
        """Restore advanced STT panel visibility after the guided tour."""
        if not getattr(self, "_tour_stt_preview_active", False):
            return
        self._tour_stt_preview_active = False
        self._apply_advanced_stt_panel_visibility()

    def begin_voice_audio_tts_tutorial_preview(self) -> None:
        """Reveal advanced TTS controls during the Voice & Audio guided tour."""
        self._tour_tts_preview_active = True
        self._apply_advanced_tts_panel_visibility()

    def end_voice_audio_tts_tutorial_preview(self) -> None:
        """Restore advanced TTS panel visibility after the guided tour."""
        if not getattr(self, "_tour_tts_preview_active", False):
            return
        self._tour_tts_preview_active = False
        self._apply_advanced_tts_panel_visibility()

    def _sync_stt_models_dir_label(self) -> None:
        if hasattr(self, "stt_dir_label"):
            self.stt_dir_label.setText(get_stt_models_dir())

    def _sync_tts_models_dir_label(self) -> None:
        if hasattr(self, "tts_dir_label"):
            self.tts_dir_label.setText(get_tts_models_dir())

    def _refresh_stt_model_list(self) -> None:
        if not hasattr(self, "stt_model_list"):
            return
        self.stt_model_list.clear()
        active = resolve_active_stt_model_spec()
        for entry in list_selectable_stt_models():
            item = QListWidgetItem(entry.display_name)
            item.setData(Qt.ItemDataRole.UserRole, entry.path)
            item.setData(SPEECH_ENTRY_DELETABLE_ROLE, entry.is_deletable)
            self.stt_model_list.addItem(item)
            if entry.path == active or (
                entry.is_bundled_default and active == BUNDLED_STT_MODEL_ID
            ):
                self.stt_model_list.setCurrentItem(item)

    def _refresh_tts_model_list(self) -> None:
        if not hasattr(self, "tts_model_list"):
            return
        self.tts_model_list.clear()
        active = resolve_active_tts_path()
        try:
            active_norm = str(Path(active).resolve()) if active else ""
        except OSError:
            active_norm = active or ""
        for entry in list_selectable_tts_models():
            item = QListWidgetItem(entry.display_name)
            item.setData(Qt.ItemDataRole.UserRole, entry.path)
            item.setData(SPEECH_ENTRY_DELETABLE_ROLE, entry.is_deletable)
            self.tts_model_list.addItem(item)
            try:
                if active_norm and str(Path(entry.path).resolve()) == active_norm:
                    self.tts_model_list.setCurrentItem(item)
            except OSError:
                if entry.path == active:
                    self.tts_model_list.setCurrentItem(item)

    def _sync_active_stt_label(self) -> None:
        if not hasattr(self, "active_stt_model_lbl"):
            return
        spec = resolve_active_stt_model_spec()
        if is_protected_stt_model(spec):
            self.active_stt_model_lbl.setText(f"{BUNDLED_STT_MODEL_ID} (bundled default)")
        elif spec and os.path.isdir(spec):
            self.active_stt_model_lbl.setText(f"{os.path.basename(spec)} (custom)")
        else:
            self.active_stt_model_lbl.setText("— (bundled default missing)")

    def _sync_active_tts_label(self) -> None:
        if not hasattr(self, "active_tts_model_lbl"):
            return
        path = resolve_active_tts_path()
        if not path or not os.path.isfile(path):
            self.active_tts_model_lbl.setText("— (bundled default missing)")
            return
        base = os.path.basename(path)
        if is_protected_tts_model(path):
            self.active_tts_model_lbl.setText(f"{base} (bundled default)")
        else:
            self.active_tts_model_lbl.setText(f"{base} (custom)")

    def _on_refresh_stt_models_clicked(self) -> None:
        self._sync_stt_models_dir_label()
        self._refresh_stt_model_list()
        self._sync_active_stt_label()

    def _on_refresh_tts_models_clicked(self) -> None:
        migrate_legacy_tts_layout()
        migrate_stale_tts_override()
        self._sync_tts_models_dir_label()
        self._refresh_tts_model_list()
        self._sync_active_tts_label()
        self._reload_tts_from_settings()
        self._sync_tts_voice_controls_state()

    def _reload_stt_from_settings(self) -> None:
        self.stt_model_changed.emit()

    def _reload_tts_from_settings(self) -> None:
        self.tts_model_changed.emit()

    def _apply_selected_stt_model(self) -> None:
        item = self.stt_model_list.currentItem()
        if not item:
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            PrestigeDialog(
                self.window(),
                "No model",
                "Select an STT model from the list.",
                is_dark=is_dark,
            ).exec()
            return
        spec = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if is_protected_stt_model(spec):
            set_stt_model_path("")
        else:
            if not user_has_pro_custom_model_paths():
                self._show_custom_model_paths_license_dialog()
                return
            ok, msg = validate_stt_model_path(spec)
            if not ok:
                is_dark = getattr(self.window(), "_is_dark_theme", True)
                PrestigeDialog(
                    self.window(),
                    "Invalid STT model",
                    msg or "That folder cannot be used as the STT model.",
                    is_dark=is_dark,
                ).exec()
                return
            set_stt_model_path(spec)
        self._sync_active_stt_label()
        self._reload_stt_from_settings()

    def _reset_stt_to_default(self) -> None:
        set_stt_model_path("")
        self._refresh_stt_model_list()
        self._sync_active_stt_label()
        self._reload_stt_from_settings()

    def _delete_selected_stt_model(self) -> None:
        item = self.stt_model_list.currentItem()
        if not item:
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            PrestigeDialog(
                self.window(),
                "No model",
                "Select an STT model folder to delete.",
                is_dark=is_dark,
            ).exec()
            return
        path = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if is_protected_stt_model(path):
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            PrestigeDialog(
                self.window(),
                "Protected model",
                "The bundled Whisper small default cannot be deleted.",
                is_dark=is_dark,
            ).exec()
            return
        if not item.data(SPEECH_ENTRY_DELETABLE_ROLE):
            return
        if not path or not os.path.isdir(path):
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            PrestigeDialog(
                self.window(),
                "Missing folder",
                "That folder is not available on disk.",
                is_dark=is_dark,
            ).exec()
            return
        name = os.path.basename(path)
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        dlg = PrestigeDialog(
            self.window(),
            "Delete STT model",
            f'Permanently delete the folder "{name}" from models/stt/? '
            "This cannot be undone.",
            is_dark=is_dark,
        )
        if not dlg.exec():
            return
        try:
            shutil.rmtree(path)
        except OSError as e:
            logger.error("Failed to delete STT model folder %s: %s", path, e)
            PrestigeDialog(
                self.window(),
                "Delete failed",
                str(e),
                is_dark=is_dark,
            ).exec()
            return

        active = resolve_active_stt_model_spec()
        try:
            was_active = str(Path(active).resolve()) == str(Path(path).resolve())
        except OSError:
            was_active = active == path
        if was_active:
            set_stt_model_path("")
            self._reload_stt_from_settings()
        self._sync_active_stt_label()
        self._refresh_stt_model_list()

    def _load_tts_model_or_dialog(self, load_path: str, *, failure_title: str = "TTS model failed to load") -> bool:
        """Try loading ``load_path`` on the TTS worker; show a dialog when load fails."""
        if not self.tts_worker:
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            PrestigeDialog(
                self.window(),
                "TTS unavailable",
                "The text-to-speech worker is not available.",
                is_dark=is_dark,
            ).exec()
            return False
        if self.tts_worker.load_voice(load_path):
            return True
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        PrestigeDialog(
            self.window(),
            failure_title,
            "The model could not be loaded. Spoken replies are unchanged.\n\n"
            "Qube supports Kokoro and Piper ONNX only. Piper models need a sibling "
            ".onnx.json file. Use Reset to default to return to Kokoro.",
            is_dark=is_dark,
            tone="danger",
            dialog_width=450,
        ).exec()
        return False

    def _apply_selected_tts_model(self) -> None:
        item = self.tts_model_list.currentItem()
        if not item:
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            PrestigeDialog(
                self.window(),
                "No model",
                "Select a TTS model from the list.",
                is_dark=is_dark,
            ).exec()
            return
        path = str(item.data(Qt.ItemDataRole.UserRole) or "")
        is_default = is_protected_tts_model(path)
        load_path = bundled_default_path() if is_default else path

        if not is_default:
            if not user_has_pro_custom_model_paths():
                self._show_custom_model_paths_license_dialog()
                return
            ok, msg = validate_tts_model_path(path)
            if not ok:
                is_dark = getattr(self.window(), "_is_dark_theme", True)
                PrestigeDialog(
                    self.window(),
                    "Invalid TTS model",
                    msg or "That file cannot be used as the TTS model.",
                    is_dark=is_dark,
                ).exec()
                return
            name = os.path.basename(path)
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            dlg = PrestigeDialog(
                self.window(),
                "Switch TTS model",
                f'Switch spoken replies to "{name}"?\n\n'
                "Only Kokoro and Piper ONNX models are supported. If speech stops working, "
                "open Settings → Voice & Audio and choose Reset to default.",
                is_dark=is_dark,
                tone="danger",
                dialog_width=450,
            )
            if not dlg.exec():
                return

        if not self._load_tts_model_or_dialog(load_path):
            return

        set_tts_model_path("" if is_default else path)
        self._sync_active_tts_label()
        self._refresh_tts_model_list()

    def _reset_tts_to_default(self) -> None:
        load_path = bundled_default_path()
        if not os.path.isfile(load_path):
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            PrestigeDialog(
                self.window(),
                "Kokoro not installed",
                "The bundled Kokoro model is missing. Download it from "
                "Settings → Voice & Audio → Text-to-speech (TTS) before resetting.",
                is_dark=is_dark,
            ).exec()
            return
        if not self._load_tts_model_or_dialog(load_path, failure_title="Reset failed"):
            return
        set_tts_model_path("")
        self._refresh_tts_model_list()
        self._sync_active_tts_label()

    def _delete_selected_tts_model(self) -> None:
        item = self.tts_model_list.currentItem()
        if not item:
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            PrestigeDialog(
                self.window(),
                "No model",
                "Select a TTS model to delete.",
                is_dark=is_dark,
            ).exec()
            return
        path = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if is_protected_tts_model(path):
            is_dark = getattr(self.window(), "_is_dark_theme", True)
            PrestigeDialog(
                self.window(),
                "Protected model",
                "The bundled Kokoro v1.0 default cannot be deleted.",
                is_dark=is_dark,
            ).exec()
            return
        if not item.data(SPEECH_ENTRY_DELETABLE_ROLE):
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
            "Delete TTS model",
            f'Permanently delete "{name}" from models/tts/? This cannot be undone.',
            is_dark=is_dark,
        )
        if not dlg.exec():
            return
        paths_to_remove = [path]
        json_sidecar = path + ".json"
        if os.path.isfile(json_sidecar):
            paths_to_remove.append(json_sidecar)
        for target in paths_to_remove:
            try:
                os.remove(target)
            except OSError as e:
                logger.error("Failed to delete TTS file %s: %s", target, e)
                PrestigeDialog(
                    self.window(),
                    "Delete failed",
                    str(e),
                    is_dark=is_dark,
                ).exec()
                return

        active = resolve_active_tts_path()
        try:
            was_active = str(Path(active).resolve()) == str(Path(path).resolve())
        except OSError:
            was_active = active == path
        if was_active:
            set_tts_model_path("")
            from core.tts_models import resolve_boot_tts_path

            self._load_tts_model_or_dialog(
                resolve_boot_tts_path(),
                failure_title="TTS reload failed",
            )
        self._sync_active_tts_label()
        self._refresh_tts_model_list()
