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
from PyQt6.QtGui import QFontMetrics, QResizeEvent, QShowEvent, QHideEvent, QPainter, QColor, QPixmap

from core.paths import resource_path

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
from core.embedding_models import get_embedding_models_dir
from core.stt_models import get_stt_models_dir
from core.tts_models import get_tts_models_dir, migrate_legacy_tts_layout
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
from ui.onboarding.settings_tour_header import sync_settings_section_tour_header
from ui.views.settings.registry import SETTINGS_SECTIONS, resolve_section_id, get_section, resolve_settings_navigation
from ui.views.settings.widgets import (
    collect_theme_buttons,
    make_settings_section_header_row,
    SettingsSectionHeaderBar,
    SettingsSectionPane,
)
from ui.views.settings.settings_card_style import refresh_settings_section_cards
from ui.views.settings.settings_theme import (
    resolve_settings_theme,
    settings_chevron_color,
    settings_hint_icon_color,
    settings_info_icon_color,
    settings_nav_icon_color,
    settings_preview_icon_color,
    style_bootstrap_warning_label,
)
from core.theme.svg_icons import tinted_svg_pixmap, themed_fa_pixmap
from core.theme.widget_styles import (
    SETTINGS_GHOST_TOOL_BUTTON,
    SETTINGS_HINT,
    SETTINGS_ICON_BUTTON,
)
from ui.views.settings.sections import (
    advanced,
    ai_models,
    appearance_themes,
    contact_feedback,
    desktop_companion,
    diagnostics,
    general,
    help,
    integrations,
    knowledge,
    license_section,
    memory,
    notifications,
    privacy_data,
    voice_audio,
)

from ui.views.settings.handlers import (
    AiModelsHandlersMixin,
    BootstrapDownloadsHandlersMixin,
    CompanionHandlersMixin,
    DiagnosticsHandlersMixin,
    GenerationMixin,
    KnowledgeHandlersMixin,
    LicenseHandlersMixin,
    UninstallHandlersMixin,
    MemoryHandlersMixin,
    PersistenceHandlersMixin,
    PrestigeMenuMixin,
    PrivacyDataHandlersMixin,
    StylingMixin,
    SupportHandlersMixin,
    ThemesHandlersMixin,
    UpdateHandlersMixin,
    VoiceHandlersMixin,
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
    "appearance.themes": appearance_themes.build_section,
    "companion.desktop": desktop_companion.build_section,
    "notifications": notifications.build_section,
    "help": help.build_section,
    "contact.feedback": contact_feedback.build_section,
    "privacy.data": privacy_data.build_section,
    "diagnostics": diagnostics.build_section,
    "license": license_section.build_section,
    "advanced": advanced.build_section,
}


def _resolve_collapsible_handle_near(widget: QWidget):
    """Find the collapsible card handle for an anchored subsection wrapper."""
    from ui.views.settings.settings_card_style import SettingsCollapsibleCardHandle

    node = widget
    for _ in range(12):
        if node is None:
            break
        handle = getattr(node, "_settings_collapsible_handle", None)
        if isinstance(handle, SettingsCollapsibleCardHandle):
            return handle
        title_lbl = getattr(node, "title_lbl", None)
        if title_lbl is not None:
            handle = getattr(title_lbl, "_settings_collapsible_handle", None)
            if handle is not None:
                return handle
        node = node.parentWidget()
    return None


def _find_settings_anchor_target(page: QWidget, anchor: str):
    """Resolve the best scroll target for a settings anchor (prefer whole card wrapper)."""
    from ui.views.settings.settings_card_style import SettingsCollapsibleCardHandle

    want = (anchor or "").strip()
    if not want:
        return None, None

    for lbl in page.findChildren(QLabel):
        if lbl.property("settings_anchor") != want:
            continue
        handle = getattr(lbl, "_settings_collapsible_handle", None)
        if isinstance(handle, SettingsCollapsibleCardHandle):
            return handle.wrapper, handle
        return lbl, None

    for wrapper in page.findChildren(QWidget):
        if wrapper.property("settings_anchor") != want:
            continue
        handle = _resolve_collapsible_handle_near(wrapper)
        if handle is not None:
            return handle.wrapper, handle
        return wrapper, None

    return None, None


def _scroll_settings_target_into_view(
    scroll: QScrollArea,
    target: QWidget,
    *,
    top_margin: int = 72,
) -> None:
    """Scroll a settings page so ``target`` sits near the top of the viewport."""
    content = scroll.widget()
    if content is None or target is None:
        return
    scroll.ensureWidgetVisible(target, 0, top_margin)
    pos = target.mapTo(content, target.rect().topLeft())
    bar = scroll.verticalScrollBar()
    if bar is not None:
        y = max(bar.minimum(), min(bar.maximum(), pos.y() - top_margin))
        bar.setValue(y)
    scroll.ensureWidgetVisible(target, 0, top_margin)


class SettingsView(
    QWidget,
    PrestigeMenuMixin,
    GenerationMixin,
    StylingMixin,
    VoiceHandlersMixin,
    AiModelsHandlersMixin,
    BootstrapDownloadsHandlersMixin,
    MemoryHandlersMixin,
    KnowledgeHandlersMixin,
    LicenseHandlersMixin,
    CompanionHandlersMixin,
    DiagnosticsHandlersMixin,
    PrivacyDataHandlersMixin,
    PersistenceHandlersMixin,
    SupportHandlersMixin,
    UpdateHandlersMixin,
    UninstallHandlersMixin,
    ThemesHandlersMixin,
):

    audio_pin_toggle = pyqtSignal(bool)
    tts_voice_pin_toggle = pyqtSignal(bool)
    auto_activator_toggle = pyqtSignal(bool) # 🔑 ADD THIS
    rag_kb_toggle = pyqtSignal(bool)
    auto_load_last_model_changed = pyqtSignal(bool)
    memory_enrichment_changed = pyqtSignal(bool)
    memory_promotion_changed = pyqtSignal(bool)
    memory_consolidation_changed = pyqtSignal(bool)
    engine_mode_changed = pyqtSignal(str)
    external_settings_reloaded = pyqtSignal(set)
    ui_language_changed = pyqtSignal()
    cognition_model_changed = pyqtSignal()
    embedding_model_changed = pyqtSignal()
    embedding_mode_change_requested = pyqtSignal(str, str)
    stt_model_changed = pyqtSignal()
    tts_model_changed = pyqtSignal()
    mic_vu_hint_requested = pyqtSignal()
    def __init__(self, workers: dict, db_manager, parent=None):
        super().__init__(parent)
        self.workers = workers
        self.db = db_manager
        
        self.audio_worker = workers.get("audio")
        self.tts_worker = workers.get("tts")
        self.llm_worker = workers.get("llm")
        self._template_override_reload_pending = False
        self._auto_reset_reload_pending = False
        self._companion_verbal_test_worker = None
        self._voice_section_data_loaded = False
        self._ai_models_section_data_loaded = False
        self._knowledge_section_data_loaded = False
        self._skip_next_menu_theme_refresh = False
        self._sections_built: set[str] = set()
        self._section_content_hosts: dict[str, QWidget] = {}
        self._settings_prefetch_queue: list[str] = []

        self._setup_ui()
        self._ensure_rag_toolbar_controls()
        self.engine_mode_changed.connect(self._sync_ai_provider_enabled_for_inference)
        self.engine_mode_changed.connect(lambda _mode: self._sync_native_chat_template_label())
        self.engine_mode_changed.connect(self._sync_internal_engine_subsections)
        native_engine = self.workers.get("native_engine")
        if native_engine is not None and hasattr(native_engine, "load_finished"):
            native_engine.load_finished.connect(self._on_native_model_load_finished)
        QTimer.singleShot(0, self._populate_audio_device_selectors)
        os.makedirs(get_llm_models_dir(), exist_ok=True)
        os.makedirs(get_embedding_models_dir(), exist_ok=True)
        migrate_legacy_tts_layout()
        self._wakeword_testbed_dialog = None
        self._settings_json_dialog: SettingsJsonEditorDialog | None = None
        self._setup_settings_file_watcher()
        self._skip_next_menu_theme_refresh = True
        self._schedule_settings_section_prefetch()
    def select_settings_section(
        self,
        section: str,
        *,
        anchor: str | None = None,
        configure_provider_id: str | None = None,
    ) -> None:
        """Show a settings section by stable id, title, or legacy title."""
        section_id, anchor = resolve_settings_navigation(section, anchor=anchor)
        if section_id is None:
            return
        previous_section = getattr(self, "_settings_active_section_id", None)
        self._ensure_section_built(section_id)
        row = self._section_row_by_id.get(section_id)
        if row is not None:
            self.settings_section_list.setCurrentRow(row)
        if anchor:
            # Defer slightly when switching sections so stack + layout settle first.
            delay_ms = 50 if previous_section and previous_section != section_id else 0
            cross_section = bool(previous_section and previous_section != section_id)

            def _scroll(a=anchor, retry=cross_section) -> None:
                self._scroll_to_settings_anchor(a, retry=retry)

            QTimer.singleShot(delay_ms, _scroll)
        if configure_provider_id:
            pid = str(configure_provider_id).strip().lower()

            def _open_configure() -> None:
                is_dark = getattr(self.window(), "_is_dark_theme", True)
                from ui.components.provider_credential_dialog import (
                    open_provider_credential_dialog,
                )

                open_provider_credential_dialog(
                    self,
                    pid,
                    is_dark=is_dark,
                    parent=self.window(),
                )

            QTimer.singleShot(120, _open_configure)

    def _ensure_section_data_loaded(self, section_id: str, *, force: bool = False) -> None:
        """Populate model lists and selectors the first time a section is shown."""
        if section_id == "voice.audio":
            if not force and self._voice_section_data_loaded:
                return
            self._voice_section_data_loaded = True
            self._refresh_stt_model_list()
            self._refresh_tts_model_list()
            self._sync_active_stt_label()
            self._sync_active_tts_label()
            return

        if section_id == "ai.models":
            if not force and self._ai_models_section_data_loaded:
                return
            self._ai_models_section_data_loaded = True
            self._refresh_local_gguf_list()
            self._sync_active_native_model_label()
            return

        if section_id == "knowledge":
            if not force and self._knowledge_section_data_loaded:
                return
            self._knowledge_section_data_loaded = True
            self._refresh_embedding_gguf_list()
            self._sync_active_embedding_label()
            if hasattr(self, "_sync_embedding_mode_selector"):
                self._sync_embedding_mode_selector()

    def _scroll_to_settings_anchor(self, anchor: str, *, retry: bool = False) -> None:
        self._apply_settings_anchor_scroll(anchor)
        if retry:
            QTimer.singleShot(120, lambda: self._apply_settings_anchor_scroll(anchor))

    def _apply_settings_anchor_scroll(self, anchor: str) -> bool:
        scroll = self.settings_section_stack.currentWidget()
        if scroll is None or not isinstance(scroll, QScrollArea):
            return False
        page = scroll.widget()
        if page is None:
            return False
        target, handle = _find_settings_anchor_target(page, anchor)
        if target is None:
            return False
        if handle is not None:
            handle.set_expanded(True)
        _scroll_settings_target_into_view(scroll, target)
        return True

    def _maybe_start_provider_status_refresh(self) -> None:
        row = self.settings_section_list.currentRow()
        if row < 0:
            return
        item = self.settings_section_list.item(row)
        if item is None:
            return
        section_id = item.data(self._SETTINGS_SECTION_ID_ROLE)
        if section_id != "knowledge":
            return
        from ui.views.settings.sections.knowledge_provider_status import (
            start_provider_status_refresh_timer,
        )

        start_provider_status_refresh_timer(self)

    def _refresh_knowledge_access_ui(self, *, is_dark: bool | None = None) -> None:
        """Re-style Knowledge live-source badges/buttons from the active window theme."""
        from ui.views.settings.knowledge_access_badge import coalesce_settings_is_dark

        resolved = coalesce_settings_is_dark(self, is_dark=is_dark)
        if hasattr(self, "knowledge_live_source_rows"):
            from ui.views.settings.sections.knowledge_sources import (
                refresh_live_source_access_badges,
            )

            refresh_live_source_access_badges(self, is_dark=resolved)
        if hasattr(self, "web_discovery_policy_section"):
            from ui.views.settings.sections.knowledge_web_discovery import (
                sync_web_discovery_policy_section,
            )

            sync_web_discovery_policy_section(self, is_dark=resolved)
        if hasattr(self, "knowledge_provider_status_table"):
            from ui.views.settings.sections.knowledge_provider_status import (
                sync_provider_status_panel,
            )

            sync_provider_status_panel(self, is_dark=resolved)

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self._sync_active_native_model_label()
        self._sync_native_chat_template_label()
        if hasattr(self, "_refresh_inference_transparency_panel"):
            self._refresh_inference_transparency_panel()
        self._ensure_settings_file_watched()
        from ui.views.settings.knowledge_access_badge import coalesce_settings_is_dark

        is_dark = coalesce_settings_is_dark(self)
        self._apply_settings_sidebar_surface(is_dark)
        QTimer.singleShot(0, self._relayout_trigger_list_rows)
        if hasattr(self, "_sync_bootstrap_download_visibility"):
            self._sync_bootstrap_download_visibility()
        self._maybe_start_provider_status_refresh()
        self._refresh_knowledge_access_ui(is_dark=is_dark)
        row = self.settings_section_list.currentRow()
        if row >= 0:
            item = self.settings_section_list.item(row)
            if item is not None:
                section_id = item.data(self._SETTINGS_SECTION_ID_ROLE)
                self._sync_settings_collapse_all_button(
                    str(section_id) if section_id else None
                )

    def hideEvent(self, event: QHideEvent) -> None:
        super().hideEvent(event)
        from ui.views.settings.sections.knowledge_provider_status import (
            stop_provider_status_refresh_timer,
        )

        stop_provider_status_refresh_timer(self)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        self._apply_settings_sidebar_surface(is_dark)
        self._relayout_trigger_list_rows()
    def _setup_ui(self):
        from ui.views.settings.knowledge_access_badge import coalesce_settings_is_dark

        is_dark = coalesce_settings_is_dark(self)
        self._section_index_by_id: dict[str, int] = {}
        self._section_row_by_id: dict[str, int] = {}
        self._section_stack_index_by_id: dict[str, int] = {}
        self._section_scroll_by_id: dict[str, QScrollArea] = {}
        self._settings_search_index: list[dict] = []
        self._theme_buttons: list = []
        self._settings_collapsible_cards_by_section: dict[str, list] = {}

        self._init_settings_layout()

        self._section_builders_for_rebuild = _SECTION_BUILDERS

        last_group: str | None = None
        for sec_def in SETTINGS_SECTIONS:
            if sec_def.group and sec_def.group != last_group:
                self._add_settings_group_header(sec_def.group)
                last_group = sec_def.group
            self._register_settings_section(sec_def)

        self._ensure_section_built("voice.audio", is_dark=is_dark)
        self._ensure_section_data_loaded("voice.audio")
        is_dark = coalesce_settings_is_dark(self)
        collect_theme_buttons(self)
        self._finalize_settings_layout(is_dark)
        if hasattr(self, "_sync_bootstrap_download_visibility"):
            self._sync_bootstrap_download_visibility()
        from ui.components.type_to_search import install_type_to_search

        install_type_to_search(self, self.settings_search_input)
    _SETTINGS_STACK_ROLE = int(Qt.ItemDataRole.UserRole)
    _SETTINGS_SECTION_ID_ROLE = int(Qt.ItemDataRole.UserRole) + 1
    def _register_settings_section(self, sec_def) -> None:
        page_content = QWidget()
        page_content.setObjectName("SettingsContent")
        page_content.setMinimumWidth(0)
        page_content.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        page_layout = QVBoxLayout(page_content)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(30)

        content_host = QWidget()
        content_host.setObjectName("SettingsSectionContentHost")
        content_host.setMinimumWidth(0)
        content_host.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        host_layout = QVBoxLayout(content_host)
        host_layout.setContentsMargins(0, 0, 0, 0)
        host_layout.setSpacing(0)
        page_layout.addWidget(content_host)
        page_layout.addStretch()

        scroll = QScrollArea()
        scroll.setObjectName("SettingsScrollArea")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(page_content)

        pane = SettingsSectionPane()
        pane.set_scroll_area(scroll)

        stack_idx = self.settings_section_stack.count()
        self.settings_section_stack.addWidget(pane)
        self._section_stack_index_by_id[sec_def.id] = stack_idx
        self._section_content_hosts[sec_def.id] = content_host
        self._section_scroll_by_id[sec_def.id] = scroll

        item = QListWidgetItem()
        item.setSizeHint(QSize(0, 44))
        item.setData(self._SETTINGS_STACK_ROLE, stack_idx)
        item.setData(self._SETTINGS_SECTION_ID_ROLE, sec_def.id)
        row = self.settings_section_list.count()
        self.settings_section_list.addItem(item)
        nav_row = self._build_settings_section_nav_row(
            sec_def.icon, sec_def.title, svg_icon=sec_def.svg_icon
        )
        if sec_def.id == "license":
            from ui.views.settings.license_status_ui import attach_nav_edition_chip

            self.license_nav_edition_chip = attach_nav_edition_chip(nav_row, host=self)
        self.settings_section_list.setItemWidget(item, nav_row)
        self._section_row_by_id[sec_def.id] = row
        self._index_section_search_base(sec_def)

        if len(self._section_row_by_id) == 1:
            self.settings_section_list.blockSignals(True)
            self.settings_section_list.setCurrentRow(row)
            self.settings_section_stack.setCurrentIndex(stack_idx)
            self.settings_section_list.blockSignals(False)
            self._settings_active_section_id = sec_def.id
            self._sync_settings_section_tour_header(sec_def.id)

    def _ensure_section_built(self, section_id: str, *, is_dark: bool | None = None) -> bool:
        if section_id in self._sections_built:
            return True

        if section_id in self._settings_prefetch_queue:
            self._settings_prefetch_queue = [
                queued_id
                for queued_id in self._settings_prefetch_queue
                if queued_id != section_id
            ]

        builder = _SECTION_BUILDERS.get(section_id)
        content_host = self._section_content_hosts.get(section_id)
        sec_def = get_section(section_id)
        if builder is None or content_host is None or sec_def is None:
            return False

        from ui.views.settings.knowledge_access_badge import coalesce_settings_is_dark

        if is_dark is None:
            is_dark = coalesce_settings_is_dark(self)

        self._current_settings_section_id = section_id
        try:
            content_widget = builder(self, is_dark=is_dark)
        finally:
            self._current_settings_section_id = None
        self._mount_settings_section_content(section_id, content_widget)
        self._sections_built.add(section_id)
        self._index_section_for_search(sec_def, content_widget)
        collect_theme_buttons(self)

        if section_id == "companion.desktop":
            self._wire_companion_cognition_hint()
        if section_id == "ai.models":
            self._sync_internal_engine_subsections(get_engine_mode())
            self._populate_engine_selectors()
            self._sync_models_dir_label()
            self._sync_native_chat_template_label()
            self._sync_active_native_model_label()
            window = self.window()
            if window is not None and hasattr(window, "_wire_generation_settings_toolbar_sync"):
                window._wire_generation_settings_toolbar_sync()
        if section_id == "voice.audio":
            self._sync_stt_models_dir_label()
            self._sync_tts_models_dir_label()
        if section_id == "knowledge":
            self._sync_embedding_models_dir_label()
        if hasattr(self, "_sync_bootstrap_download_visibility"):
            self._sync_bootstrap_download_visibility()
        self._apply_settings_section_control_styles(content_widget, is_dark=is_dark)
        from ui.views.settings.settings_card_style import sync_settings_collapsible_cards

        sync_settings_collapsible_cards(self, is_dark=is_dark)
        return True

    def _mount_settings_section_content(
        self, section_id: str, content_widget: QWidget
    ) -> None:
        content_host = self._section_content_hosts.get(section_id)
        if content_host is None:
            return

        content_widget.setMinimumWidth(0)
        content_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        layout = content_host.layout()
        if layout is None:
            layout = QVBoxLayout(content_host)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        layout.addWidget(content_widget)

    def _schedule_settings_section_prefetch(self) -> None:
        self._settings_prefetch_queue = [
            sec_def.id
            for sec_def in SETTINGS_SECTIONS
            if sec_def.id not in self._sections_built
        ]
        if self._settings_prefetch_queue:
            QTimer.singleShot(0, self._prefetch_next_settings_section)

    def _prefetch_next_settings_section(self) -> None:
        if not self._settings_prefetch_queue:
            return
        section_id = self._settings_prefetch_queue.pop(0)
        self._ensure_section_built(section_id)
        if self._settings_prefetch_queue:
            QTimer.singleShot(0, self._prefetch_next_settings_section)
    def _add_settings_group_header(self, group_text: str) -> None:
        item = QListWidgetItem()
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        item.setSizeHint(QSize(0, 28))
        header = QLabel(group_text)
        header.setObjectName("SettingsSectionGroupHeader")
        self.settings_section_list.addItem(item)
        self.settings_section_list.setItemWidget(item, header)

    def _upsert_section_search_index(self, sec_def, keywords: set[str]) -> None:
        self._settings_search_index = [
            entry
            for entry in self._settings_search_index
            if entry["section_id"] != sec_def.id
        ]
        self._settings_search_index.append(
            {
                "section_id": sec_def.id,
                "keywords": keywords,
            }
        )

    def _index_section_search_base(self, sec_def) -> None:
        keywords: set[str] = {sec_def.title.lower(), sec_def.id.lower()}
        for legacy in sec_def.legacy_titles:
            keywords.add(legacy.lower())
        self._upsert_section_search_index(sec_def, keywords)

    def _index_section_for_search(self, sec_def, content_widget: QWidget) -> None:
        keywords: set[str] = {sec_def.title.lower(), sec_def.id.lower()}
        for legacy in sec_def.legacy_titles:
            keywords.add(legacy.lower())
        for lbl in content_widget.findChildren(QLabel):
            text = lbl.text().strip()
            if text:
                keywords.add(text.lower())
            anchor = lbl.property("settings_anchor")
            if anchor:
                keywords.add(str(anchor).lower())
        for cb in content_widget.findChildren(QCheckBox):
            text = cb.text().strip()
            if text:
                keywords.add(text.lower())
        self._upsert_section_search_index(sec_def, keywords)

    def _wire_companion_cognition_hint(self) -> None:
        lbl = getattr(self, "companion_cognition_hint_lbl", None)
        if lbl is None:
            return
        lbl.setTextFormat(Qt.TextFormat.RichText)
        lbl.setText(
            'Uses auxiliary cognition model — configure under '
            '<a href="cognition">AI &amp; Models → Auxiliary cognition</a>.'
        )
        lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        lbl.setOpenExternalLinks(False)
        lbl.linkActivated.connect(
            lambda _href: self.select_settings_section("ai.models", anchor="cognition")
        )
    def _sync_internal_engine_subsections(self, mode: str) -> None:
        internal = str(mode).lower().strip() == "internal"
        local_startup_card = getattr(self, "_ai_local_startup_card", None)
        if local_startup_card is not None:
            local_startup_card.setVisible(internal)
        for attr in (
            "_ai_local_models_subsection",
            "_ai_startup_subsection",
        ):
            wrapper = getattr(self, attr, None)
            if wrapper is not None:
                wrapper.setVisible(internal)
        for lbl in getattr(self, "_ai_internal_subsection_labels", []):
            if lbl is not None:
                lbl.setVisible(internal)
        internal_tuning_card = getattr(self, "_ai_internal_tuning_card", None)
        if internal_tuning_card is not None:
            internal_tuning_card.setVisible(internal)
        hint = getattr(self, "_ai_external_engine_hint", None)
        if hint is not None:
            hint.setVisible(not internal)
        if hasattr(self, "_sync_hardware_chat_template_panels"):
            self._sync_hardware_chat_template_panels()
        from ui.views.settings.knowledge_access_badge import coalesce_settings_is_dark
        from ui.views.settings.settings_card_style import sync_settings_collapsible_cards

        sync_settings_collapsible_cards(self, is_dark=coalesce_settings_is_dark(self))
    def _on_settings_search_changed(self, text: str) -> None:
        query = text.strip().lower()
        first_match_row: int | None = None
        for row in range(self.settings_section_list.count()):
            item = self.settings_section_list.item(row)
            if item is None:
                continue
            section_id = item.data(self._SETTINGS_SECTION_ID_ROLE)
            if section_id is None:
                continue
            if not query:
                item.setHidden(False)
                if first_match_row is None:
                    first_match_row = row
                continue
            entry = next(
                (e for e in self._settings_search_index if e["section_id"] == section_id),
                None,
            )
            matches = entry is not None and any(query in kw for kw in entry["keywords"])
            item.setHidden(not matches)
            if matches and first_match_row is None:
                first_match_row = row

        for row in range(self.settings_section_list.count()):
            item = self.settings_section_list.item(row)
            if item is None or item.data(self._SETTINGS_SECTION_ID_ROLE) is not None:
                continue
            if not query:
                item.setHidden(False)
                continue
            show_header = False
            for r in range(row + 1, self.settings_section_list.count()):
                next_item = self.settings_section_list.item(r)
                if next_item is None:
                    break
                if next_item.data(self._SETTINGS_SECTION_ID_ROLE) is None:
                    break
                if not next_item.isHidden():
                    show_header = True
                    break
            item.setHidden(not show_header)

        if query and first_match_row is not None:
            self.settings_section_list.setCurrentRow(first_match_row)
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
        theme = resolve_settings_theme(self, is_dark=is_dark)
        color = settings_chevron_color(theme, enabled=button.isEnabled())
        button.setIcon(qta.icon("fa5s.chevron-down", color=color))
    def _make_settings_info_button(self, tooltip_text: str) -> QToolButton:
        theme = resolve_settings_theme(self)
        btn = QToolButton()
        btn.setObjectName("SettingsInfoButton")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setFixedSize(22, 22)
        btn.setToolTip(tooltip_text)
        btn.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        btn.setIcon(qta.icon("fa5s.info-circle", color=settings_info_icon_color(theme)))
        btn.setIconSize(QSize(14, 14))
        btn.setAutoRaise(True)
        btn.setStyleSheet(theme.style(SETTINGS_GHOST_TOOL_BUTTON))
        btn.clicked.connect(
            lambda _checked=False, anchor=btn, text=tooltip_text: self._show_settings_info_tooltip(
                anchor, text
            )
        )
        return btn

    def _show_settings_info_tooltip(self, anchor: QToolButton, text: str) -> None:
        """Show help copy on info-button click (hover tips can miss tiny targets)."""
        from core.qube_tooltip import QubeToolTipController

        gpos = anchor.mapToGlobal(anchor.rect().center())
        QubeToolTipController.instance().show_tip(anchor, gpos, text)
    def sync_active_native_model_label(self) -> None:
        """Public hook for MainWindow when the toolbar/native load state changes."""
        self._sync_active_native_model_label()
    def refresh_native_local_library(self) -> None:
        """Call when a .gguf is saved elsewhere (e.g. Model Manager download)."""
        self._sync_models_dir_label()
        self._sync_active_native_model_label()
        self._ensure_section_data_loaded("ai.models", force=True)
        if hasattr(self, "_refresh_cognition_gguf_list"):
            self._refresh_cognition_gguf_list()
    def refresh_menu_themes(self, is_dark: bool):
        """Standardizes icons and borders when the theme is toggled."""
        if getattr(self, "_skip_next_menu_theme_refresh", False):
            self._skip_next_menu_theme_refresh = False
            return
        from ui.views.settings.knowledge_access_badge import coalesce_settings_is_dark

        is_dark = coalesce_settings_is_dark(self, is_dark=is_dark)
        theme = resolve_settings_theme(self, is_dark=is_dark)
        if not getattr(self, "_theme_buttons", None):
            collect_theme_buttons(self)
        for toggle in self.findChildren(PrestigeToggle):
            toggle.apply_theme(is_dark=is_dark)
        for btn in self._theme_buttons:
            if btn is None:
                continue
            if isinstance(btn, SelectorButton):
                btn.apply_theme(is_dark)
            if btn.menu():
                self._apply_menu_theme(btn.menu(), is_dark)

        info_btn = getattr(self, "advanced_engine_info_btn", None)
        icon_color = settings_info_icon_color(theme)
        for btn in self.findChildren(QToolButton):
            if btn.objectName() == "SettingsInfoButton":
                btn.setIcon(qta.icon("fa5s.info-circle", color=icon_color))
        if info_btn is not None and info_btn.objectName() != "SettingsInfoButton":
            info_btn.setIcon(qta.icon("fa5s.info-circle", color=icon_color))

        hint_color = settings_hint_icon_color(theme)
        for hint_btn in (getattr(self, "audio_input_hint_btn", None),):
            if hint_btn is not None:
                hint_btn.setIcon(qta.icon("fa5s.lightbulb", color=hint_color))

        preview_color = settings_preview_icon_color(theme)
        for preview_btn in (
            getattr(self, "tts_voice_preview_btn", None),
            getattr(self, "audio_output_preview_btn", None),
        ):
            if preview_btn is not None:
                preview_btn.setIcon(qta.icon("fa5s.play", color=preview_color))

        wakeword_info_btn = getattr(self, "wakeword_info_btn", None)
        if wakeword_info_btn is not None:
            wakeword_info_btn.setIcon(
                qta.icon("fa5s.info-circle", color=settings_info_icon_color(theme))
            )

        for label in getattr(self, "_bootstrap_warning_labels", ()) or ():
            if label is not None:
                style_bootstrap_warning_label(label, theme)

        from ui.views.settings.widgets import SettingsSectionDivider

        for divider in self.findChildren(SettingsSectionDivider):
            divider.apply_theme(is_dark)

        hint_style = theme.style(SETTINGS_HINT)
        for lbl in self.findChildren(QLabel):
            if lbl.objectName() == "SettingsHint" or lbl.property("class") == "SettingsHint":
                lbl.setStyleSheet(hint_style)

        if hasattr(self, "custom_sources_table"):
            from ui.views.settings.sections.knowledge_custom_sources import (
                _refresh_custom_sources_list,
            )

            _refresh_custom_sources_list(self, is_dark=is_dark)
        if hasattr(self, "knowledge_presets_table"):
            from ui.views.settings.sections.knowledge_presets import _refresh_presets_list

            _refresh_presets_list(self, is_dark=is_dark)

        refresh_settings_section_cards(self, is_dark=is_dark)

        icon_color = settings_nav_icon_color(theme)

        for icon_lbl in getattr(self, "_settings_section_icon_labels", []):
            self._refresh_settings_icon_label(icon_lbl, icon_color)

        for icon_lbl in getattr(self, "_settings_nav_icon_labels", []):
            self._refresh_settings_icon_label(icon_lbl, icon_color)

        self._update_settings_section_nav_colors()

        if hasattr(self, "trigger_add_btn"):
            self.trigger_add_btn.setIcon(qta.icon("fa5s.plus", color=theme.accent))
            self.trigger_add_btn.setStyleSheet(theme.style(SETTINGS_ICON_BUTTON))

        self._apply_spinbox_style(is_dark)
        self._apply_settings_sidebar_surface(is_dark)
        self._refresh_trigger_list() # Repaints the list fonts & trash icons!
        self._sync_ai_provider_enabled_for_inference(get_engine_mode())

        if self._wakeword_testbed_dialog is not None:
            self._wakeword_testbed_dialog.refresh_theme(is_dark)

        if self._settings_json_dialog is not None:
            self._settings_json_dialog.refresh_theme(is_dark)

        if hasattr(self, "companion_preview"):
            self.companion_preview.apply_theme(is_dark)

        if hasattr(self, "companion_snap_compass"):
            self.companion_snap_compass.apply_theme(is_dark)

        if hasattr(self, "knowledge_provider_status_table"):
            from ui.views.settings.sections.knowledge_provider_status import (
                sync_provider_status_panel,
            )

            sync_provider_status_panel(self, is_dark=is_dark)

        self._refresh_knowledge_access_ui(is_dark=is_dark)

        if hasattr(self, "_apply_themes_action_button_styles"):
            self._apply_themes_action_button_styles(is_dark)
    def _init_settings_layout(self) -> None:
        main_layout = QVBoxLayout(self)
        # Keep right breathing room, but let the sidebar reach top and bottom like Model Manager.
        main_layout.setContentsMargins(0, 0, 40, 0)
        main_layout.setSpacing(16)

        hub_container = QWidget()
        hub_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._settings_hub_container = hub_container
        hub_h = QHBoxLayout(hub_container)
        hub_h.setContentsMargins(0, 0, 0, 0)
        hub_h.setSpacing(0)

        left = QFrame()
        left.setFixedWidth(LEFT_NAV_LIST_SIDEBAR_WIDTH)
        left.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        left.setObjectName("SettingsSidebar")
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(15, 20, 15, 20)
        left_l.setSpacing(15)
        self.settings_sidebar = left

        title = QLabel("System Settings")
        title.setObjectName("ViewTitle")
        title.setProperty("class", "PageTitle")
        left_l.addWidget(title)

        self.settings_search_input = QLineEdit()
        self.settings_search_input.setObjectName("SettingsSectionSearchBar")
        self.settings_search_input.setPlaceholderText("Search settings…")
        self.settings_search_input.setClearButtonEnabled(True)
        self.settings_search_input.setToolTip(
            "Filter settings sections by name or keyword."
        )
        self.settings_search_input.textChanged.connect(self._on_settings_search_changed)
        left_l.addWidget(self.settings_search_input)

        self.settings_section_list = QListWidget()
        self.settings_section_list.setObjectName("SettingsSectionList")
        self.settings_section_list.setMinimumWidth(0)
        self.settings_section_list.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.settings_section_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.settings_section_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.settings_section_list.setToolTip(
            "Choose a settings category to view and edit on the right."
        )
        left_l.addWidget(self.settings_section_list, stretch=1)

        right = QWidget()
        right.setMinimumWidth(0)
        right.setMaximumWidth(900)
        right.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(8, 75, 8, 40)
        right_l.setSpacing(10)

        self._settings_nav_icon_labels: list[QLabel] = []
        self._settings_section_icon_labels: list[QLabel] = []
        self.settings_section_header_bar = SettingsSectionHeaderBar(right)
        (
            self.settings_section_title_lbl,
            self.settings_section_tour_btn,
            self.settings_section_icon_lbl,
            section_header,
            self.settings_section_collapse_all_btn,
        ) = make_settings_section_header_row(
            self.settings_section_header_bar,
            initial_tour_id="settings.voice_audio",
            initial_area_display_name="Voice & Audio settings",
        )
        from ui.views.settings.license_status_ui import attach_settings_edition_chip

        self.settings_edition_chip = attach_settings_edition_chip(section_header, host=self)
        self.settings_section_collapse_all_btn.clicked.connect(
            self._on_settings_collapse_all_clicked
        )
        self._settings_section_icon_labels.append(self.settings_section_icon_lbl)
        self.settings_section_header_bar.set_header_content(section_header)
        right_l.addWidget(self.settings_section_header_bar)

        self.settings_section_stack = QStackedWidget()
        self.settings_section_stack.setObjectName("SettingsSectionStack")
        right_l.addWidget(self.settings_section_stack, stretch=1)

        right_host = QWidget()
        right_host_l = QHBoxLayout(right_host)
        right_host_l.setContentsMargins(10, 0, 0, 0)
        right_host_l.setSpacing(0)
        right_host_l.addWidget(right, 1)

        hub_h.addWidget(left)
        hub_h.addWidget(right_host, stretch=1)

        main_layout.addWidget(hub_container, stretch=1)

        self.settings_section_list.currentRowChanged.connect(self._on_settings_section_changed)
        self.settings_section_list.itemSelectionChanged.connect(
            self._update_settings_section_nav_colors
        )
    def _finalize_settings_layout(self, is_dark: bool) -> None:
        from ui.views.settings.knowledge_access_badge import coalesce_settings_is_dark

        is_dark = coalesce_settings_is_dark(self, is_dark=is_dark)
        self._apply_spinbox_style(is_dark)
        self._apply_settings_sidebar_surface(is_dark)
        self._update_settings_section_nav_colors()
        self._refresh_knowledge_access_ui(is_dark=is_dark)
        if hasattr(self, "_refresh_license_status_ui"):
            self._refresh_license_status_ui()

    def _settings_section_icon_pixmap(
        self,
        *,
        icon_name: str,
        svg_icon: tuple[str, ...] | None,
        size: int,
        color: str,
    ) -> QPixmap:
        if svg_icon is not None:
            return tinted_svg_pixmap(
                str(resource_path(*svg_icon)), color, size
            )
        return themed_fa_pixmap(icon_name, color, size)

    def _refresh_settings_icon_label(self, icon_lbl: QLabel, color: str) -> None:
        svg_path = icon_lbl.property("svg_path")
        icon_name = icon_lbl.property("icon_name")
        size = int(icon_lbl.property("icon_size") or 16)
        if svg_path:
            icon_lbl.setPixmap(tinted_svg_pixmap(str(svg_path), color, size))
        elif icon_name:
            icon_lbl.setPixmap(
                themed_fa_pixmap(str(icon_name), color, size)
            )

    def _build_settings_section_nav_row(
        self,
        icon_name: str,
        title_text: str,
        *,
        svg_icon: tuple[str, ...] | None = None,
    ) -> QWidget:
        row = QWidget()
        row.setObjectName("HistoryRowWidget")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(12, 8, 10, 8)
        layout.setSpacing(10)

        icon_label = QLabel()
        icon_label.setProperty("icon_name", icon_name)
        icon_label.setProperty(
            "svg_path",
            str(resource_path(*svg_icon)) if svg_icon is not None else "",
        )
        icon_label.setProperty("icon_size", 16)
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        theme = resolve_settings_theme(self, is_dark=is_dark)
        icon_color = settings_nav_icon_color(theme)
        icon_label.setPixmap(
            self._settings_section_icon_pixmap(
                icon_name=icon_name,
                svg_icon=svg_icon,
                size=16,
                color=icon_color,
            )
        )
        icon_label.setFixedSize(18, 18)
        self._settings_nav_icon_labels.append(icon_label)

        title_lbl = QLabel(title_text)
        title_lbl.setObjectName("HistoryRowTitle")
        title_lbl.setWordWrap(False)

        layout.addWidget(icon_label, stretch=0, alignment=Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(title_lbl, stretch=1, alignment=Qt.AlignmentFlag.AlignVCenter)
        return row
    def _on_settings_section_changed(self, row: int) -> None:
        if row < 0:
            return
        item = self.settings_section_list.item(row)
        if item is None:
            return
        stack_idx = item.data(self._SETTINGS_STACK_ROLE)
        if stack_idx is None:
            return
        previous_section = getattr(self, "_settings_active_section_id", None)
        section_id = item.data(self._SETTINGS_SECTION_ID_ROLE)
        if section_id:
            self._ensure_section_built(section_id)
        if (
            previous_section == "appearance.themes"
            and section_id != "appearance.themes"
            and hasattr(self, "_on_themes_section_leave")
        ):
            self._on_themes_section_leave()
        self.settings_section_stack.setCurrentIndex(int(stack_idx))
        self._settings_active_section_id = section_id
        self._ensure_section_data_loaded(section_id)
        self._sync_settings_section_tour_header(section_id)
        self._sync_settings_collapse_all_button(section_id)
        if section_id == "appearance.themes" and hasattr(self, "_on_themes_section_enter"):
            self._on_themes_section_enter()
        if section_id in ("privacy.data", "diagnostics") and hasattr(
            self, "_sync_all_diagnostic_log_recording_toggles"
        ):
            self._sync_all_diagnostic_log_recording_toggles()
        if section_id == "privacy.data" and hasattr(self, "_sync_privacy_data_section_ui"):
            self._sync_privacy_data_section_ui()
        if section_id == "license" and hasattr(self, "_refresh_license_status_ui"):
            self._refresh_license_status_ui()
        if section_id == "knowledge":
            from ui.views.settings.sections.knowledge_provider_status import (
                start_provider_status_refresh_timer,
                stop_provider_status_refresh_timer,
            )

            start_provider_status_refresh_timer(self)
            QTimer.singleShot(0, self._refresh_knowledge_access_ui)
        else:
            from ui.views.settings.sections.knowledge_provider_status import (
                stop_provider_status_refresh_timer,
            )

            stop_provider_status_refresh_timer(self)
        QTimer.singleShot(0, self._relayout_trigger_list_rows)

    def _sync_settings_collapse_all_button(self, section_id: str | None) -> None:
        btn = getattr(self, "settings_section_collapse_all_btn", None)
        if btn is None:
            return
        from core.app_settings import get_settings_section_cards_collapsible

        if not get_settings_section_cards_collapsible() or not section_id:
            btn.hide()
            return
        cards = getattr(self, "_settings_collapsible_cards_by_section", {}).get(
            section_id, ()
        )
        from ui.views.settings.settings_card_style import collapsible_card_has_title

        titled_cards = [
            handle
            for handle in cards
            if collapsible_card_has_title(handle)
        ]
        if len(titled_cards) < 2:
            btn.hide()
            return
        btn.show()
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        color = "#89b4fa" if is_dark else "#64748b"
        all_expanded = all(handle.expanded for handle in titled_cards)
        icon_name = "fa5s.chevron-up" if all_expanded else "fa5s.chevron-down"
        btn.setIcon(qta.icon(icon_name, color=color))
        btn.setToolTip(
            "Collapse all sections on this page"
            if all_expanded
            else "Expand all sections on this page"
        )
        btn.setProperty("_collapse_all_expanded", all_expanded)

    def _on_settings_collapse_all_clicked(self) -> None:
        row = self.settings_section_list.currentRow()
        if row < 0:
            return
        item = self.settings_section_list.item(row)
        if item is None:
            return
        section_id = item.data(self._SETTINGS_SECTION_ID_ROLE)
        if not section_id:
            return
        btn = self.settings_section_collapse_all_btn
        expand_all = not bool(btn.property("_collapse_all_expanded"))
        from ui.views.settings.settings_card_style import (
            set_settings_collapsible_cards_expanded,
        )

        set_settings_collapsible_cards_expanded(
            self, str(section_id), expanded=expand_all
        )
        self._sync_settings_collapse_all_button(str(section_id))

    def _sync_settings_section_tour_header(self, section_id: str | None) -> None:
        if not hasattr(self, "settings_section_tour_btn"):
            return
        sync_settings_section_tour_header(
            self.settings_section_title_lbl,
            self.settings_section_tour_btn,
            section_id,
            icon_lbl=getattr(self, "settings_section_icon_lbl", None),
        )
        icon_lbl = getattr(self, "settings_section_icon_lbl", None)
        if icon_lbl is not None and section_id:
            theme = resolve_settings_theme(self)
            self._refresh_settings_icon_label(
                icon_lbl, settings_nav_icon_color(theme)
            )

    def _update_settings_section_nav_colors(self) -> None:
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        apply_sidebar_row_title_colors(self.settings_section_list, is_dark=is_dark)

    def _build_divider(self):
        line = QFrame()
        line.setObjectName("SettingsDivider")
        line.setFrameShape(QFrame.Shape.HLine)
        return line
    def update_voice_dropdown(self, model_name: str, voices: list) -> None:
        window = self.window()
        if window is not None and hasattr(window, "update_tts_voice_dropdowns"):
            window.update_tts_voice_dropdowns(model_name, voices)
            return
        if not voices:
            return
        self._build_prestige_menu(
            self.voice_selector,
            [(v, v) for v in voices],
            lambda v: self.tts_worker.set_voice(v) if self.tts_worker else None,
        )
        from core.tts_models import resolve_default_tts_voice

        active = (
            self.tts_worker.active_voice_name
            if self.tts_worker and hasattr(self.tts_worker, "active_voice_name")
            else resolve_default_tts_voice(voices)
        )
        if active not in voices:
            active = resolve_default_tts_voice(voices)
        self.voice_selector.setText(active)
        if self.tts_worker:
            self.tts_worker.set_voice(active)
