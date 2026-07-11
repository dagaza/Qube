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
from ui.views.settings.registry import SETTINGS_SECTIONS, resolve_section_id
from ui.views.settings.widgets import collect_theme_buttons
from ui.views.settings.sections import (
    advanced,
    ai_models,
    contact_feedback,
    desktop_companion,
    general,
    help,
    knowledge,
    memory,
    notifications,
    voice_audio,
)

from ui.views.settings.handlers import (
    AiModelsHandlersMixin,
    BootstrapDownloadsHandlersMixin,
    CompanionHandlersMixin,
    DiagnosticsHandlersMixin,
    GenerationMixin,
    KnowledgeHandlersMixin,
    MemoryHandlersMixin,
    PersistenceHandlersMixin,
    PrestigeMenuMixin,
    StylingMixin,
    SupportHandlersMixin,
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
    "general": general.build_section,
    "companion.desktop": desktop_companion.build_section,
    "notifications": notifications.build_section,
    "help": help.build_section,
    "contact.feedback": contact_feedback.build_section,
    "advanced": advanced.build_section,
}


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
    CompanionHandlersMixin,
    DiagnosticsHandlersMixin,
    PersistenceHandlersMixin,
    SupportHandlersMixin,
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
    def __init__(self, workers: dict, db_manager):
        super().__init__()
        self.workers = workers
        self.db = db_manager
        
        self.audio_worker = workers.get("audio")
        self.tts_worker = workers.get("tts")
        self.llm_worker = workers.get("llm")
        self._template_override_reload_pending = False
        self._auto_reset_reload_pending = False
        self._companion_verbal_test_worker = None

        self._setup_ui()
        self.engine_mode_changed.connect(self._sync_ai_provider_enabled_for_inference)
        self.engine_mode_changed.connect(lambda _mode: self._sync_native_chat_template_label())
        native_engine = self.workers.get("native_engine")
        if native_engine is not None and hasattr(native_engine, "load_finished"):
            native_engine.load_finished.connect(self._on_native_model_load_finished)
        self._populate_hardware_selectors()
        os.makedirs(get_llm_models_dir(), exist_ok=True)
        os.makedirs(get_embedding_models_dir(), exist_ok=True)
        migrate_legacy_tts_layout()
        self._sync_models_dir_label()
        self._sync_embedding_models_dir_label()
        self._sync_stt_models_dir_label()
        self._sync_tts_models_dir_label()
        self._sync_active_native_model_label()
        self._sync_native_chat_template_label()
        self._refresh_local_gguf_list()
        self._refresh_embedding_gguf_list()
        self._sync_active_embedding_label()
        if hasattr(self, "_sync_embedding_mode_selector"):
            self._sync_embedding_mode_selector()
        self._refresh_stt_model_list()
        self._refresh_tts_model_list()
        self._sync_active_stt_label()
        self._sync_active_tts_label()
        self._wakeword_testbed_dialog = None
        self._settings_json_dialog: SettingsJsonEditorDialog | None = None
        self._setup_settings_file_watcher()
    def select_settings_section(
        self,
        section: str,
        *,
        anchor: str | None = None,
        configure_provider_id: str | None = None,
    ) -> None:
        """Show a settings section by stable id, title, or legacy title."""
        section_id = resolve_section_id(section)
        if section_id is None:
            return
        row = self._section_row_by_id.get(section_id)
        if row is not None:
            self.settings_section_list.setCurrentRow(row)
        if anchor:
            QTimer.singleShot(0, lambda: self._scroll_to_settings_anchor(anchor))
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
    def _scroll_to_settings_anchor(self, anchor: str) -> None:
        scroll = self.settings_section_stack.currentWidget()
        if scroll is None or not isinstance(scroll, QScrollArea):
            return
        page = scroll.widget()
        if page is None:
            return
        for lbl in page.findChildren(QLabel):
            if lbl.property("settings_anchor") == anchor:
                scroll.ensureWidgetVisible(lbl, 0, 80)
                return
        for wrapper in page.findChildren(QWidget):
            if wrapper.property("settings_anchor") == anchor:
                scroll.ensureWidgetVisible(wrapper, 0, 80)
                return

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

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self._sync_active_native_model_label()
        self._sync_native_chat_template_label()
        if hasattr(self, "_refresh_inference_transparency_panel"):
            self._refresh_inference_transparency_panel()
        self._ensure_settings_file_watched()
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        self._apply_settings_sidebar_surface(is_dark)
        QTimer.singleShot(0, self._relayout_trigger_list_rows)
        if hasattr(self, "_sync_bootstrap_download_visibility"):
            self._sync_bootstrap_download_visibility()
        self._maybe_start_provider_status_refresh()

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
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        self._section_index_by_id: dict[str, int] = {}
        self._section_row_by_id: dict[str, int] = {}
        self._section_stack_index_by_id: dict[str, int] = {}
        self._settings_search_index: list[dict] = []
        self._theme_buttons: list = []

        self._init_settings_layout()

        self._section_builders_for_rebuild = _SECTION_BUILDERS

        last_group: str | None = None
        for sec_def in SETTINGS_SECTIONS:
            if sec_def.group and sec_def.group != last_group:
                self._add_settings_group_header(sec_def.group)
                last_group = sec_def.group
            builder = _SECTION_BUILDERS[sec_def.id]
            content_widget = builder(self, is_dark=is_dark)
            self._add_settings_section(sec_def, content_widget)
            self._index_section_for_search(sec_def, content_widget)

        self._wire_companion_cognition_hint()
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        collect_theme_buttons(self)
        self._finalize_settings_layout(is_dark)
        self._sync_internal_engine_subsections(get_engine_mode())
        if hasattr(self, "_sync_bootstrap_download_visibility"):
            self._sync_bootstrap_download_visibility()
    _SETTINGS_STACK_ROLE = int(Qt.ItemDataRole.UserRole)
    _SETTINGS_SECTION_ID_ROLE = int(Qt.ItemDataRole.UserRole) + 1
    def _add_settings_group_header(self, group_text: str) -> None:
        item = QListWidgetItem()
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        item.setSizeHint(QSize(0, 28))
        header = QLabel(group_text)
        header.setObjectName("SettingsSectionGroupHeader")
        self.settings_section_list.addItem(item)
        self.settings_section_list.setItemWidget(item, header)
    def _add_settings_section(self, sec_def, content_widget: QWidget) -> None:
        header = self._build_section_header(
            sec_def.icon, sec_def.title, svg_icon=sec_def.svg_icon
        )

        content_widget.setMinimumWidth(0)
        content_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )

        page_content = QWidget()
        page_content.setObjectName("SettingsContent")
        page_content.setMinimumWidth(0)
        page_content.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        page_layout = QVBoxLayout(page_content)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(30)
        page_layout.addWidget(header)
        page_layout.addWidget(content_widget)
        page_layout.addStretch()

        scroll = QScrollArea()
        scroll.setObjectName("SettingsScrollArea")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(page_content)

        stack_idx = self.settings_section_stack.count()
        self.settings_section_stack.addWidget(scroll)
        self._section_stack_index_by_id[sec_def.id] = stack_idx

        item = QListWidgetItem()
        item.setSizeHint(QSize(0, 44))
        item.setData(self._SETTINGS_STACK_ROLE, stack_idx)
        item.setData(self._SETTINGS_SECTION_ID_ROLE, sec_def.id)
        row = self.settings_section_list.count()
        self.settings_section_list.addItem(item)
        self.settings_section_list.setItemWidget(
            item,
            self._build_settings_section_nav_row(
                sec_def.icon, sec_def.title, svg_icon=sec_def.svg_icon
            ),
        )
        self._section_row_by_id[sec_def.id] = row

        if len(self._section_row_by_id) == 1:
            self.settings_section_list.setCurrentRow(row)
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
        self._settings_search_index.append(
            {
                "section_id": sec_def.id,
                "keywords": keywords,
            }
        )
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
        for attr in (
            "_ai_local_models_subsection",
            "_ai_startup_subsection",
        ):
            wrapper = getattr(self, attr, None)
            if wrapper is not None:
                wrapper.setVisible(internal)
        for lbl in getattr(self, "_ai_internal_subsection_labels", []):
            lbl.setVisible(internal)
        hint = getattr(self, "_ai_external_engine_hint", None)
        if hint is not None:
            hint.setVisible(not internal)
        if hasattr(self, "_sync_hardware_chat_template_panels"):
            self._sync_hardware_chat_template_panels()
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
        muted = "#3f3f46" if is_dark else "#a1a1aa"
        active = "#64748b"
        color = active if button.isEnabled() else muted
        button.setIcon(qta.icon("fa5s.chevron-down", color=color))
    def _make_settings_info_button(self, tooltip_text: str) -> QToolButton:
        btn = QToolButton()
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setToolTip(tooltip_text)
        btn.setIcon(qta.icon("fa5s.info-circle", color="#64748b"))
        btn.setIconSize(QSize(14, 14))
        btn.setAutoRaise(True)
        btn.setStyleSheet(
            "QToolButton { border: none; padding: 0px; background: transparent; }"
        )
        return btn
    def sync_active_native_model_label(self) -> None:
        """Public hook for MainWindow when the toolbar/native load state changes."""
        self._sync_active_native_model_label()
    def refresh_native_local_library(self) -> None:
        """Call when a .gguf is saved elsewhere (e.g. Model Manager download)."""
        self._sync_models_dir_label()
        self._sync_active_native_model_label()
        self._refresh_local_gguf_list()
        if hasattr(self, "_refresh_cognition_gguf_list"):
            self._refresh_cognition_gguf_list()
    def refresh_menu_themes(self, is_dark: bool):
        """Standardizes icons and borders when the theme is toggled."""
        if not getattr(self, "_theme_buttons", None):
            collect_theme_buttons(self)
        for btn in self._theme_buttons:
            if btn is None:
                continue
            if isinstance(btn, SelectorButton):
                btn.apply_theme(is_dark)
            if btn.menu():
                self._apply_menu_theme(btn.menu(), is_dark)

        info_btn = getattr(self, "advanced_engine_info_btn", None)
        if info_btn is not None:
            info_color = "#94a3b8" if is_dark else "#64748b"
            info_btn.setIcon(qta.icon("fa5s.info-circle", color=info_color))

        hint_color = "#eab308" if is_dark else "#ca8a04"
        for hint_btn in (getattr(self, "audio_input_hint_btn", None),):
            if hint_btn is not None:
                hint_btn.setIcon(qta.icon("fa5s.lightbulb", color=hint_color))

        for preview_btn in (
            getattr(self, "tts_voice_preview_btn", None),
            getattr(self, "audio_output_preview_btn", None),
        ):
            if preview_btn is not None:
                preview_color = "#94a3b8" if is_dark else "#64748b"
                preview_btn.setIcon(qta.icon("fa5s.play", color=preview_color))

        divider = getattr(self, "voice_audio_section_divider", None)
        if divider is not None and hasattr(divider, "apply_theme"):
            divider.apply_theme(is_dark)

        # Update section header + sidebar nav icons
        icon_color = "#8b5cf6" if is_dark else "#4c4f69"

        for icon_lbl in getattr(self, "_settings_section_icon_labels", []):
            self._refresh_settings_icon_label(icon_lbl, icon_color)

        for icon_lbl in getattr(self, "_settings_nav_icon_labels", []):
            self._refresh_settings_icon_label(icon_lbl, icon_color)

        self._update_settings_section_nav_colors()

        # Update Trigger Add Button
        if hasattr(self, 'trigger_add_btn'):
            btn_bg = "#313244" if is_dark else "#e2e8f0"
            btn_hover = "#45475a" if is_dark else "#cbd5e1"
            self.trigger_add_btn.setIcon(qta.icon('fa5s.plus', color=icon_color))
            self.trigger_add_btn.setStyleSheet(f"""
                QPushButton {{ background: {btn_bg}; border: none; border-radius: 8px; }}
                QPushButton:hover {{ background: {btn_hover}; }}
            """)

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

        if hasattr(self, "knowledge_provider_status_table"):
            from ui.views.settings.sections.knowledge_provider_status import (
                sync_provider_status_panel,
            )

            sync_provider_status_panel(self, is_dark=is_dark)

        if hasattr(self, "knowledge_live_source_rows"):
            from ui.views.settings.sections.knowledge_sources import (
                refresh_live_source_access_badges,
            )

            refresh_live_source_access_badges(self)
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
        right_l.setContentsMargins(8, 75, 0, 40)
        right_l.setSpacing(10)

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

        self._settings_nav_icon_labels: list[QLabel] = []
        self._settings_section_icon_labels: list[QLabel] = []
        self.settings_section_list.currentRowChanged.connect(self._on_settings_section_changed)
        self.settings_section_list.itemSelectionChanged.connect(
            self._update_settings_section_nav_colors
        )
    def _finalize_settings_layout(self, is_dark: bool) -> None:
        self._apply_spinbox_style(is_dark)
        self._apply_settings_sidebar_surface(is_dark)
        self._update_settings_section_nav_colors()
    def _make_tinted_svg_pixmap(self, svg_path, color_hex: str, size: int) -> QPixmap:
        pixmap = QPixmap(str(svg_path))
        if pixmap.isNull():
            return QPixmap(size, size)
        target_size = QSize(size, size)
        pixmap = pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        tinted = QPixmap(pixmap.size())
        tinted.fill(Qt.GlobalColor.transparent)
        painter = QPainter(tinted)
        painter.drawPixmap(0, 0, pixmap)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(tinted.rect(), QColor(color_hex))
        painter.end()
        return tinted

    def _settings_section_icon_pixmap(
        self,
        *,
        icon_name: str,
        svg_icon: tuple[str, ...] | None,
        size: int,
        color: str,
    ) -> QPixmap:
        if svg_icon is not None:
            return self._make_tinted_svg_pixmap(
                resource_path(*svg_icon), color, size
            )
        return qta.icon(icon_name, color=color).pixmap(QSize(size, size))

    def _refresh_settings_icon_label(self, icon_lbl: QLabel, color: str) -> None:
        svg_path = icon_lbl.property("svg_path")
        icon_name = icon_lbl.property("icon_name")
        size = int(icon_lbl.property("icon_size") or 16)
        if svg_path:
            icon_lbl.setPixmap(self._make_tinted_svg_pixmap(svg_path, color, size))
        elif icon_name:
            icon_lbl.setPixmap(
                qta.icon(icon_name, color=color).pixmap(QSize(size, size))
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
        icon_color = "#8b5cf6" if is_dark else "#4c4f69"
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
        self.settings_section_stack.setCurrentIndex(int(stack_idx))
        section_id = item.data(self._SETTINGS_SECTION_ID_ROLE)
        if section_id == "advanced" and hasattr(
            self, "_sync_all_diagnostic_log_recording_toggles"
        ):
            self._sync_all_diagnostic_log_recording_toggles()
        if section_id == "knowledge":
            from ui.views.settings.sections.knowledge_provider_status import (
                start_provider_status_refresh_timer,
                stop_provider_status_refresh_timer,
            )

            start_provider_status_refresh_timer(self)
        else:
            from ui.views.settings.sections.knowledge_provider_status import (
                stop_provider_status_refresh_timer,
            )

            stop_provider_status_refresh_timer(self)
        QTimer.singleShot(0, self._relayout_trigger_list_rows)
    def _update_settings_section_nav_colors(self) -> None:
        is_dark = getattr(self.window(), "_is_dark_theme", True)
        apply_sidebar_row_title_colors(self.settings_section_list, is_dark=is_dark)
    def _build_section_header(
        self,
        icon_name,
        title_text,
        *,
        svg_icon: tuple[str, ...] | None = None,
    ):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        icon_label = QLabel()
        icon_label.setProperty("icon_name", icon_name)
        icon_label.setProperty(
            "svg_path",
            str(resource_path(*svg_icon)) if svg_icon is not None else "",
        )
        icon_label.setProperty("icon_size", 18)
        
        is_dark = getattr(self.window(), '_is_dark_theme', True)
        icon_color = "#8b5cf6" if is_dark else "#4c4f69"
        icon_label.setPixmap(
            self._settings_section_icon_pixmap(
                icon_name=icon_name,
                svg_icon=svg_icon,
                size=18,
                color=icon_color,
            )
        )
        icon_label.setProperty("class", "SectionHeaderIcon")
        self._settings_section_icon_labels.append(icon_label)
        
        text_label = QLabel(title_text)
        text_label.setProperty("class", "SectionHeaderLabel")
        
        layout.addWidget(icon_label)
        layout.addWidget(text_label)
        layout.addStretch()
        return container
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
        active = (
            self.tts_worker.active_voice_name
            if self.tts_worker and hasattr(self.tts_worker, "active_voice_name")
            else voices[0]
        )
        if active not in voices:
            active = voices[0]
        self.voice_selector.setText(active)
        if self.tts_worker:
            self.tts_worker.set_voice(active)
