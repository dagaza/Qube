"""Settings → Backup & restore section."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core import app_settings as _backup_settings
from core.paths import user_data_root
from core.state_backup.paths import auto_backups_dir, default_backups_dir
from ui.components.brand_buttons import apply_brand_primary
from ui.components.selector_button import SelectorButton
from ui.components.toggle import PrestigeToggle
from ui.views.settings.controls import NoScrollSpinBox
from ui.views.settings.settings_card_style import begin_settings_section_card
from ui.views.settings.widgets import (
    add_settings_card_form,
    add_settings_full_width_row,
    add_subsection_to_form,
    make_settings_action_row,
    make_settings_hint,
    register_settings_selector_width,
)


def _interval_label(days: int) -> str:
    return f"Every {days} days"


def build_section(host, *, is_dark: bool) -> QWidget:
    widget = QWidget()
    widget.setObjectName("SettingsFormContainer")
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(15, 0, 15, 10)
    layout.setSpacing(15)

    data_root = user_data_root()
    backups_dir = default_backups_dir(data_root)
    auto_dir = auto_backups_dir(data_root)

    overview_card, overview_layout = begin_settings_section_card(host, is_dark=is_dark)
    overview_form = add_settings_card_form(overview_layout)
    add_subsection_to_form(overview_form, "Local backup", anchor="overview")

    host.state_backup_overview_hint = make_settings_hint(
        "Save or restore essential Qube state — conversations, library indexes, memory "
        "vectors, settings, knowledge configuration, and themes. Model weights under "
        f"{data_root / 'models'} are not included. Restore replaces matching files and "
        "requires restarting Qube."
    )
    add_settings_full_width_row(overview_form, host.state_backup_overview_hint)

    host.state_backup_storage_hint = make_settings_hint("")
    add_settings_full_width_row(overview_form, host.state_backup_storage_hint)
    if hasattr(host, "_refresh_state_backup_storage_hint"):
        host._refresh_state_backup_storage_hint()

    host.state_backup_open_guide_btn = QPushButton("Read backup guide")
    apply_brand_primary(host.state_backup_open_guide_btn, icon_name="fa5s.book-open")
    host.state_backup_open_guide_btn.setToolTip(
        "Open the Back up or restore Qube state workflow in Library → Qube."
    )
    host.state_backup_open_guide_btn.clicked.connect(
        host._on_open_backup_restore_guide_clicked
    )
    add_settings_full_width_row(
        overview_form,
        make_settings_action_row(host.state_backup_open_guide_btn),
    )
    layout.addWidget(overview_card)

    auto_card, auto_layout = begin_settings_section_card(host, is_dark=is_dark)
    auto_form = add_settings_card_form(auto_layout)
    add_subsection_to_form(auto_form, "Automatic backup", anchor="automatic")

    host.state_backup_auto_enabled_toggle = PrestigeToggle()
    host.state_backup_auto_enabled_toggle.setChecked(
        _backup_settings.get_backup_auto_enabled()
    )
    host.state_backup_auto_enabled_toggle.toggled.connect(
        host._on_state_backup_auto_enabled_toggled
    )
    auto_enabled_row = QWidget()
    auto_enabled_layout = QHBoxLayout(auto_enabled_row)
    auto_enabled_layout.setContentsMargins(0, 0, 0, 0)
    auto_enabled_layout.setSpacing(12)
    auto_enabled_label = make_settings_hint(
        "Run a local backup on startup when the interval has elapsed."
    )
    auto_enabled_layout.addWidget(auto_enabled_label, 1)
    auto_enabled_layout.addWidget(host.state_backup_auto_enabled_toggle, 0)
    add_settings_full_width_row(auto_form, auto_enabled_row)

    host.state_backup_interval_selector = SelectorButton(
        _interval_label(_backup_settings.get_backup_interval_days()),
        is_dark=is_dark,
    )
    register_settings_selector_width(host.state_backup_interval_selector)
    host.state_backup_interval_selector.setToolTip(
        "Minimum time between automatic backups."
    )
    host.state_backup_interval_selector.clicked.connect(
        host._on_state_backup_interval_menu_requested
    )
    interval_row = QWidget()
    interval_row_layout = QHBoxLayout(interval_row)
    interval_row_layout.setContentsMargins(0, 0, 0, 0)
    interval_row_layout.setSpacing(12)
    interval_row_layout.addWidget(make_settings_hint("Backup interval"), 0)
    interval_row_layout.addWidget(host.state_backup_interval_selector, 1)
    add_settings_full_width_row(auto_form, interval_row)

    retention_row = QWidget()
    retention_row_layout = QHBoxLayout(retention_row)
    retention_row_layout.setContentsMargins(0, 0, 0, 0)
    retention_row_layout.setSpacing(12)
    retention_row_layout.addWidget(make_settings_hint("Keep automatic backups"), 0)
    host.state_backup_retention_spin = NoScrollSpinBox()
    host.state_backup_retention_spin.setRange(1, 10)
    host.state_backup_retention_spin.setValue(_backup_settings.get_backup_retention_count())
    host.state_backup_retention_spin.setToolTip(
        f"Older archives under {auto_dir} are deleted when this limit is exceeded."
    )
    host.state_backup_retention_spin.valueChanged.connect(
        host._on_state_backup_retention_changed
    )
    retention_row_layout.addWidget(host.state_backup_retention_spin, 0)
    retention_row_layout.addStretch()
    add_settings_full_width_row(auto_form, retention_row)

    host.state_backup_include_wallpapers_cb = QCheckBox(
        "Include wallpapers in automatic backups"
    )
    host.state_backup_include_wallpapers_cb.setChecked(
        _backup_settings.get_backup_include_wallpapers()
    )
    host.state_backup_include_wallpapers_cb.toggled.connect(
        host._on_state_backup_include_wallpapers_toggled
    )
    add_settings_full_width_row(auto_form, host.state_backup_include_wallpapers_cb)

    host.state_backup_auto_hint = make_settings_hint(
        f"Automatic archives are saved under {auto_dir}. Manual backups and pre-restore "
        f"safety snapshots use {backups_dir}."
    )
    add_settings_full_width_row(auto_form, host.state_backup_auto_hint)

    host.state_backup_status_hint = make_settings_hint("")
    add_settings_full_width_row(auto_form, host.state_backup_status_hint)
    if hasattr(host, "_refresh_state_backup_status_hint"):
        host._refresh_state_backup_status_hint()

    layout.addWidget(auto_card)

    manual_card, manual_layout = begin_settings_section_card(host, is_dark=is_dark)
    manual_form = add_settings_card_form(manual_layout)
    add_subsection_to_form(manual_form, "Manual backup", anchor="manual")

    host.state_backup_create_btn = QPushButton("Create backup now")
    apply_brand_primary(host.state_backup_create_btn, icon_name="fa5s.archive")
    host.state_backup_create_btn.setToolTip(
        "Export essential state to a .qube-backup.zip file on disk."
    )
    host.state_backup_create_btn.clicked.connect(host._on_state_backup_create_clicked)

    host.state_backup_restore_btn = QPushButton("Restore from backup…")
    apply_brand_primary(host.state_backup_restore_btn, icon_name="fa5s.upload")
    host.state_backup_restore_btn.setToolTip(
        "Replace local state from a backup archive. A pre-restore snapshot is saved first."
    )
    host.state_backup_restore_btn.clicked.connect(host._on_state_backup_restore_clicked)

    manual_row = QWidget()
    manual_row_layout = QHBoxLayout(manual_row)
    manual_row_layout.setContentsMargins(0, 0, 0, 0)
    manual_row_layout.setSpacing(12)
    manual_row_layout.addWidget(host.state_backup_create_btn)
    manual_row_layout.addWidget(host.state_backup_restore_btn)
    manual_row_layout.addStretch()
    add_settings_full_width_row(manual_form, manual_row)

    host.state_backup_manual_hint = make_settings_hint(
        f"Pre-restore safety snapshots are written to {backups_dir} before any restore."
    )
    add_settings_full_width_row(manual_form, host.state_backup_manual_hint)

    host.state_backup_open_backups_btn = QPushButton("Open backups folder")
    apply_brand_primary(host.state_backup_open_backups_btn, icon_name="fa5s.folder-open")
    host.state_backup_open_backups_btn.clicked.connect(host._on_state_backup_open_backups_clicked)
    add_settings_full_width_row(
        manual_form,
        make_settings_action_row(host.state_backup_open_backups_btn),
    )
    layout.addWidget(manual_card)

    return widget
