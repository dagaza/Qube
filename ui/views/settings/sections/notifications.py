"""Notifications settings section."""

from __future__ import annotations

from PyQt6.QtWidgets import QCheckBox, QVBoxLayout, QWidget

from core import app_settings as _notif_settings
from ui.views.settings.settings_card_style import begin_settings_section_card
from ui.views.settings.widgets import (
    add_settings_card_form,
    add_subsection_to_form,
    add_section_reset_footer,
    make_settings_page_action_button,
    add_settings_full_width_row,
)


def build_section(host, *, is_dark: bool) -> QWidget:
    notif_widget = QWidget()
    notif_widget.setObjectName("SettingsFormContainer")
    notif_layout = QVBoxLayout(notif_widget)
    notif_layout.setContentsMargins(15, 0, 15, 10)
    notif_layout.setSpacing(15)

    # --- Master card ---
    master_card, master_card_layout = begin_settings_section_card(host, is_dark=is_dark)
    master_form = add_settings_card_form(master_card_layout)
    add_subsection_to_form(master_form, "Master")

    host.notifications_enabled_cb = QCheckBox("Enable notifications")
    host.notifications_enabled_cb.setChecked(_notif_settings.get_notifications_enabled())
    host.notifications_enabled_cb.toggled.connect(_notif_settings.set_notifications_enabled)
    add_settings_full_width_row(master_form, host.notifications_enabled_cb)

    host.notifications_dnd_cb = QCheckBox("Do Not Disturb (critical only)")
    host.notifications_dnd_cb.setChecked(_notif_settings.get_notifications_dnd())
    host.notifications_dnd_cb.toggled.connect(host._on_notifications_dnd_toggled)
    add_settings_full_width_row(master_form, host.notifications_dnd_cb)
    notif_layout.addWidget(master_card)

    # --- Behavior card ---
    behavior_card, behavior_card_layout = begin_settings_section_card(host, is_dark=is_dark)
    behavior_form = add_settings_card_form(behavior_card_layout)
    add_subsection_to_form(behavior_form, "Behavior")

    host.notifications_suppress_focus_cb = QCheckBox(
        "Suppress info/success while app is focused"
    )
    host.notifications_suppress_focus_cb.setChecked(
        _notif_settings.get_notifications_suppress_when_focused()
    )
    host.notifications_suppress_focus_cb.toggled.connect(
        _notif_settings.set_notifications_suppress_when_focused
    )
    add_settings_full_width_row(behavior_form, host.notifications_suppress_focus_cb)

    host.notifications_os_hidden_cb = QCheckBox("OS notifications when hidden")
    host.notifications_os_hidden_cb.setChecked(
        _notif_settings.get_notifications_os_when_hidden()
    )
    host.notifications_os_hidden_cb.toggled.connect(
        _notif_settings.set_notifications_os_when_hidden
    )
    add_settings_full_width_row(behavior_form, host.notifications_os_hidden_cb)

    host.notifications_sound_cb = QCheckBox("Play alert sounds")
    host.notifications_sound_cb.setChecked(_notif_settings.get_notifications_sound_enabled())
    host.notifications_sound_cb.toggled.connect(_notif_settings.set_notifications_sound_enabled)
    add_settings_full_width_row(behavior_form, host.notifications_sound_cb)

    host.notifications_preview_cb = QCheckBox("Show message preview in notifications")
    host.notifications_preview_cb.setChecked(_notif_settings.get_notifications_show_preview())
    host.notifications_preview_cb.toggled.connect(_notif_settings.set_notifications_show_preview)
    add_settings_full_width_row(behavior_form, host.notifications_preview_cb)
    notif_layout.addWidget(behavior_card)

    # --- Categories card ---
    categories_card, categories_card_layout = begin_settings_section_card(host, is_dark=is_dark)
    categories_form = add_settings_card_form(categories_card_layout)
    add_subsection_to_form(categories_form, "Categories")

    host.notifications_memory_cb = QCheckBox("Memory extraction notifications")
    host.notifications_memory_cb.setChecked(_notif_settings.get_notifications_category_memory())
    host.notifications_memory_cb.toggled.connect(_notif_settings.set_notifications_category_memory)
    add_settings_full_width_row(categories_form, host.notifications_memory_cb)
    notif_layout.addWidget(categories_card)

    # --- Actions card ---
    actions_card, actions_card_layout = begin_settings_section_card(host, is_dark=is_dark)
    actions_form = add_settings_card_form(actions_card_layout)
    add_subsection_to_form(actions_form, "Actions")

    clear_history_btn = make_settings_page_action_button("Clear notification history")
    clear_history_btn.clicked.connect(host._clear_notification_history)
    host.notifications_clear_history_btn = clear_history_btn
    add_settings_full_width_row(actions_form, clear_history_btn)
    notif_layout.addWidget(actions_card)

    add_section_reset_footer(notif_layout, host, "notifications", is_dark=is_dark)

    return notif_widget
