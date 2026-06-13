"""Notifications settings section."""

from __future__ import annotations

from PyQt6.QtWidgets import QCheckBox, QPushButton, QVBoxLayout, QWidget

from core import app_settings as _notif_settings
from ui.views.settings.widgets import add_subsection_to_layout


def build_section(host, *, is_dark: bool) -> QWidget:
    del is_dark

    notif_widget = QWidget()
    notif_widget.setObjectName("SettingsFormContainer")
    notif_layout = QVBoxLayout(notif_widget)
    notif_layout.setContentsMargins(15, 0, 15, 10)
    notif_layout.setSpacing(8)

    add_subsection_to_layout(notif_layout, "Master")

    host.notifications_enabled_cb = QCheckBox("Enable notifications")
    host.notifications_enabled_cb.setChecked(_notif_settings.get_notifications_enabled())
    host.notifications_enabled_cb.toggled.connect(_notif_settings.set_notifications_enabled)
    notif_layout.addWidget(host.notifications_enabled_cb)

    host.notifications_dnd_cb = QCheckBox("Do Not Disturb (critical only)")
    host.notifications_dnd_cb.setChecked(_notif_settings.get_notifications_dnd())
    host.notifications_dnd_cb.toggled.connect(host._on_notifications_dnd_toggled)
    notif_layout.addWidget(host.notifications_dnd_cb)

    add_subsection_to_layout(notif_layout, "Behavior")

    host.notifications_suppress_focus_cb = QCheckBox(
        "Suppress info/success while app is focused"
    )
    host.notifications_suppress_focus_cb.setChecked(
        _notif_settings.get_notifications_suppress_when_focused()
    )
    host.notifications_suppress_focus_cb.toggled.connect(
        _notif_settings.set_notifications_suppress_when_focused
    )
    notif_layout.addWidget(host.notifications_suppress_focus_cb)

    host.notifications_os_hidden_cb = QCheckBox("OS notifications when hidden")
    host.notifications_os_hidden_cb.setChecked(
        _notif_settings.get_notifications_os_when_hidden()
    )
    host.notifications_os_hidden_cb.toggled.connect(
        _notif_settings.set_notifications_os_when_hidden
    )
    notif_layout.addWidget(host.notifications_os_hidden_cb)

    host.notifications_sound_cb = QCheckBox("Play alert sounds")
    host.notifications_sound_cb.setChecked(_notif_settings.get_notifications_sound_enabled())
    host.notifications_sound_cb.toggled.connect(_notif_settings.set_notifications_sound_enabled)
    notif_layout.addWidget(host.notifications_sound_cb)

    host.notifications_preview_cb = QCheckBox("Show message preview in notifications")
    host.notifications_preview_cb.setChecked(_notif_settings.get_notifications_show_preview())
    host.notifications_preview_cb.toggled.connect(_notif_settings.set_notifications_show_preview)
    notif_layout.addWidget(host.notifications_preview_cb)

    add_subsection_to_layout(notif_layout, "Categories")

    host.notifications_memory_cb = QCheckBox("Memory extraction notifications")
    host.notifications_memory_cb.setChecked(_notif_settings.get_notifications_category_memory())
    host.notifications_memory_cb.toggled.connect(_notif_settings.set_notifications_category_memory)
    notif_layout.addWidget(host.notifications_memory_cb)

    add_subsection_to_layout(notif_layout, "Actions")

    clear_history_btn = QPushButton("Clear notification history")
    clear_history_btn.clicked.connect(host._clear_notification_history)
    notif_layout.addWidget(clear_history_btn)

    return notif_widget
