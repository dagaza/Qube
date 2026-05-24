"""System tray icon, menu, and presence visualization."""

from __future__ import annotations

from typing import Callable

import qtawesome as qta
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import QMenu, QSystemTrayIcon, QWidget

from core import app_settings
from core.assistant_activity import (
    AssistantActivity,
    menu_status_line,
    tray_tooltip_for_activity,
)


_ICON_COLORS: dict[AssistantActivity, str] = {
    AssistantActivity.ASSISTANT_OFF: "#64748b",
    AssistantActivity.IDLE_LISTEN: "#89b4fa",
    AssistantActivity.CAPTURING: "#f38ba8",
    AssistantActivity.WORKING: "#74c7ec",
    AssistantActivity.SPEAKING: "#a6e3a1",
    AssistantActivity.NEEDS_ATTENTION: "#f9e2af",
    AssistantActivity.ERROR: "#f38ba8",
    AssistantActivity.BACKGROUND_BUSY: "#cba6f7",
}


class TrayController(QWidget):
    """Owns QSystemTrayIcon state, menu, and animation ticks."""

    open_requested = pyqtSignal()
    exit_requested = pyqtSignal()
    restart_requested = pyqtSignal()
    voice_input_toggled = pyqtSignal(bool)
    voice_output_toggled = pyqtSignal(bool)
    dnd_toggled = pyqtSignal(bool)
    navigate_requested = pyqtSignal(str)  # action_id

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        voice_input_enabled: Callable[[], bool] | None = None,
        voice_output_enabled: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(parent)
        self._is_dark = True
        self._activity = AssistantActivity.IDLE_LISTEN
        self._voice_paused = False
        self._pulse_on = False
        self._tray_available = QSystemTrayIcon.isSystemTrayAvailable()
        self._tray_icon: QSystemTrayIcon | None = None
        self._status_action: QAction | None = None
        self._voice_in_action: QAction | None = None
        self._voice_out_action: QAction | None = None
        self._dnd_action: QAction | None = None
        self._recent_menu: QMenu | None = None
        self._voice_input_enabled = voice_input_enabled or (lambda: True)
        self._voice_output_enabled = voice_output_enabled or (lambda: True)

        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(2000)
        self._anim_timer.timeout.connect(self._toggle_pulse)
        self._work_anim_timer = QTimer(self)
        self._work_anim_timer.setInterval(700)
        self._work_anim_timer.timeout.connect(self._toggle_work_pulse)
        self._work_pulse_on = False

        if self._tray_available:
            self._build_tray()
        else:
            self._anim_timer.stop()

    @property
    def available(self) -> bool:
        return self._tray_available and self._tray_icon is not None

    @property
    def tray_icon(self) -> QSystemTrayIcon | None:
        return self._tray_icon

    def apply_theme(self, is_dark: bool) -> None:
        self._is_dark = is_dark
        menu_bg = "#1e1e2e" if is_dark else "#ffffff"
        menu_fg = "#cdd6f4" if is_dark else "#1e293b"
        if self._tray_icon is not None:
            menu = self._tray_icon.contextMenu()
            if menu is not None:
                menu.setStyleSheet(
                    f"QMenu {{ background-color: {menu_bg}; color: {menu_fg}; }}"
                    f"QMenu::item:selected {{ background-color: {'#313244' if is_dark else '#e2e8f0'}; }}"
                )
        self._refresh_icon()

    def set_activity(self, activity: AssistantActivity, *, voice_paused: bool = False) -> None:
        self._activity = activity
        self._voice_paused = voice_paused
        self._refresh_presence()
        if activity in (AssistantActivity.IDLE_LISTEN, AssistantActivity.BACKGROUND_BUSY):
            self._work_anim_timer.stop()
            self._work_pulse_on = False
            if not self._anim_timer.isActive():
                self._anim_timer.start()
        elif activity == AssistantActivity.WORKING:
            self._anim_timer.stop()
            self._pulse_on = False
            if not self._work_anim_timer.isActive():
                self._work_anim_timer.start()
        else:
            self._anim_timer.stop()
            self._work_anim_timer.stop()
            self._pulse_on = False
            self._work_pulse_on = False

    def update_recent_notifications(self, items: list[tuple[str, str]]) -> None:
        if self._recent_menu is None:
            return
        self._recent_menu.clear()
        if not items:
            empty = self._recent_menu.addAction("No recent notifications")
            empty.setEnabled(False)
            return
        for title, body in items:
            label = title if not body else f"{title} — {body[:60]}"
            action = self._recent_menu.addAction(label)
            action.setEnabled(False)

    def _build_tray(self) -> None:
        self._tray_icon = QSystemTrayIcon(self)
        self._tray_icon.setToolTip("Qube")

        menu = QMenu()
        self._status_action = QAction("Listening", self)
        self._status_action.setEnabled(False)
        menu.addAction(self._status_action)
        menu.addSeparator()

        open_action = QAction("Open Qube", self)
        open_action.triggered.connect(self.open_requested.emit)
        menu.addAction(open_action)
        menu.addSeparator()

        self._voice_in_action = QAction("Voice input", self)
        self._voice_in_action.setCheckable(True)
        self._voice_in_action.setChecked(self._voice_input_enabled())
        self._voice_in_action.triggered.connect(self._on_voice_in_toggled)
        menu.addAction(self._voice_in_action)

        self._voice_out_action = QAction("Voice responses", self)
        self._voice_out_action.setCheckable(True)
        self._voice_out_action.setChecked(self._voice_output_enabled())
        self._voice_out_action.triggered.connect(self._on_voice_out_toggled)
        menu.addAction(self._voice_out_action)

        menu.addSeparator()

        notif_menu = menu.addMenu("Notifications")
        self._dnd_action = QAction("Do Not Disturb", self)
        self._dnd_action.setCheckable(True)
        self._dnd_action.setChecked(app_settings.get_notifications_dnd())
        self._dnd_action.triggered.connect(self._on_dnd_toggled)
        notif_menu.addAction(self._dnd_action)

        open_settings = QAction("Notification settings…", self)
        open_settings.triggered.connect(lambda: self.navigate_requested.emit("open_settings"))
        notif_menu.addAction(open_settings)

        self._recent_menu = notif_menu.addMenu("Recent")
        self.update_recent_notifications([])

        menu.addSeparator()

        restart_action = QAction("Restart Qube", self)
        restart_action.triggered.connect(self.restart_requested.emit)
        menu.addAction(restart_action)

        quit_action = QAction("Exit Qube", self)
        quit_action.triggered.connect(self.exit_requested.emit)
        menu.addAction(quit_action)

        self._tray_icon.setContextMenu(menu)
        self._tray_icon.activated.connect(self._on_activated)
        self.apply_theme(self._is_dark)
        self._tray_icon.show()

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self.open_requested.emit()

    def _on_voice_in_toggled(self, checked: bool) -> None:
        self.voice_input_toggled.emit(checked)

    def _on_voice_out_toggled(self, checked: bool) -> None:
        self.voice_output_toggled.emit(checked)

    def _on_dnd_toggled(self, checked: bool) -> None:
        app_settings.set_notifications_dnd(checked)
        self.dnd_toggled.emit(checked)

    def sync_voice_toggles(self, *, voice_in: bool, voice_out: bool) -> None:
        if self._voice_in_action is not None:
            self._voice_in_action.blockSignals(True)
            self._voice_in_action.setChecked(voice_in)
            self._voice_in_action.blockSignals(False)
        if self._voice_out_action is not None:
            self._voice_out_action.blockSignals(True)
            self._voice_out_action.setChecked(voice_out)
            self._voice_out_action.blockSignals(False)

    def sync_dnd_toggle(self) -> None:
        if self._dnd_action is not None:
            self._dnd_action.blockSignals(True)
            self._dnd_action.setChecked(app_settings.get_notifications_dnd())
            self._dnd_action.blockSignals(False)

    def hide_tray(self) -> None:
        if self._tray_icon is not None:
            self._tray_icon.hide()

    def _toggle_pulse(self) -> None:
        self._pulse_on = not self._pulse_on
        self._refresh_icon()

    def _toggle_work_pulse(self) -> None:
        self._work_pulse_on = not self._work_pulse_on
        self._refresh_icon()

    def _refresh_presence(self) -> None:
        line = menu_status_line(self._activity, voice_paused=self._voice_paused)
        if self._status_action is not None:
            self._status_action.setText(f"● {line}")
        if self._tray_icon is not None:
            self._tray_icon.setToolTip(tray_tooltip_for_activity(self._activity, voice_paused=self._voice_paused))
        self._refresh_icon()

    def _refresh_icon(self) -> None:
        if self._tray_icon is None:
            return
        color = _ICON_COLORS.get(self._activity, "#89b4fa")
        if self._pulse_on and self._activity in (
            AssistantActivity.IDLE_LISTEN,
            AssistantActivity.BACKGROUND_BUSY,
        ):
            color = "#b4d0fb" if self._activity == AssistantActivity.IDLE_LISTEN else "#dcc6fa"
        icon_name = "fa5s.cube"
        if self._activity == AssistantActivity.CAPTURING:
            icon_name = "fa5s.microphone"
        elif self._activity == AssistantActivity.WORKING:
            icon_name = "fa5s.brain" if not self._work_pulse_on else "fa5s.cog"
            if self._work_pulse_on:
                color = "#89b4fa"
        elif self._activity == AssistantActivity.SPEAKING:
            icon_name = "fa5s.volume-up"
        elif self._activity == AssistantActivity.BACKGROUND_BUSY:
            icon_name = "fa5s.cloud-upload-alt"
        elif self._activity in (AssistantActivity.NEEDS_ATTENTION, AssistantActivity.ERROR):
            icon_name = "fa5s.exclamation-circle"
        self._tray_icon.setIcon(qta.icon(icon_name, color=color))
