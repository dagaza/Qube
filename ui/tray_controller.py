"""System tray icon, menu, and presence visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import qtawesome as qta
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QIcon, QPixmap
from PyQt6.QtWidgets import QMenu, QSystemTrayIcon, QWidget

from core import app_settings
from core.paths import resource_path
from core.assistant_activity import (
    AssistantActivity,
    menu_status_line,
    tray_tooltip_for_activity,
)

_TRAY_LOGO_NAME = "qube_logo_256.png"
_TRAY_ICON_SIZES_PX = (16, 22, 24, 32)
_TRAY_ICON_FALLBACK_COLOR = "#8b5cf6"


def resolve_qube_logo_path() -> Path | None:
    """Resolve the Qube logo across new and legacy asset directories."""
    for rel in (
        ("assets", "logos", _TRAY_LOGO_NAME),
        ("assets", "icons", _TRAY_LOGO_NAME),
        ("assets", _TRAY_LOGO_NAME),
    ):
        candidate = resource_path(*rel)
        if candidate.is_file():
            return candidate
    return None


def build_tray_logo_icon(logo_path: Path | str | None = None) -> QIcon:
    """Build a multi-resolution QIcon suitable for Linux panel trays."""
    path = Path(logo_path) if logo_path is not None else resolve_qube_logo_path()
    if path is None or not path.is_file():
        return qta.icon("fa5s.cube", color=_TRAY_ICON_FALLBACK_COLOR)

    source = QPixmap(str(path))
    if source.isNull():
        return qta.icon("fa5s.cube", color=_TRAY_ICON_FALLBACK_COLOR)

    icon = QIcon()
    for size in _TRAY_ICON_SIZES_PX:
        icon.addPixmap(
            source.scaled(
                size,
                size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
    return icon


class TrayController(QWidget):
    """Owns QSystemTrayIcon state, menu, and activity tooltip."""

    open_requested = pyqtSignal()
    exit_requested = pyqtSignal()
    restart_requested = pyqtSignal()
    voice_input_toggled = pyqtSignal(bool)
    voice_output_toggled = pyqtSignal(bool)
    dnd_toggled = pyqtSignal(bool)
    companion_toggled = pyqtSignal(bool)
    navigate_requested = pyqtSignal(str)  # action_id

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        voice_input_enabled: Callable[[], bool] | None = None,
        voice_output_enabled: Callable[[], bool] | None = None,
        tray_logo_path: Path | str | None = None,
    ) -> None:
        super().__init__(parent)
        self._is_dark = True
        self._activity = AssistantActivity.IDLE_LISTEN
        self._voice_paused = False
        self._tray_logo_icon = build_tray_logo_icon(tray_logo_path)
        self._tray_available = QSystemTrayIcon.isSystemTrayAvailable()
        self._tray_icon: QSystemTrayIcon | None = None
        self._status_action: QAction | None = None
        self._voice_in_action: QAction | None = None
        self._voice_out_action: QAction | None = None
        self._dnd_action: QAction | None = None
        self._companion_action: QAction | None = None
        self._recent_menu: QMenu | None = None
        self._voice_input_enabled = voice_input_enabled or (lambda: True)
        self._voice_output_enabled = voice_output_enabled or (lambda: True)

        if self._tray_available:
            self._build_tray()

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

    def set_activity(self, activity: AssistantActivity, *, voice_paused: bool = False) -> None:
        self._activity = activity
        self._voice_paused = voice_paused
        self._refresh_presence()

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
        self._tray_icon.setIcon(self._tray_logo_icon)
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

        companion_menu = menu.addMenu("Companion")
        self._companion_action = QAction("Show desktop companion", self)
        self._companion_action.setCheckable(True)
        self._companion_action.setChecked(app_settings.get_companion_enabled())
        self._companion_action.triggered.connect(self._on_companion_toggled)
        companion_menu.addAction(self._companion_action)

        companion_settings = QAction("Companion settings…", self)
        companion_settings.triggered.connect(
            lambda: self.navigate_requested.emit("open_settings")
        )
        companion_menu.addAction(companion_settings)

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
        self._refresh_presence()
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

    def _on_companion_toggled(self, checked: bool) -> None:
        app_settings.set_companion_enabled(checked)
        self.companion_toggled.emit(checked)

    def sync_companion_toggle(self) -> None:
        if self._companion_action is not None:
            self._companion_action.blockSignals(True)
            self._companion_action.setChecked(app_settings.get_companion_enabled())
            self._companion_action.blockSignals(False)

    def hide_tray(self) -> None:
        if self._tray_icon is not None:
            self._tray_icon.hide()

    def show_tray(self) -> None:
        """Ensure the tray icon is visible (e.g. after hide-to-tray from the panel close button)."""
        if self._tray_icon is not None:
            self._tray_icon.show()

    def _refresh_presence(self) -> None:
        line = menu_status_line(self._activity, voice_paused=self._voice_paused)
        if self._status_action is not None:
            self._status_action.setText(f"● {line}")
        if self._tray_icon is not None:
            self._tray_icon.setToolTip(tray_tooltip_for_activity(self._activity, voice_paused=self._voice_paused))

    def refresh_icon(self) -> None:
        """Re-apply the static Qube logo tray icon."""
        if self._tray_icon is not None and not self._tray_logo_icon.isNull():
            self._tray_icon.setIcon(self._tray_logo_icon)
