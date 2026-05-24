"""Platform OS notification adapter."""

from __future__ import annotations

import logging
import shutil
import subprocess

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QSystemTrayIcon

from core.notification_types import NotificationEvent, NotificationSeverity

logger = logging.getLogger("Qube.Notifications")


class OsNotificationAdapter:
    """Delivers notifications via tray balloon or libnotify on Linux."""

    def __init__(self, tray_icon: QSystemTrayIcon | None = None) -> None:
        self._tray_icon = tray_icon
        self._libnotify_available = shutil.which("notify-send") is not None
        self._warned_unavailable = False

    def set_tray_icon(self, tray_icon: QSystemTrayIcon | None) -> None:
        self._tray_icon = tray_icon

    def show(self, event: NotificationEvent) -> None:
        icon = self._icon_type(event.severity)
        if self._try_libnotify(event, icon):
            return
        self._try_tray_message(event, icon)

    def _icon_type(self, severity: NotificationSeverity) -> QSystemTrayIcon.MessageIcon:
        mapping = {
            NotificationSeverity.INFO: QSystemTrayIcon.MessageIcon.Information,
            NotificationSeverity.SUCCESS: QSystemTrayIcon.MessageIcon.Information,
            NotificationSeverity.WARNING: QSystemTrayIcon.MessageIcon.Warning,
            NotificationSeverity.ERROR: QSystemTrayIcon.MessageIcon.Critical,
            NotificationSeverity.CRITICAL: QSystemTrayIcon.MessageIcon.Critical,
        }
        return mapping.get(severity, QSystemTrayIcon.MessageIcon.Information)

    def _try_libnotify(self, event: NotificationEvent, icon: QSystemTrayIcon.MessageIcon) -> bool:
        if not self._libnotify_available:
            return False
        urgency = "normal"
        if event.severity in (NotificationSeverity.ERROR, NotificationSeverity.CRITICAL):
            urgency = "critical"
        elif event.severity == NotificationSeverity.WARNING:
            urgency = "low"
        try:
            subprocess.run(
                [
                    "notify-send",
                    "-a",
                    "Qube",
                    "-u",
                    urgency,
                    event.title,
                    event.body,
                ],
                check=False,
                timeout=3,
            )
            return True
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("libnotify failed: %s", exc)
            return False

    def _try_tray_message(self, event: NotificationEvent, icon: QSystemTrayIcon.MessageIcon) -> None:
        if self._tray_icon is None or not QSystemTrayIcon.isSystemTrayAvailable():
            if not self._warned_unavailable:
                logger.warning("OS notifications unavailable (no tray / notify-send).")
                self._warned_unavailable = True
            return
        self._tray_icon.showMessage(event.title, event.body, icon, 5000)
