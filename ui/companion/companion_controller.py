"""Companion lifecycle controller — visibility, persistence, and policy wiring."""

from __future__ import annotations

import os
import sys
import time

from PyQt6.QtCore import QObject, QTimer, pyqtSignal, Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QApplication, QWidget

from core import app_settings
from core.assistant_presence import AssistantPresenceService, AssistantPresenceSnapshot
from core.companion_policy import plan_companion_visibility
from ui.companion.companion_window import CompanionWindow


_SNOOZE_ONE_HOUR_SEC = 3600


class CompanionController(QObject):
    """Owns CompanionWindow visibility and syncs with assistant presence."""

    open_requested = pyqtSignal()
    navigate_settings_requested = pyqtSignal()

    def __init__(
        self,
        presence: AssistantPresenceService,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._presence = presence
        self._window = CompanionWindow()
        self._main_window: QWidget | None = None
        self._user_visible = True
        self._snooze_until = 0.0
        self._idle_since: float | None = time.time()
        self._fullscreen_detected = False
        self._companion_visible_for_policy = False
        self._shutting_down = False

        self._window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._window.open_requested.connect(self.open_requested.emit)
        self._window.hide_for_one_hour_requested.connect(self._snooze_one_hour)
        self._window.snooze_requested.connect(self.navigate_settings_requested.emit)

        self._presence.presence_changed.connect(self._on_presence_changed)

        self._visibility_timer = QTimer(self)
        self._visibility_timer.setInterval(1000)
        self._visibility_timer.timeout.connect(self._refresh_visibility)

        self._idle_timer = QTimer(self)
        self._idle_timer.setInterval(500)
        self._idle_timer.timeout.connect(self._check_idle_fade)

        self._fullscreen_timer = QTimer(self)
        self._fullscreen_timer.setInterval(2000)
        self._fullscreen_timer.timeout.connect(self._detect_fullscreen)

    @property
    def window(self) -> CompanionWindow:
        return self._window

    @property
    def is_visible_for_policy(self) -> bool:
        return self._companion_visible_for_policy

    @property
    def is_shutting_down(self) -> bool:
        return self._shutting_down

    def bind_main_window(self, main_window: QWidget) -> None:
        self._main_window = main_window
        is_dark = getattr(main_window, "_is_dark_theme", True)
        self._window.apply_theme(is_dark)
        self._apply_reduced_motion()
        self._restore_position()
        self._visibility_timer.start()
        self._idle_timer.start()
        if app_settings.get_companion_suppress_on_fullscreen():
            self._fullscreen_timer.start()
        self._refresh_visibility()

    def apply_theme(self, is_dark: bool) -> None:
        self._window.apply_theme(is_dark)

    def on_main_hidden(self) -> None:
        self._refresh_visibility()

    def on_main_shown(self) -> None:
        self._refresh_visibility()

    def on_settings_changed(self) -> None:
        self._presence.refresh_platform_tier()
        self._apply_reduced_motion()
        dock = app_settings.get_companion_dock_mode()
        self._window.set_dock_mode(dock)
        self._refresh_visibility()

    def set_user_enabled(self, enabled: bool) -> None:
        """Persist companion on/off from tray or settings; clears snooze when enabling."""
        app_settings.set_companion_enabled(enabled)
        if enabled:
            self._user_visible = True
            self._snooze_until = 0.0
        self.on_settings_changed()

    def pulse_notification(self) -> None:
        if self._shutting_down:
            return
        if self._window.isVisible():
            self._window.pulse_notification()

    def _snooze_one_hour(self) -> None:
        self._snooze_until = time.time() + _SNOOZE_ONE_HOUR_SEC
        self._user_visible = False
        self._window.hide()
        self._companion_visible_for_policy = False

    def _apply_reduced_motion(self) -> None:
        override = app_settings.get_companion_reduced_motion()
        if override is not None:
            self._window.set_reduced_motion(override)
            return
        hints = QGuiApplication.styleHints()
        try:
            reduced = hints.useAnimations() is False
        except Exception:
            reduced = os.environ.get("QUBE_REDUCED_MOTION", "").strip() in ("1", "true", "yes")
        self._window.set_reduced_motion(reduced)

    def _on_presence_changed(self, snapshot: AssistantPresenceSnapshot) -> None:
        if self._shutting_down:
            return
        from core.assistant_activity import AssistantActivity

        if snapshot.activity == AssistantActivity.IDLE_LISTEN:
            if self._idle_since is None:
                self._idle_since = time.time()
        else:
            self._idle_since = None
            self._window.set_idle_faded(False)

        self._window.set_snapshot(snapshot)
        self._refresh_visibility()

    def _check_idle_fade(self) -> None:
        if self._shutting_down:
            return
        from core.assistant_activity import AssistantActivity
        from core.companion_policy import should_auto_hide

        snap = self._presence.snapshot()
        if should_auto_hide(snap, idle_since=self._idle_since):
            self._window.set_idle_faded(True)
        elif snap.activity != AssistantActivity.IDLE_LISTEN:
            self._window.set_idle_faded(False)

    def _main_visible(self) -> bool:
        if self._main_window is None:
            return True
        return self._main_window.isVisible()

    def _main_minimized(self) -> bool:
        if self._main_window is None:
            return False
        return self._main_window.isMinimized()

    def _refresh_visibility(self) -> None:
        if self._shutting_down:
            return
        snap = self._presence.snapshot()
        plan = plan_companion_visibility(
            snap,
            main_window_visible=self._main_visible(),
            main_window_minimized=self._main_minimized(),
            companion_user_visible=self._user_visible,
            fullscreen_detected=self._fullscreen_detected,
            snooze_until=self._snooze_until,
        )

        self._window.set_dock_mode(plan.use_dock_mode or app_settings.get_companion_dock_mode())

        if plan.show:
            if not self._window.isVisible():
                self._restore_position()
            self._window.show()
            self._companion_visible_for_policy = True
            if plan.auto_hide and self._idle_since:
                from core.companion_policy import should_auto_hide

                self._window.set_idle_faded(should_auto_hide(snap, idle_since=self._idle_since))
        else:
            self._window.hide()
            self._companion_visible_for_policy = False

    def _restore_position(self) -> None:
        pos = app_settings.get_companion_position()
        screen_name = pos.get("screen") or ""
        target_screen = None
        for screen in QApplication.screens():
            if screen.name() == screen_name:
                target_screen = screen
                break
        if target_screen is None:
            target_screen = QApplication.primaryScreen()
        if target_screen is None:
            return

        geo = target_screen.availableGeometry()
        x = pos.get("x")
        y = pos.get("y")
        if x is None or y is None:
            norm_x = pos.get("norm_x")
            norm_y = pos.get("norm_y")
            if norm_x is not None and norm_y is not None:
                x = int(geo.left() + float(norm_x) * geo.width())
                y = int(geo.top() + float(norm_y) * geo.height())
            else:
                x = geo.right() - self._window.width() - 24
                y = geo.bottom() - self._window.height() - 24

        edge = str(pos.get("dock_edge") or "none")
        if edge == "left":
            x = geo.left() + 4
        elif edge == "right":
            x = geo.right() - self._window.width() - 4
        elif edge == "bottom":
            y = geo.bottom() - self._window.height() - 4

        self._window.move(int(x), int(y))

    def _detect_fullscreen(self) -> None:
        if not app_settings.get_companion_suppress_on_fullscreen():
            self._fullscreen_detected = False
            return

        if sys.platform == "win32":
            self._fullscreen_detected = self._detect_fullscreen_windows()
        elif sys.platform == "darwin":
            self._fullscreen_detected = self._detect_fullscreen_macos()
        else:
            self._fullscreen_detected = False

        self._refresh_visibility()

    @staticmethod
    def _detect_fullscreen_windows() -> bool:
        try:
            import ctypes

            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return False
            rect = ctypes.wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            sw = user32.GetSystemMetrics(0)
            sh = user32.GetSystemMetrics(1)
            return (rect.right - rect.left) >= sw and (rect.bottom - rect.top) >= sh
        except Exception:
            return False

    @staticmethod
    def _detect_fullscreen_macos() -> bool:
        try:
            from AppKit import NSWorkspace

            app = NSWorkspace.sharedWorkspace().frontmostApplication()
            if app is None:
                return False
            name = str(app.localizedName() or "")
            return name not in ("Qube", "Python", "Cursor")
        except Exception:
            return False

    def shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        self._companion_visible_for_policy = False

        if self._window.isVisible():
            pos = self._window.pos()
            screen = QApplication.screenAt(self._window.orb_center_global())
            screen_name = screen.name() if screen else ""
            geo = screen.availableGeometry() if screen else None
            norm_x = None
            norm_y = None
            if geo is not None:
                norm_x = (pos.x() - geo.left()) / max(1, geo.width())
                norm_y = (pos.y() - geo.top()) / max(1, geo.height())
            app_settings.set_companion_position(
                x=pos.x(),
                y=pos.y(),
                screen=screen_name,
                norm_x=norm_x,
                norm_y=norm_y,
            )

        try:
            self._presence.presence_changed.disconnect(self._on_presence_changed)
        except TypeError:
            pass

        self._visibility_timer.stop()
        self._idle_timer.stop()
        self._fullscreen_timer.stop()
        self._window.hide()
        self._window.close()
