"""Raise an existing top-level window when another instance requests activation."""

from __future__ import annotations

import logging
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QApplication, QWidget

logger = logging.getLogger("Qube.Platform.WindowActivation")


def splash_window_flags() -> Qt.WindowType:
    """Window flags for frameless startup / bootstrap splash shells."""
    return (
        Qt.WindowType.SplashScreen
        | Qt.WindowType.FramelessWindowHint
        | Qt.WindowType.WindowStaysOnTopHint
    )


def screen_for_new_window():
    """Prefer the screen under the cursor; fall back to the primary screen."""
    app = QApplication.instance()
    if app is None:
        return None
    screen = app.screenAt(QCursor.pos())
    if screen is not None:
        return screen
    return app.primaryScreen()


def center_widget_on_screen(widget: QWidget, *, screen=None) -> None:
    """Center ``widget`` on ``screen`` (or :func:`screen_for_new_window`)."""
    target = screen if screen is not None else screen_for_new_window()
    if target is None:
        return
    available = target.availableGeometry()
    widget.adjustSize()
    frame = widget.frameGeometry()
    frame.moveCenter(available.center())
    left = max(
        available.left(),
        min(frame.left(), available.right() - frame.width() + 1),
    )
    top = max(
        available.top(),
        min(frame.top(), available.bottom() - frame.height() + 1),
    )
    widget.move(left, top)


def activate_toplevel_window(widget: QWidget | None) -> None:
    """Best-effort focus for an already-running Qube window."""
    if widget is None:
        return
    try:
        if widget.isMinimized():
            widget.showNormal()
        widget.show()
        widget.raise_()
        widget.activateWindow()
        if sys.platform == "win32":
            _win32_bring_to_front(widget)
    except RuntimeError:
        # Widget may already be destroyed during shutdown.
        return


def _win32_bring_to_front(widget: QWidget) -> None:
    try:
        import ctypes

        hwnd = int(widget.winId())
        if hwnd == 0:
            return

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        user32.BringWindowToTop(hwnd)

        foreground = user32.GetForegroundWindow()
        if foreground == hwnd:
            return

        foreground_thread = user32.GetWindowThreadProcessId(foreground, None)
        current_thread = kernel32.GetCurrentThreadId()
        attached = False
        if foreground and foreground_thread != current_thread:
            attached = bool(user32.AttachThreadInput(foreground_thread, current_thread, True))
        try:
            user32.SetForegroundWindow(hwnd)
        finally:
            if attached:
                user32.AttachThreadInput(foreground_thread, current_thread, False)
    except Exception as exc:
        logger.debug("Windows foreground activation failed: %s", exc)
