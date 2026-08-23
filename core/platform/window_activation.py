"""Raise an existing top-level window when another instance requests activation."""

from __future__ import annotations

import logging
import sys

from PyQt6.QtWidgets import QWidget

logger = logging.getLogger("Qube.Platform.WindowActivation")


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
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        user32.SetForegroundWindow(hwnd)
    except Exception as exc:
        logger.debug("Windows foreground activation failed: %s", exc)
