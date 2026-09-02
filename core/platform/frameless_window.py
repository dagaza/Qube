"""Helpers for frameless translucent top-level Qt windows."""

from __future__ import annotations

import logging
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget

logger = logging.getLogger("Qube.Platform.FramelessWindow")

_DWMWA_WINDOW_CORNER_PREFERENCE = 33
_DWMWCP_DONOTROUND = 1


def frameless_dialog_window_flags() -> Qt.WindowType:
    """Window flags for prestige-style frameless modals."""
    return (
        Qt.WindowType.FramelessWindowHint
        | Qt.WindowType.Dialog
        | Qt.WindowType.NoDropShadowWindowHint
    )


def configure_frameless_dialog_host(widget: QWidget) -> None:
    """Apply standard frameless translucent host settings (call from ``__init__``)."""
    widget.setWindowFlags(frameless_dialog_window_flags())
    widget.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    widget.setAutoFillBackground(False)


def apply_frameless_dialog_chrome(widget: QWidget) -> None:
    """Re-apply borderless chrome once the native window handle exists (``showEvent``)."""
    apply_translucent_window_chrome(widget, transparent_stylesheet=True)


def apply_translucent_window_chrome(
    widget: QWidget,
    *,
    transparent_stylesheet: bool = True,
) -> None:
    """Keep a frameless host visually borderless (Qt attrs + Windows 11 DWM)."""
    widget.setAutoFillBackground(False)
    if transparent_stylesheet:
        if widget.objectName():
            name = widget.objectName()
            widget.setStyleSheet(
                f"QWidget#{name} {{ background: transparent; border: none; }}"
            )
        else:
            widget.setStyleSheet("background: transparent; border: none;")

    if sys.platform != "win32":
        return

    try:
        import ctypes

        hwnd = int(widget.winId())
        if hwnd == 0:
            return
        preference = ctypes.c_int(_DWMWCP_DONOTROUND)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd,
            _DWMWA_WINDOW_CORNER_PREFERENCE,
            ctypes.byref(preference),
            ctypes.sizeof(preference),
        )
    except Exception as exc:
        logger.debug("Windows translucent chrome apply failed: %s", exc)
