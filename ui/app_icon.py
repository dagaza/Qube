"""Resolve the Qube icon used for windows and the Windows taskbar."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QGuiApplication, QIcon, QPixmap
from PyQt6.QtWidgets import QApplication, QWidget

from core.paths import resource_path

logger = logging.getLogger("Qube.UI.AppIcon")

_WINDOW_ICON_SIZES_PX = (16, 24, 32, 48, 64, 128, 256)


def resolve_qube_window_icon_path() -> Path | None:
    """Prefer ``qube.ico`` on Windows for correct taskbar rendering."""
    if sys.platform == "win32":
        ico = resource_path("assets", "logos", "qube.ico")
        if ico.is_file():
            return ico
    for rel in (
        ("assets", "logos", "qube_logo_256.png"),
        ("assets", "icons", "qube_logo_256.png"),
        ("assets", "qube_logo_256.png"),
    ):
        candidate = resource_path(*rel)
        if candidate.is_file():
            return candidate
    return None


def _png_source_path() -> Path | None:
    for rel in (
        ("assets", "logos", "qube_logo_256.png"),
        ("assets", "icons", "qube_logo_256.png"),
        ("assets", "qube_logo_256.png"),
    ):
        candidate = resource_path(*rel)
        if candidate.is_file():
            return candidate
    return None


def qube_window_icon() -> QIcon:
    path = resolve_qube_window_icon_path()
    if path is not None and path.suffix.lower() == ".ico":
        icon = QIcon(str(path))
        if not icon.isNull():
            return icon

    png = _png_source_path()
    if png is None:
        return QIcon()

    source = QPixmap(str(png))
    if source.isNull():
        return QIcon()

    icon = QIcon()
    for size in _WINDOW_ICON_SIZES_PX:
        icon.addPixmap(
            source.scaled(
                size,
                size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
    return icon


def apply_window_branding(widget: QWidget) -> None:
    """Set title + icon on a top-level window when assets are available."""
    icon = qube_window_icon()
    if not icon.isNull():
        widget.setWindowIcon(icon)
    if not (widget.windowTitle() or "").strip():
        widget.setWindowTitle("Qube")


def apply_windows_taskbar_icon(widget: QWidget) -> None:
    """Force the taskbar icon on Windows (needed for python.exe / frameless shells)."""
    if sys.platform != "win32":
        return
    path = resolve_qube_window_icon_path()
    if path is None or not path.is_file():
        return
    try:
        import ctypes

        hwnd = int(widget.winId())
        if hwnd == 0:
            return

        LR_LOADFROMFILE = 0x0010
        LR_DEFAULTSIZE = 0x0040
        IMAGE_ICON = 1
        WM_SETICON = 0x0080
        ICON_SMALL = 0
        ICON_BIG = 1
        user32 = ctypes.windll.user32

        for size in (32, 16):
            hicon = user32.LoadImageW(
                None,
                str(path),
                IMAGE_ICON,
                size,
                size,
                LR_LOADFROMFILE,
            )
            if not hicon:
                hicon = user32.LoadImageW(
                    None,
                    str(path),
                    IMAGE_ICON,
                    0,
                    0,
                    LR_LOADFROMFILE | LR_DEFAULTSIZE,
                )
            if hicon:
                which = ICON_SMALL if size <= 16 else ICON_BIG
                user32.SendMessageW(hwnd, WM_SETICON, which, hicon)
    except Exception as exc:
        logger.debug("Windows taskbar icon apply failed: %s", exc)


def finalize_window_branding(widget: QWidget) -> None:
    """Apply branding after the native window handle exists (call from showEvent)."""
    apply_window_branding(widget)
    apply_windows_taskbar_icon(widget)


def apply_linux_desktop_integration(app: QApplication | None = None) -> None:
    """Help Linux panels pick up the Qube icon when not launched via a .desktop file."""
    if sys.platform != "linux":
        return
    app = app or QApplication.instance()
    if app is None:
        return
    app.setApplicationName("Qube")
    app.setApplicationDisplayName("Qube")
    try:
        QGuiApplication.setDesktopFileName("qube")
    except Exception as exc:
        logger.debug("Linux desktop file name apply failed: %s", exc)
