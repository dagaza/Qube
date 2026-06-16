"""Work-area bounds that respect OS panels and taskbars."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys

from PyQt6.QtCore import QRect
from PyQt6.QtGui import QScreen

logger = logging.getLogger("Qube.Platform")

# Used only when Qt and X11 both report the full monitor rectangle.
_LINUX_PANEL_FALLBACK_PX = 48


def _clamp_rect_to_monitor(rect: QRect, monitor: QRect) -> QRect:
    left = max(rect.left(), monitor.left())
    top = max(rect.top(), monitor.top())
    right = min(rect.right(), monitor.right())
    bottom = min(rect.bottom(), monitor.bottom())
    if right < left or bottom < top:
        return monitor
    return QRect(left, top, right - left + 1, bottom - top + 1)


def _linux_session_type() -> str:
    if not sys.platform.startswith("linux"):
        return ""
    return os.environ.get("XDG_SESSION_TYPE", "").lower()


def _x11_net_workarea_rect() -> QRect | None:
    """Read EWMH _NET_WORKAREA from the root window (X11 only)."""
    if _linux_session_type() == "wayland":
        return None
    try:
        proc = subprocess.run(
            ["xprop", "-root", "_NET_WORKAREA"],
            capture_output=True,
            text=True,
            timeout=1.0,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout:
            return None
        match = re.search(r"_NET_WORKAREA\(CARDINAL\) = ([\d, ]+)", proc.stdout)
        if not match:
            return None
        vals = [int(v.strip()) for v in match.group(1).split(",") if v.strip()]
        if len(vals) < 4:
            return None
        left, top, right, bottom = vals[0], vals[1], vals[2], vals[3]
        if right <= left or bottom <= top:
            return None
        return QRect(left, top, right - left, bottom - top)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        logger.debug("xprop _NET_WORKAREA lookup failed: %s", exc)
        return None


def _linux_fallback_bounds(monitor: QRect) -> QRect:
    """Conservative shrink when Qt and xprop both report the full monitor."""
    bottom = monitor.bottom() - _LINUX_PANEL_FALLBACK_PX
    bottom = max(bottom, monitor.top())
    return QRect(monitor.left(), monitor.top(), monitor.width(), bottom - monitor.top() + 1)


def workspace_bounds_for_screen(screen: QScreen) -> QRect:
    """
    Monitor bounds excluding panels/taskbars when possible.

    Qt's QScreen.availableGeometry() often equals geometry() on Linux X11 when
    the window manager does not publish _NET_WORKAREA to Qt. For frameless
    maximize we must not treat that as permission to cover the panel bar.
    """
    monitor = screen.geometry()
    avail = screen.availableGeometry()

    if avail != monitor:
        return _clamp_rect_to_monitor(avail, monitor)

    if sys.platform.startswith("linux"):
        x11_area = _x11_net_workarea_rect()
        if x11_area is not None:
            intersected = monitor.intersected(x11_area)
            if (
                intersected.isValid()
                and intersected.width() > 0
                and intersected.height() > 0
                and intersected != monitor
            ):
                return intersected
        return _linux_fallback_bounds(monitor)

    return monitor


def parse_net_workarea_line(line: str) -> QRect | None:
    """Parse `xprop -root _NET_WORKAREA` output for tests and diagnostics."""
    match = re.search(r"_NET_WORKAREA\(CARDINAL\) = ([\d, ]+)", line)
    if not match:
        return None
    vals = [int(v.strip()) for v in match.group(1).split(",") if v.strip()]
    if len(vals) < 4:
        return None
    left, top, right, bottom = vals[0], vals[1], vals[2], vals[3]
    if right <= left or bottom <= top:
        return None
    return QRect(left, top, right - left, bottom - top)
