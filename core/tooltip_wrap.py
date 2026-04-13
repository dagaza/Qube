"""Plain Qt tooltips ignore QSS max-width for word-wrap; wrap as rich text app-wide."""

from __future__ import annotations

import html
from typing import Optional

from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QWidget

# Keep in sync with QToolTip max-width in assets/styles/base.qss and light.qss
_DEFAULT_WRAP_PX = 300

_original_qwidget_set_tooltip = QWidget.setToolTip
_original_qaction_set_tooltip = QAction.setToolTip

_wrap_px = _DEFAULT_WRAP_PX


def _looks_like_rich_html(text: str) -> bool:
    s = text.lstrip()
    if not s:
        return False
    head = s[:120].lower()
    if head.startswith("<html") or head.startswith("<!doctype"):
        return True
    # Heuristic: explicit closing tag near start → assume author HTML
    if s[0] == "<" and "</" in s[:400]:
        return True
    return False


def _wrap_plain_tooltip(text: str) -> str:
    """Return rich-HTML tooltip so Qt lays out with width and wrapping."""
    esc = html.escape(text, quote=False).replace("\n", "<br/>")
    w = _wrap_px
    return (
        f"<html><head/><body style='margin:0;'>"
        f"<div style='max-width:{w}px; width:{w}px; white-space:pre-wrap;'>{esc}</div>"
        f"</body></html>"
    )


def _wrapped_widget_set_tooltip(self: QWidget, text: Optional[str]) -> None:
    if text is None or text == "":
        _original_qwidget_set_tooltip(self, text)
        return
    if _looks_like_rich_html(text):
        _original_qwidget_set_tooltip(self, text)
    else:
        _original_qwidget_set_tooltip(self, _wrap_plain_tooltip(text))


def _wrapped_action_set_tooltip(self: QAction, text: Optional[str]) -> None:
    if text is None or text == "":
        _original_qaction_set_tooltip(self, text)
        return
    if _looks_like_rich_html(text):
        _original_qaction_set_tooltip(self, text)
    else:
        _original_qaction_set_tooltip(self, _wrap_plain_tooltip(text))


_installed = False


def install_wrapping_tooltips(max_width_px: Optional[int] = None) -> None:
    """
    Monkey-patch QWidget.setToolTip / QAction.setToolTip once, before building UI.

    Qt stylesheet max-width on QToolTip does not reflow plain text; wrapping plain
    strings as minimal HTML fixes clipping while keeping QSS colors/borders.
    """
    global _wrap_px, _installed
    if _installed:
        return
    _installed = True
    if max_width_px is not None:
        _wrap_px = max(120, int(max_width_px))
    QWidget.setToolTip = _wrapped_widget_set_tooltip  # type: ignore[method-assign]
    QAction.setToolTip = _wrapped_action_set_tooltip  # type: ignore[method-assign]
