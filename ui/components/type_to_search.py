"""Route printable keys into a view's primary text field (search bar or composer)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Union

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QLineEdit,
    QPlainTextEdit,
    QSpinBox,
    QTextBrowser,
    QTextEdit,
    QWidget,
)

TypeToFocusTarget = Union[QLineEdit, QPlainTextEdit, QTextEdit]

_NON_PRINTABLE_KEYS = frozenset(
    {
        Qt.Key.Key_Escape,
        Qt.Key.Key_Tab,
        Qt.Key.Key_Backtab,
        Qt.Key.Key_Return,
        Qt.Key.Key_Enter,
        Qt.Key.Key_Delete,
        Qt.Key.Key_Backspace,
        Qt.Key.Key_Home,
        Qt.Key.Key_End,
        Qt.Key.Key_PageUp,
        Qt.Key.Key_PageDown,
        Qt.Key.Key_Insert,
        Qt.Key.Key_CapsLock,
        Qt.Key.Key_NumLock,
        Qt.Key.Key_Up,
        Qt.Key.Key_Down,
        Qt.Key.Key_Left,
        Qt.Key.Key_Right,
    }
)

_TEXT_ENTRY_TYPES = (
    QLineEdit,
    QPlainTextEdit,
    QTextEdit,
    QTextBrowser,
    QSpinBox,
    QDoubleSpinBox,
)


def is_type_to_search_key(event: QKeyEvent) -> bool:
    """Return True when the key should begin or extend a type-to-focus query."""
    modifiers = event.modifiers()
    if modifiers & (
        Qt.KeyboardModifier.ControlModifier
        | Qt.KeyboardModifier.AltModifier
        | Qt.KeyboardModifier.MetaModifier
    ):
        return False
    key = event.key()
    if key in _NON_PRINTABLE_KEYS:
        return False
    if Qt.Key.Key_F1 <= key <= Qt.Key.Key_F35:
        return False
    text = event.text()
    return bool(text) and text.isprintable()


def _focused_text_entry(focus: QWidget | None) -> QWidget | None:
    if focus is None:
        return None
    if isinstance(focus, _TEXT_ENTRY_TYPES):
        return focus
    parent = focus.parentWidget()
    while parent is not None:
        if isinstance(parent, _TEXT_ENTRY_TYPES):
            return parent
        parent = parent.parentWidget()
    return None


def _is_focus_on_target(focus: QWidget | None, target: TypeToFocusTarget) -> bool:
    if focus is None:
        return False
    if focus is target:
        return True
    return target.isAncestorOf(focus)


def should_handle_type_to_focus(
    host: QWidget,
    target: TypeToFocusTarget,
    event: QKeyEvent,
    *,
    extra_block: Callable[[], bool] | None = None,
) -> bool:
    """Return True when a key press should be redirected into ``target``."""
    if not host.isVisible() or not target.isEnabled():
        return False
    if not is_type_to_search_key(event):
        return False
    if extra_block is not None and extra_block():
        return False

    app = QApplication.instance()
    if app is not None:
        if app.activeModalWidget() is not None:
            return False
        if app.activePopupWidget() is not None:
            return False

    win = host.window()
    if win is not None:
        tour = getattr(win, "_active_tour", None)
        if tour is not None and getattr(tour, "is_active", False):
            return False

    focus = app.focusWidget() if app is not None else None
    if _is_focus_on_target(focus, target):
        return False

    text_entry = _focused_text_entry(focus)
    if text_entry is not None and text_entry is not target:
        return False

    return True


def focus_target_with_key(target: TypeToFocusTarget, event: QKeyEvent) -> None:
    """Move keyboard focus to ``target`` and insert the typed character."""
    target.setFocus(Qt.FocusReason.ShortcutFocusReason)
    text = event.text()
    if not text:
        return
    if isinstance(target, QLineEdit):
        target.insert(text)
    else:
        target.textCursor().insertText(text)


class TypeToFocusFilter(QObject):
    """App- and view-scoped filter that routes printable keys into a text field."""

    def __init__(
        self,
        host: QWidget,
        target: TypeToFocusTarget,
        *,
        extra_block: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(host)
        self._host = host
        self._target = target
        self._extra_block = extra_block

    def attach(self) -> None:
        self._host.installEventFilter(self)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # type: ignore[override]
        if event.type() != QEvent.Type.KeyPress:
            return False
        if not isinstance(event, QKeyEvent):
            return False
        if not should_handle_type_to_focus(
            self._host,
            self._target,
            event,
            extra_block=self._extra_block,
        ):
            return False
        focus_target_with_key(self._target, event)
        event.accept()
        return True


def install_type_to_focus(
    host: QWidget,
    target: TypeToFocusTarget,
    *,
    extra_block: Callable[[], bool] | None = None,
) -> TypeToFocusFilter:
    """Install type-to-focus behavior on ``host`` targeting ``target``."""
    handler = TypeToFocusFilter(host, target, extra_block=extra_block)
    handler.attach()
    return handler


def install_type_to_search(host: QWidget, search: QLineEdit) -> TypeToFocusFilter:
    """Install type-to-search behavior on ``host`` targeting a ``QLineEdit``."""
    return install_type_to_focus(host, search)


# Backward-compatible aliases
should_handle_type_to_search = should_handle_type_to_focus
focus_search_with_key = focus_target_with_key
TypeToSearchFilter = TypeToFocusFilter
