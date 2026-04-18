"""
Application-owned tooltips (QFrame popup).

Native QToolTip + QSS is unreliable under frameless/translucent shells on some
Linux compositors; we intercept QEvent.ToolTip (QHelpEvent) and paint our own.
"""

from __future__ import annotations

from PyQt6.QtCore import QEvent, QObject, QPoint, QSize, Qt, QTimer
from PyQt6.QtGui import QCursor, QHelpEvent
from PyQt6.QtWidgets import QApplication, QFrame, QLabel, QVBoxLayout, QWidget
import weakref

_TOOLTIP_TEXT_WIDTH_PX = 300

_ET_TOOLTIP = int(QEvent.Type.ToolTip)
_ET_HIDE = frozenset({
    int(QEvent.Type.MouseButtonPress),
    int(QEvent.Type.Wheel),
    int(QEvent.Type.Leave),
    int(QEvent.Type.HoverLeave),
    int(QEvent.Type.WindowDeactivate),
    int(QEvent.Type.FocusOut),
})
_ET_MOVE = frozenset({
    int(QEvent.Type.MouseMove),
    int(QEvent.Type.HoverMove),
})


class QubeApplication(QApplication):
    """Routes tooltips through QubeToolTipController instead of native QToolTip."""

    def notify(self, receiver: QObject, event: QEvent) -> bool:
        try:
            et = int(event.type())
        except RecursionError:
            return super().notify(receiver, event)

        ctrl = QubeToolTipController.instance()
        if et == _ET_TOOLTIP:
            if isinstance(receiver, QWidget):
                raw = receiver.toolTip()
                if raw and str(raw).strip():
                    if isinstance(event, QHelpEvent):
                        gpos = event.globalPos()
                    else:
                        gpos = QCursor.pos()
                    ctrl.show_tip(receiver, gpos, str(raw))
                    return True
        if et in _ET_HIDE:
            ctrl.hide_tip()
        elif et in _ET_MOVE:
            ctrl.hide_if_cursor_left_anchor()
        return super().notify(receiver, event)


class QubeToolTipController(QObject):
    _instance: QubeToolTipController | None = None

    @classmethod
    def instance(cls) -> QubeToolTipController:
        if cls._instance is None:
            cls._instance = QubeToolTipController()
        return cls._instance

    def __init__(self) -> None:
        super().__init__()
        self._popup: QWidget | None = None
        self._shell: QFrame | None = None
        self._label: QLabel | None = None
        self._anchor_ref: weakref.ReferenceType[QWidget] | None = None
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide_tip)
        self._refine_seq: int = 0
        self._refine_anchor_pos = QPoint()
        self._is_dark = True

    def set_dark_theme(self, is_dark: bool) -> None:
        self._is_dark = is_dark
        self._apply_shell_style()

    def _apply_shell_style(self) -> None:
        if self._shell is None:
            return
        if self._is_dark:
            self._shell.setStyleSheet(
                "QFrame#QubeToolTipFrame {"
                " background-color: #1e1e2e;"
                " border: 1px solid #89b4fa;"
                " border-radius: 6px;"
                "}"
                "QLabel#QubeToolTipLabel {"
                " color: #cdd6f4;"
                " background: transparent;"
                " border: none;"
                " padding: 0px;"
                " font-size: 11px;"
                "}"
            )
        else:
            self._shell.setStyleSheet(
                "QFrame#QubeToolTipFrame {"
                " background-color: #ffffff;"
                " border: 1px solid #cbd5e1;"
                " border-radius: 6px;"
                "}"
                "QLabel#QubeToolTipLabel {"
                " color: #1e293b;"
                " background: transparent;"
                " border: none;"
                " padding: 0px;"
                " font-size: 11px;"
                "}"
            )

    def _ensure_popup(self) -> None:
        if self._popup is not None:
            return
        self._popup = QWidget()
        self._popup.setWindowFlags(
            Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint
        )
        # Transparent host removes the visible square backdrop behind rounded corners.
        self._popup.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._popup.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        # Tooltip must never steal pointer events from its anchor widget.
        self._popup.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        root_layout = QVBoxLayout(self._popup)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self._shell = QFrame(self._popup)
        self._shell.setObjectName("QubeToolTipFrame")
        root_layout.addWidget(self._shell)

        self._label = QLabel(self._shell)
        self._label.setObjectName("QubeToolTipLabel")
        self._label.setWordWrap(True)
        self._label.setMinimumWidth(_TOOLTIP_TEXT_WIDTH_PX)
        self._label.setMaximumWidth(_TOOLTIP_TEXT_WIDTH_PX)
        self._label.setTextFormat(Qt.TextFormat.PlainText)
        layout = QVBoxLayout(self._shell)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(0)
        layout.addWidget(self._label)
        self._apply_shell_style()

    def _reset_label_vertical_constraints(self) -> None:
        if self._label is None:
            return
        self._label.setMinimumHeight(0)
        self._label.setMaximumHeight(16777215)

    def _size_tip_to_content(self) -> QSize:
        assert self._popup is not None and self._label is not None and self._shell is not None
        self._reset_label_vertical_constraints()
        self._label.ensurePolished()
        tw = self._label.maximumWidth()
        hfw = self._label.heightForWidth(tw)
        if hfw > 0:
            self._label.setFixedHeight(hfw)
        shell_layout = self._shell.layout()
        if shell_layout is not None:
            shell_layout.activate()
        self._popup.adjustSize()
        sz = self._popup.sizeHint()
        self._popup.resize(sz)
        return sz

    def _place_tip(self, help_global_pos: QPoint, sz: QSize) -> QPoint:
        offset = QPoint(12, 18)
        p = help_global_pos + offset
        screen = QApplication.screenAt(p) or QApplication.primaryScreen()
        geo = screen.availableGeometry() if screen else None
        if geo is not None:
            x, y = p.x(), p.y()
            if x + sz.width() > geo.right():
                x = max(geo.left(), geo.right() - sz.width())
            if y + sz.height() > geo.bottom():
                y = max(geo.top(), help_global_pos.y() - sz.height() - 8)
            p = QPoint(x, y)
        return p

    def _refine_tip_if_still_current(self, token: int) -> None:
        if token != self._refine_seq:
            return
        if self._popup is None or self._label is None or not self._popup.isVisible():
            return
        sz = self._size_tip_to_content()
        self._popup.move(self._place_tip(self._refine_anchor_pos, sz))

    def show_tip(self, _anchor: QWidget, global_pos: QPoint, text: str) -> None:
        self._ensure_popup()
        assert self._popup is not None and self._label is not None
        self._anchor_ref = weakref.ref(_anchor)
        self._hide_timer.stop()
        self._refine_seq += 1
        refine_token = self._refine_seq
        self._refine_anchor_pos = QPoint(global_pos)

        self._label.setText(text)
        sz = self._size_tip_to_content()
        self._popup.move(self._place_tip(global_pos, sz))
        self._popup.show()
        self._popup.raise_()
        self._hide_timer.start(15_000)
        QTimer.singleShot(0, lambda t=refine_token: self._refine_tip_if_still_current(t))

    def hide_tip(self) -> None:
        self._hide_timer.stop()
        self._refine_seq += 1
        self._anchor_ref = None
        self._reset_label_vertical_constraints()
        if self._popup is not None:
            self._popup.hide()

    def hide_if_cursor_left_anchor(self) -> None:
        ref = self._anchor_ref
        anchor = ref() if ref is not None else None
        if anchor is None or not anchor.isVisible():
            self.hide_tip()
            return
        # Prefer underMouse fast path; fall back to global-geometry containment.
        if anchor.underMouse():
            return
        p = anchor.mapFromGlobal(QCursor.pos())
        if not anchor.rect().contains(p):
            self.hide_tip()


def qube_tooltip_set_theme(is_dark: bool) -> None:
    QubeToolTipController.instance().set_dark_theme(is_dark)
