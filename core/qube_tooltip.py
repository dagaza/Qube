"""
Application-owned tooltips (QFrame popup).

Native QToolTip + QSS is unreliable under frameless/translucent shells on some
Linux compositors; we intercept QEvent.ToolTip (QHelpEvent) and paint our own.
"""

from __future__ import annotations

from PyQt6.QtCore import QEvent, QObject, QPoint, QSize, Qt, QTimer
from PyQt6.QtGui import QCursor, QHelpEvent
from PyQt6.QtWidgets import QApplication, QFrame, QLabel, QVBoxLayout, QWidget


class QubeApplication(QApplication):
    """Routes tooltips through QubeToolTipController instead of native QToolTip."""

    def notify(self, receiver: QObject, event: QEvent) -> bool:
        et = event.type()
        if et == QEvent.Type.ToolTip:
            if isinstance(receiver, QWidget):
                raw = receiver.toolTip()
                if raw and str(raw).strip():
                    if isinstance(event, QHelpEvent):
                        gpos = event.globalPos()
                    else:
                        gpos = QCursor.pos()
                    QubeToolTipController.instance().show_tip(
                        receiver, gpos, str(raw)
                    )
                    return True
        if et in (QEvent.Type.MouseButtonPress, QEvent.Type.Wheel):
            QubeToolTipController.instance().hide_tip()
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
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide_tip)
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
        root_layout = QVBoxLayout(self._popup)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self._shell = QFrame(self._popup)
        self._shell.setObjectName("QubeToolTipFrame")
        root_layout.addWidget(self._shell)

        self._label = QLabel(self._shell)
        self._label.setObjectName("QubeToolTipLabel")
        self._label.setWordWrap(True)
        self._label.setMaximumWidth(300)
        self._label.setTextFormat(Qt.TextFormat.PlainText)
        layout = QVBoxLayout(self._shell)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(0)
        layout.addWidget(self._label)
        self._apply_shell_style()

    def show_tip(self, _anchor: QWidget, global_pos: QPoint, text: str) -> None:
        self._ensure_popup()
        assert self._popup is not None and self._label is not None
        self._hide_timer.stop()
        self._label.setText(text)
        self._popup.adjustSize()
        offset = QPoint(12, 18)
        p = global_pos + offset
        sz: QSize = self._popup.sizeHint()
        screen = QApplication.screenAt(p) or QApplication.primaryScreen()
        geo = screen.availableGeometry() if screen else None
        if geo is not None:
            x, y = p.x(), p.y()
            if x + sz.width() > geo.right():
                x = max(geo.left(), geo.right() - sz.width())
            if y + sz.height() > geo.bottom():
                y = max(geo.top(), global_pos.y() - sz.height() - 8)
            p = QPoint(x, y)
        self._popup.move(p)
        self._popup.show()
        self._popup.raise_()
        self._hide_timer.start(15_000)

    def hide_tip(self) -> None:
        self._hide_timer.stop()
        if self._popup is not None:
            self._popup.hide()


def qube_tooltip_set_theme(is_dark: bool) -> None:
    QubeToolTipController.instance().set_dark_theme(is_dark)
