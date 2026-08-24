"""
Application-owned tooltips (QFrame popup).

Native QToolTip + QSS is unreliable under frameless/translucent shells on some
Linux compositors; we intercept QEvent.ToolTip (QHelpEvent) and paint our own.
"""

from __future__ import annotations

from PyQt6.QtCore import QEvent, QObject, QPoint, QRect, QSize, Qt, QTimer
from PyQt6.QtGui import QCursor, QHelpEvent
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
import weakref

# Long tooltips wrap at this cap; shorter strings shrink to content width.
_TOOLTIP_MAX_WIDTH_PX = 420
_TOOLTIP_CLIP_MARGIN_PX = 8

def _tooltip_label_width_px(natural_text_width: int) -> int:
    """Clamp label width: compact for short tips, capped for multi-line copy."""
    return min(max(int(natural_text_width), 1), _TOOLTIP_MAX_WIDTH_PX)


def _tooltip_widget_and_text(widget: QWidget) -> tuple[QWidget, str]:
    """Resolve tooltip from widget or ancestor (e.g. QSpinBox line edit → spinbox)."""
    current: QWidget | None = widget
    while current is not None:
        raw = current.toolTip()
        if raw and str(raw).strip():
            return current, str(raw)
        current = current.parentWidget()
    return widget, ""


def _tooltip_clip_rect(anchor: QWidget | None) -> QRect | None:
    """Global clip rect for tooltips when an ancestor sets ``qube_tooltip_clip``."""
    if anchor is None:
        return None
    widget: QWidget | None = anchor
    while widget is not None:
        if bool(widget.property("qube_tooltip_clip")):
            top_left = widget.mapToGlobal(QPoint(0, 0))
            rect = QRect(top_left, widget.size())
            return rect.adjusted(
                _TOOLTIP_CLIP_MARGIN_PX,
                _TOOLTIP_CLIP_MARGIN_PX,
                -_TOOLTIP_CLIP_MARGIN_PX,
                -_TOOLTIP_CLIP_MARGIN_PX,
            )
        widget = widget.parentWidget()
    return None


def _clamp_tip_position(
    pos: QPoint,
    sz: QSize,
    bounds: QRect | None,
) -> QPoint:
    if bounds is None or bounds.isEmpty():
        return pos
    x, y = pos.x(), pos.y()
    max_x = bounds.right() - sz.width() + 1
    max_y = bounds.bottom() - sz.height() + 1
    if max_x < bounds.left():
        x = bounds.left()
    else:
        x = min(x, max_x)
        x = max(x, bounds.left())
    if max_y < bounds.top():
        y = bounds.top()
    else:
        y = min(y, max_y)
        y = max(y, bounds.top())
    return QPoint(x, y)


def _tooltip_text_height(text: str, label: QLabel, label_w: int) -> int:
    """Wrapped text height using QTextDocument (matches QLabel word-wrap layout)."""
    from PyQt6.QtGui import QTextDocument

    label.ensurePolished()
    doc = QTextDocument()
    doc.setDefaultFont(label.font())
    doc.setPlainText(text)
    doc.setTextWidth(float(label_w))
    doc.setDocumentMargin(0)
    fm = label.fontMetrics()
    # Pad by one line to avoid clipping the last descender / wrapped row.
    return max(int(doc.size().height()) + fm.leading() + 2, fm.height())

_ET_TOOLTIP = int(QEvent.Type.ToolTip)
_ET_HIDE_IMMEDIATE = frozenset({
    int(QEvent.Type.MouseButtonPress),
    int(QEvent.Type.Wheel),
    int(QEvent.Type.WindowDeactivate),
    int(QEvent.Type.FocusOut),
})
_ET_HIDE_IF_LEFT = frozenset({
    int(QEvent.Type.Leave),
    int(QEvent.Type.HoverLeave),
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

        if QubeToolTipController._initializing:
            return super().notify(receiver, event)

        ctrl = QubeToolTipController.instance()
        if et == _ET_TOOLTIP:
            if isinstance(receiver, QWidget):
                anchor, text = _tooltip_widget_and_text(receiver)
                if text:
                    if isinstance(event, QHelpEvent):
                        gpos = event.globalPos()
                    else:
                        gpos = QCursor.pos()
                    ctrl.show_tip(anchor, gpos, text)
                    return True
        if et in _ET_HIDE_IMMEDIATE:
            ctrl.hide_tip()
        elif et in _ET_HIDE_IF_LEFT:
            ctrl.hide_if_cursor_left_anchor()
        elif et in _ET_MOVE:
            ctrl.hide_if_cursor_left_anchor()
        return super().notify(receiver, event)


class QubeToolTipController(QObject):
    _instance: QubeToolTipController | None = None
    _initializing: bool = False

    @classmethod
    def instance(cls) -> QubeToolTipController:
        if cls._instance is not None:
            try:
                cls._instance.objectName()
            except RuntimeError:
                cls._instance = None
        if cls._instance is None:
            cls._initializing = True
            try:
                parent = QApplication.instance()
                cls._instance = QubeToolTipController(parent)
            finally:
                cls._initializing = False
        return cls._instance

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._popup: QWidget | None = None
        self._shell: QFrame | None = None
        self._label: QLabel | None = None
        self._anchor_ref: weakref.ReferenceType[QWidget] | None = None
        self._hide_timer: QTimer | None = None
        self._refine_seq: int = 0
        self._refine_anchor_pos = QPoint()
        self._is_dark = True
        self._theme: ResolvedTheme | None = None

    def _ensure_hide_timer(self) -> QTimer:
        if self._hide_timer is None:
            self._hide_timer = QTimer(self)
            self._hide_timer.setSingleShot(True)
            self._hide_timer.timeout.connect(self.hide_tip)
        return self._hide_timer

    def set_dark_theme(self, is_dark: bool) -> None:
        self.set_theme(is_dark=is_dark)

    def set_theme(
        self,
        *,
        is_dark: bool | None = None,
        theme: "ResolvedTheme | None" = None,
    ) -> None:
        from core.theme.accessors import theme_for

        resolved = theme_for(
            is_dark=is_dark if is_dark is not None else self._is_dark,
            resolved=theme,
        )
        self._is_dark = resolved.is_dark
        self._theme = resolved
        self._apply_shell_style(resolved)

    def _apply_shell_style(self, theme: "ResolvedTheme | None" = None) -> None:
        if self._shell is None:
            return
        from core.theme.accessors import theme_for

        resolved = theme or self._theme or theme_for(is_dark=self._is_dark)
        self._shell.setStyleSheet(
            "QFrame#QubeToolTipFrame {"
            f" background-color: {resolved.tooltip_bg};"
            f" border: 1px solid {resolved.tooltip_border};"
            " border-radius: 6px;"
            "}"
            "QLabel#QubeToolTipLabel {"
            f" color: {resolved.text_primary};"
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
        self._label.setTextFormat(Qt.TextFormat.PlainText)
        layout = QVBoxLayout(self._shell)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(0)
        layout.addWidget(self._label)
        self._apply_shell_style()

    def _reset_label_constraints(self) -> None:
        if self._label is None:
            return
        self._label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        self._label.setMinimumSize(0, 0)
        self._label.setMaximumSize(16777215, 16777215)

    def _natural_text_width(self, text: str) -> int:
        assert self._label is not None
        self._label.ensurePolished()
        fm = self._label.fontMetrics()
        lines = text.split("\n") or [""]
        return max(fm.horizontalAdvance(line) for line in lines)

    def _label_content_size(self, text: str) -> QSize:
        """Compute wrapped label size; avoids stale fixed heights between tips."""
        assert self._label is not None
        self._reset_label_constraints()
        self._label.setText(text)
        label_w = _tooltip_label_width_px(self._natural_text_width(text))
        label_h = _tooltip_text_height(text, self._label, label_w)
        return QSize(label_w, label_h)

    def _size_tip_to_content(self) -> QSize:
        assert self._popup is not None and self._label is not None and self._shell is not None
        text = self._label.text()
        content = self._label_content_size(text)
        self._label.setFixedSize(content)
        shell_layout = self._shell.layout()
        if shell_layout is not None:
            shell_layout.activate()
            margins = shell_layout.contentsMargins()
            shell_size = QSize(
                content.width() + margins.left() + margins.right(),
                content.height() + margins.top() + margins.bottom(),
            )
        else:
            shell_size = content
        self._shell.setFixedSize(shell_size)
        self._popup.setFixedSize(shell_size)
        return shell_size

    def _place_tip(
        self,
        help_global_pos: QPoint,
        sz: QSize,
        anchor: QWidget | None = None,
    ) -> QPoint:
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
        clip = _tooltip_clip_rect(anchor)
        if clip is not None:
            p = _clamp_tip_position(p, sz, clip)
        return p

    def _refine_tip_if_still_current(self, token: int) -> None:
        if token != self._refine_seq:
            return
        if self._popup is None or self._label is None or not self._popup.isVisible():
            return
        anchor = self._anchor_ref() if self._anchor_ref is not None else None
        sz = self._size_tip_to_content()
        self._popup.move(self._place_tip(self._refine_anchor_pos, sz, anchor=anchor))

    def show_tip(self, anchor: QWidget, global_pos: QPoint, text: str) -> None:
        self._ensure_popup()
        assert self._popup is not None and self._label is not None
        self._anchor_ref = weakref.ref(anchor)
        hide_timer = self._ensure_hide_timer()
        hide_timer.stop()
        self._refine_seq += 1
        refine_token = self._refine_seq
        self._refine_anchor_pos = QPoint(global_pos)

        self._label.setText(text)
        sz = self._size_tip_to_content()
        self._popup.move(self._place_tip(global_pos, sz, anchor=anchor))
        self._popup.show()
        self._popup.raise_()
        hide_timer.start(15_000)
        QTimer.singleShot(0, lambda t=refine_token: self._refine_tip_if_still_current(t))

    def hide_tip(self) -> None:
        if self._hide_timer is not None:
            self._hide_timer.stop()
        self._refine_seq += 1
        self._anchor_ref = None
        self._reset_label_constraints()
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
    QubeToolTipController.instance().set_theme(is_dark=is_dark)
