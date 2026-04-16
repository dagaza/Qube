"""SelectorButton: a self-contained dropdown-style QPushButton.

Replaces the legacy ``QPushButton + setIcon(fa chevron) + setLayoutDirection(RightToLeft)``
pattern for dropdown selectors. That pattern has a long-standing Qt quirk where
QSS ``padding`` / ``text-align`` values are silently dropped when an icon is
attached in RTL layout, causing the label to hug the left inner edge and the
chevron to hug the right inner edge with zero horizontal inset.

Rendering strategy
------------------
We do NOT rely on Qt's native QPushButton text layout at all, because different
Qt styles honor QSS ``padding`` inconsistently when combined with icons or menu
indicators. Instead we:

1. Ask QStyle to draw only the background / border (``opt.text = ""``).
2. Paint the label text ourselves with an explicit left inset.
3. Paint the chevron ourselves on the right.

This gives pixel-precise control over both the text inset and the chevron inset
regardless of the active Qt style or theme.

Object name
-----------
Uses ``QubeSelectorButton`` -- a unique name that does NOT collide with the
global ``#SettingsMenuButton`` rules in ``assets/styles/*.qss``, so this widget
has no dependency on, and cannot regress, any other dropdown in the app.

Theming
-------
The button stores its own text / chevron colors and sets its own stylesheet in
``apply_theme(is_dark)``. To cope with construction happening before the widget
is parented to the main window (common in settings views built eagerly at
startup), ``showEvent`` re-checks ``window()._is_dark_theme`` and re-applies the
theme if needed.
"""

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen
from PyQt6.QtWidgets import QPushButton, QStyle, QStyleOptionButton


class SelectorButton(QPushButton):
    # Horizontal insets used when we manually paint the label / chevron.
    # PADDING_LEFT matches the global `#SettingsMenuButton` QSS (`padding: 8px 15px;`)
    # so this button visually lines up with every other dropdown in the app.
    # PADDING_RIGHT is larger than PADDING_LEFT only to reserve room for the
    # painted chevron; it does not affect the button's outer size.
    PADDING_LEFT = 15
    PADDING_RIGHT = 32
    CHEVRON_WIDTH = 9.0
    CHEVRON_HEIGHT = 5.0
    CHEVRON_RIGHT_INSET = 14.0
    CHEVRON_STROKE = 1.6

    def __init__(self, text: str = "", parent=None, is_dark: bool = True):
        super().__init__(text, parent)
        self.setObjectName("QubeSelectorButton")
        self.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self._is_dark = is_dark
        self._text_color = QColor("#cdd6f4")
        self._text_disabled_color = QColor("#64748b")
        self._chevron_active = QColor("#64748b")
        self._chevron_muted = QColor("#94a3b8")
        self.apply_theme(is_dark)

    def apply_theme(self, is_dark: bool) -> None:
        self._is_dark = is_dark
        if is_dark:
            bg = "rgba(0, 0, 0, 0.2)"
            border = "rgba(255, 255, 255, 0.08)"
            disabled_bg = "rgba(0, 0, 0, 0.12)"
            disabled_border = "rgba(255, 255, 255, 0.04)"
            self._text_color = QColor("#cdd6f4")
            self._text_disabled_color = QColor("#64748b")
            self._chevron_muted = QColor("#3f3f46")
        else:
            bg = "#ffffff"
            border = "#cbd5e1"
            disabled_bg = "#f1f5f9"
            disabled_border = "#e2e8f0"
            self._text_color = QColor("#1e293b")
            self._text_disabled_color = QColor("#94a3b8")
            self._chevron_muted = QColor("#a1a1aa")
        self._chevron_active = QColor("#64748b")

        # `padding: 8px 15px;` matches the global `#SettingsMenuButton` rule so
        # this selector has the same outer height / left inset as every other
        # dropdown in the app. We still custom-paint the text (see paintEvent),
        # but the QSS padding is what drives Qt's sizeHint for the button.
        self.setStyleSheet(
            f"""
            QPushButton#QubeSelectorButton {{
                background-color: {bg};
                border: 1px solid {border};
                border-radius: 6px;
                padding: 8px 15px;
                text-align: left;
            }}
            QPushButton#QubeSelectorButton:disabled {{
                background-color: {disabled_bg};
                border: 1px solid {disabled_border};
            }}
            QPushButton#QubeSelectorButton::menu-indicator {{
                image: none;
                width: 0px;
            }}
            """
        )
        self.update()

    def showEvent(self, event):
        super().showEvent(event)
        window = self.window()
        resolved = getattr(window, "_is_dark_theme", None)
        if isinstance(resolved, bool) and resolved != self._is_dark:
            self.apply_theme(resolved)

    def paintEvent(self, event):
        opt = QStyleOptionButton()
        self.initStyleOption(opt)
        opt.text = ""
        opt.icon = QIcon()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        self.style().drawControl(QStyle.ControlElement.CE_PushButton, opt, painter, self)

        text_color = self._text_color if self.isEnabled() else self._text_disabled_color
        painter.setPen(text_color)
        painter.setFont(self.font())
        text_rect = self.rect().adjusted(self.PADDING_LEFT, 0, -self.PADDING_RIGHT, 0)
        elided = self.fontMetrics().elidedText(
            self.text(), Qt.TextElideMode.ElideRight, text_rect.width()
        )
        painter.drawText(
            text_rect,
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            elided,
        )

        chevron_color = self._chevron_active if self.isEnabled() else self._chevron_muted
        pen = QPen(chevron_color, self.CHEVRON_STROKE)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        r = self.rect()
        w = self.CHEVRON_WIDTH
        h = self.CHEVRON_HEIGHT
        cx = r.right() - self.CHEVRON_RIGHT_INSET - w / 2.0
        cy = r.center().y() + 0.5
        painter.drawLine(
            QPointF(cx - w / 2.0, cy - h / 2.0), QPointF(cx, cy + h / 2.0)
        )
        painter.drawLine(
            QPointF(cx, cy + h / 2.0), QPointF(cx + w / 2.0, cy - h / 2.0)
        )
        painter.end()
