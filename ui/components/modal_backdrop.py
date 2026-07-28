"""Shared fullscreen dim layer for modal dialogs over the main window."""

from __future__ import annotations

from PyQt6.QtCore import QEasingCurve, QEvent, QPropertyAnimation, Qt
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import QGraphicsOpacityEffect, QWidget

from core.theme.overlay import overlay_scrim_qcolor


def resolve_modal_backdrop_host(widget: QWidget | None) -> QWidget | None:
    """Walk parent widgets until we find the object that owns the modal backdrop."""
    current = widget
    while current is not None:
        if hasattr(current, "acquire_modal_backdrop") and hasattr(
            current, "release_modal_backdrop"
        ):
            return current
        current = current.parentWidget()
    return None


class ModalBackdrop(QWidget):
    """Theme-aware scrim that dims the main window behind modal dialogs."""

    _FADE_MS = 180

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._is_dark = True
        self._scrim = overlay_scrim_qcolor(is_dark=True)
        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity_effect)
        self._fade_anim: QPropertyAnimation | None = None
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.hide()
        if parent is not None:
            parent.installEventFilter(self)

    def apply_theme(self, is_dark: bool) -> None:
        self._is_dark = bool(is_dark)
        self._scrim = overlay_scrim_qcolor(is_dark=is_dark)
        self.update()

    def _dim_color(self):
        return self._scrim

    def eventFilter(self, obj, event) -> bool:
        parent = self.parentWidget()
        if obj is parent and event.type() == QEvent.Type.Resize:
            self._sync_geometry()
        return super().eventFilter(obj, event)

    def _sync_geometry(self) -> None:
        parent = self.parentWidget()
        if parent is not None:
            self.setGeometry(parent.rect())

    def _stop_fade(self) -> None:
        if self._fade_anim is not None:
            self._fade_anim.stop()
            self._fade_anim = None

    def _animate_opacity(
        self,
        start: float,
        end: float,
        *,
        on_finished=None,
    ) -> None:
        self._stop_fade()
        anim = QPropertyAnimation(self._opacity_effect, b"opacity", self)
        anim.setDuration(self._FADE_MS)
        anim.setStartValue(start)
        anim.setEndValue(end)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        if on_finished is not None:
            anim.finished.connect(on_finished)
        self._fade_anim = anim
        anim.start()

    def show_animated(self) -> None:
        self._sync_geometry()
        self._stop_fade()
        self._opacity_effect.setOpacity(0.0)
        self.show()
        self.raise_()
        self._animate_opacity(0.0, 1.0)

    def hide_animated(self) -> None:
        if not self.isVisible():
            self._opacity_effect.setOpacity(0.0)
            return

        def _finish() -> None:
            self.hide()
            self._opacity_effect.setOpacity(0.0)
            self._fade_anim = None

        current = float(self._opacity_effect.opacity())
        self._animate_opacity(current, 0.0, on_finished=_finish)

    def paintEvent(self, _event) -> None:
        if self.width() <= 0 or self.height() <= 0:
            return
        painter = QPainter(self)
        try:
            painter.fillRect(self.rect(), self._dim_color())
        finally:
            painter.end()
