"""Spotlight onboarding overlay with coach-mark panel for guided first-run tours."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PyQt6.QtCore import QPoint, QRect, QRectF, Qt, QTimer, QObject, QEvent, pyqtSignal
from PyQt6.QtGui import QColor, QKeyEvent, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

StepPredicate = Callable[[QWidget], bool] | None
StepCallback = Callable[[QWidget], None] | None
TargetGetter = Callable[[QWidget], QWidget | None] | None
BodyGetter = Callable[[QWidget], str] | None


@dataclass
class OnboardingStep:
    step_id: str
    title: str
    body: str
    target_getter: TargetGetter = None
    on_enter: StepCallback = None
    predicate: StepPredicate = None
    predicate_hint: str = ""
    body_getter: BodyGetter = None


def _pad_rect(rect: QRect, margin: int) -> QRect:
    if rect.isNull():
        return rect
    return rect.adjusted(-margin, -margin, margin, margin)


def _global_widget_rect(widget: QWidget | None, *, margin: int = 6) -> QRect:
    if widget is None or not widget.isVisible():
        return QRect()
    top_left = widget.mapToGlobal(QPoint(0, 0))
    return _pad_rect(QRect(top_left, widget.size()), margin)


class SpotlightOverlay(QWidget):
    """Fullscreen dim layer with a rounded cut-out around the active target."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._spotlight_global = QRect()
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

    def set_spotlight_global_rect(self, rect: QRect) -> None:
        self._spotlight_global = QRect(rect)
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.raise_()

    def paintEvent(self, _event) -> None:
        if self.width() <= 0 or self.height() <= 0:
            return
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            overlay_path = QPainterPath()
            overlay_path.addRect(QRectF(self.rect()))

            hole_rect = QRectF()
            if not self._spotlight_global.isNull():
                top_left = self.mapFromGlobal(self._spotlight_global.topLeft())
                hole_rect = QRectF(
                    QRect(top_left, self._spotlight_global.size())
                )
                hole_path = QPainterPath()
                hole_path.addRoundedRect(hole_rect, 10.0, 10.0)
                overlay_path = overlay_path.subtracted(hole_path)

            painter.fillPath(overlay_path, QColor(0, 0, 0, 175))

            if not hole_rect.isNull():
                painter.setPen(QPen(QColor("#89b4fa"), 2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRoundedRect(hole_rect, 10.0, 10.0)
        finally:
            painter.end()


class OnboardingCoachPanel(QFrame):
    escape_pressed = pyqtSignal()

    _TEXT_LABEL_VERTICAL_PAD = 6

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("OnboardingCoachPanel")
        self.setMinimumWidth(320)
        self.setMaximumWidth(420)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 16)
        layout.setSpacing(10)

        self.step_lbl = QLabel("")
        self.step_lbl.setObjectName("OnboardingCoachStep")
        self.title_lbl = QLabel("")
        self.title_lbl.setObjectName("OnboardingCoachTitle")
        self.title_lbl.setWordWrap(True)
        self.body_lbl = QLabel("")
        self.body_lbl.setObjectName("OnboardingCoachBody")
        self.body_lbl.setWordWrap(True)
        self.body_lbl.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )
        self.hint_lbl = QLabel("")
        self.hint_lbl.setObjectName("OnboardingCoachHint")
        self.hint_lbl.setWordWrap(True)
        self.hint_lbl.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )
        self.hint_lbl.hide()

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.skip_btn = QPushButton("Skip tour")
        self.skip_btn.setProperty("class", "SecondaryButton")
        self.back_btn = QPushButton("Back")
        self.back_btn.setProperty("class", "SecondaryButton")
        self.next_btn = QPushButton("Next")
        self.next_btn.setProperty("class", "PrimaryActionButton")
        btn_row.addWidget(self.skip_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.back_btn)
        btn_row.addWidget(self.next_btn)

        layout.addWidget(self.step_lbl)
        layout.addWidget(self.title_lbl)
        layout.addWidget(self.body_lbl)
        layout.addWidget(self.hint_lbl)
        layout.addLayout(btn_row)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def _content_inner_width(self) -> int:
        lay = self.layout()
        if lay is None:
            return self.maximumWidth() - 36
        m = lay.contentsMargins()
        return max(200, self.maximumWidth() - m.left() - m.right())

    def _label_wrapped_height(self, lbl: QLabel, content_w: int) -> int:
        if not lbl.text().strip():
            return 0
        h = lbl.heightForWidth(content_w)
        if h > 0:
            return h
        flags = int(Qt.TextFlag.TextWordWrap)
        rect = lbl.fontMetrics().boundingRect(
            0, 0, content_w, 10_000, flags, lbl.text()
        )
        return max(rect.height(), lbl.fontMetrics().lineSpacing())

    def recalculate_content_size(self) -> None:
        """Size word-wrapped labels before adjustSize (Qt under-measures wrapped QLabel)."""
        content_w = self._content_inner_width()
        pad = self._TEXT_LABEL_VERTICAL_PAD
        for lbl in (self.title_lbl, self.body_lbl, self.hint_lbl):
            if lbl.isHidden() or not lbl.text().strip():
                lbl.setMinimumHeight(0)
                continue
            lbl.setMinimumHeight(self._label_wrapped_height(lbl, content_w) + pad)
        self.adjustSize()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            event.accept()
            self.escape_pressed.emit()
            return
        super().keyPressEvent(event)

    def apply_theme(self, is_dark: bool) -> None:
        if is_dark:
            self.setStyleSheet(
                """
                QFrame#OnboardingCoachPanel {
                    background-color: #1e1e2e;
                    border: 1px solid rgba(137, 180, 250, 0.45);
                    border-radius: 12px;
                }
                QLabel#OnboardingCoachStep { color: #89b4fa; font-size: 11px; font-weight: 600; }
                QLabel#OnboardingCoachTitle { color: #cdd6f4; font-size: 15px; font-weight: 700; }
                QLabel#OnboardingCoachBody { color: rgba(205, 214, 244, 0.92); font-size: 13px; }
                QLabel#OnboardingCoachHint { color: #f9e2af; font-size: 12px; font-style: italic; }
                """
            )
        else:
            self.setStyleSheet(
                """
                QFrame#OnboardingCoachPanel {
                    background-color: #ffffff;
                    border: 1px solid #cbd5e1;
                    border-radius: 12px;
                }
                QLabel#OnboardingCoachStep { color: #2563eb; font-size: 11px; font-weight: 600; }
                QLabel#OnboardingCoachTitle { color: #1e293b; font-size: 15px; font-weight: 700; }
                QLabel#OnboardingCoachBody { color: #334155; font-size: 13px; }
                QLabel#OnboardingCoachHint { color: #b45309; font-size: 12px; font-style: italic; }
                """
            )


class _OnboardingKeyHandler(QObject):
    """Captures Escape while a tour is active, regardless of focus widget."""

    def __init__(self, tour: OnboardingTour) -> None:
        super().__init__(tour._host)
        self._tour = tour
        self._app_filter_installed = False

    def install(self) -> None:
        self._tour._host.installEventFilter(self)
        app = QApplication.instance()
        if app is not None and not self._app_filter_installed:
            app.installEventFilter(self)
            self._app_filter_installed = True

    def remove(self) -> None:
        self._tour._host.removeEventFilter(self)
        app = QApplication.instance()
        if app is not None and self._app_filter_installed:
            app.removeEventFilter(self)
            self._app_filter_installed = False

    def eventFilter(self, watched, event) -> bool:
        if not self._tour.is_active or event.type() != QEvent.Type.KeyPress:
            return False
        if not isinstance(event, QKeyEvent) or event.key() != Qt.Key.Key_Escape:
            return False
        modal = QApplication.activeModalWidget()
        if modal is not None:
            return False
        self._tour.skip()
        event.accept()
        return True


class OnboardingTour:
    """Runs a linear sequence of spotlight coach-mark steps over a host window."""

    def __init__(
        self,
        host: QWidget,
        steps: list[OnboardingStep],
        *,
        on_finished: Callable[[], None] | None = None,
    ) -> None:
        self._host = host
        self._steps = steps
        self._on_finished = on_finished
        self._index = 0
        self._active = False

        self._overlay = SpotlightOverlay(host)
        self._overlay.hide()
        self._panel = OnboardingCoachPanel(host)
        self._panel.hide()
        self._key_handler = _OnboardingKeyHandler(self)

        self._panel.escape_pressed.connect(self.skip)
        self._panel.skip_btn.clicked.connect(self.skip)
        self._panel.back_btn.clicked.connect(self.back)
        self._panel.next_btn.clicked.connect(self.next)

        self._refresh_timer = QTimer(host)
        self._refresh_timer.setInterval(250)
        self._refresh_timer.timeout.connect(self._refresh_step_ui)

    @property
    def is_active(self) -> bool:
        return self._active

    def start(self) -> None:
        if not self._steps:
            return
        self._active = True
        self._index = 0
        self._overlay.setGeometry(self._host.rect())
        self._overlay.show()
        self._panel.show()
        self._panel.raise_()
        self._overlay.raise_()
        self._panel.raise_()
        self._apply_theme()
        self._key_handler.install()
        self._enter_step(self._index)
        self._refresh_timer.start()
        self._host.activateWindow()
        QTimer.singleShot(50, self._panel.setFocus)

    def skip(self) -> None:
        self.finish()

    def finish(self) -> None:
        if not self._active:
            return
        self._active = False
        self._refresh_timer.stop()
        self._key_handler.remove()
        self._overlay.hide()
        self._panel.hide()
        if self._on_finished:
            self._on_finished()

    def back(self) -> None:
        if not self._active or self._index <= 0:
            return
        self._index -= 1
        self._enter_step(self._index)

    def next(self) -> None:
        if not self._active:
            return
        step = self._steps[self._index]
        if step.predicate and not step.predicate(self._host):
            return
        if self._index >= len(self._steps) - 1:
            self.finish()
            return
        self._index += 1
        self._enter_step(self._index)

    def refresh_layout(self) -> None:
        if self._active:
            self._refresh_step_ui()

    def _apply_theme(self) -> None:
        is_dark = getattr(self._host, "_is_dark_theme", True)
        self._panel.apply_theme(is_dark)

    def _enter_step(self, index: int) -> None:
        step = self._steps[index]
        if step.on_enter:
            step.on_enter(self._host)
        QTimer.singleShot(0, self._refresh_step_ui)

    def _refresh_step_ui(self) -> None:
        if not self._active:
            return
        self._overlay.setGeometry(self._host.rect())
        step = self._steps[self._index]
        target = step.target_getter(self._host) if step.target_getter else None
        if target is not None and target.isVisible():
            self._overlay.set_spotlight_global_rect(_global_widget_rect(target))
        else:
            self._overlay.set_spotlight_global_rect(QRect())

        total = len(self._steps)
        self._panel.step_lbl.setText(f"Step {self._index + 1} of {total}")
        self._panel.title_lbl.setText(step.title)
        body_text = step.body_getter(self._host) if step.body_getter else step.body
        self._panel.body_lbl.setText(body_text)

        predicate_ok = step.predicate is None or step.predicate(self._host)
        if step.predicate and not predicate_ok and step.predicate_hint:
            self._panel.hint_lbl.setText(step.predicate_hint)
            self._panel.hint_lbl.show()
        else:
            self._panel.hint_lbl.hide()

        self._panel.back_btn.setEnabled(self._index > 0)
        self._panel.next_btn.setEnabled(predicate_ok)
        self._panel.next_btn.setText(
            "Finish" if self._index >= total - 1 else "Next"
        )
        self._panel.recalculate_content_size()
        self._position_panel(_global_widget_rect(target) if target else QRect())
        self._panel.raise_()

    def _position_panel(self, target_global: QRect) -> None:
        margin = 16
        panel = self._panel
        pw, ph = panel.width(), panel.height()
        host_rect = self._host.rect()

        if target_global.isNull():
            x = (host_rect.width() - pw) // 2
            y = (host_rect.height() - ph) // 2
        else:
            local = QRect(
                self._host.mapFromGlobal(target_global.topLeft()),
                target_global.size(),
            )
            x = local.center().x() - pw // 2
            y = local.bottom() + margin
            if y + ph > host_rect.height() - margin:
                y = local.top() - ph - margin
            x = max(margin, min(x, host_rect.width() - pw - margin))
            y = max(margin, min(y, host_rect.height() - ph - margin))

        panel.move(x, y)
