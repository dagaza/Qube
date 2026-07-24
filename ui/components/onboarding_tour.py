"""Spotlight onboarding overlay with coach-mark panel for guided first-run tours."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PyQt6.QtCore import QPoint, QRect, QRectF, Qt, QTimer, QObject, QEvent, pyqtSignal
from PyQt6.QtGui import QColor, QKeyEvent, QMouseEvent, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.theme.accessors import theme_for
from core.theme.overlay import overlay_scrim_qcolor
from core.theme.widget_styles import ONBOARDING_COACH_PANEL, ONBOARDING_SPOTLIGHT_RING

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


def _scroll_areas_for_widget(widget: QWidget | None) -> list[QScrollArea]:
    """Return scroll areas from innermost to outermost that contain widget."""
    areas: list[QScrollArea] = []
    current = widget.parentWidget() if widget is not None else None
    while current is not None:
        if isinstance(current, QScrollArea):
            areas.append(current)
        current = current.parentWidget()
    return areas


def _scroll_target_into_view(
    target: QWidget | None,
    *,
    x_margin: int = 16,
    y_margin: int = 48,
) -> None:
    """Scroll ancestor QScrollAreas so the spotlight target is fully visible."""
    if target is None or not target.isVisible():
        return
    for scroll in _scroll_areas_for_widget(target):
        scroll.ensureWidgetVisible(target, x_margin, y_margin)


def _target_menu_global_rect(target: QWidget | None) -> QRect:
    if target is None or not hasattr(target, "menu"):
        return QRect()
    menu = target.menu()
    if menu is None or not menu.isVisible():
        return QRect()
    return _global_widget_rect(menu, margin=0)


def _find_open_menu_global_rect(
    target: QWidget | None,
    target_global: QRect,
) -> QRect:
    """Return the visible dropdown menu tied to the spotlight target, if any."""
    attached = _target_menu_global_rect(target)
    if not attached.isNull():
        return attached

    app = QApplication.instance()
    if app is None:
        return QRect()

    from PyQt6.QtWidgets import QMenu

    popup = app.activePopupWidget()
    if isinstance(popup, QMenu) and popup.isVisible():
        popup_rect = _global_widget_rect(popup, margin=0)
        if not popup_rect.isNull():
            if target_global.isNull():
                return popup_rect
            anchor = QRect(
                target_global.left() - 48,
                target_global.top(),
                target_global.width() + 96,
                max(target_global.height() * 10, 280),
            )
            if popup_rect.intersects(anchor):
                return popup_rect

    if target_global.isNull():
        return QRect()

    anchor = QRect(
        target_global.left() - 48,
        target_global.top(),
        target_global.width() + 96,
        max(target_global.height() * 10, 280),
    )
    best = QRect()
    for widget in app.topLevelWidgets():
        if not isinstance(widget, QMenu) or not widget.isVisible():
            continue
        popup_rect = _global_widget_rect(widget, margin=0)
        if popup_rect.isNull() or not popup_rect.intersects(anchor):
            continue
        if best.isNull() or popup_rect.bottom() > best.bottom():
            best = popup_rect
    return best


def _dropdown_menu_step_active(
    target: QWidget | None,
    target_global: QRect,
) -> bool:
    return not _find_open_menu_global_rect(target, target_global).isNull()


def _clamp_panel_x(x: int, *, panel_width: int, host_rect: QRect, margin: int) -> int:
    return max(margin, min(x, host_rect.width() - panel_width - margin))


def _panel_fits_y(y: int, *, panel_height: int, host_rect: QRect, margin: int) -> bool:
    return margin <= y and y + panel_height <= host_rect.height() - margin


def _resolve_panel_position(
    host: QWidget,
    target_global: QRect,
    target: QWidget | None,
    *,
    panel_width: int,
    panel_height: int,
    margin: int,
) -> tuple[int, int]:
    host_rect = host.rect()

    if target_global.isNull():
        return (
            (host_rect.width() - panel_width) // 2,
            (host_rect.height() - panel_height) // 2,
        )

    local = QRect(
        host.mapFromGlobal(target_global.topLeft()),
        target_global.size(),
    )
    x_target = _clamp_panel_x(
        local.center().x() - panel_width // 2,
        panel_width=panel_width,
        host_rect=host_rect,
        margin=margin,
    )

    menu_global = _find_open_menu_global_rect(target, target_global)
    if not menu_global.isNull():
        # Dropdown steps: avoid covering the menu. Try above, then below the
        # menu, then dock to the bottom of the window.
        y_above = local.top() - panel_height - margin
        if _panel_fits_y(
            y_above, panel_height=panel_height, host_rect=host_rect, margin=margin
        ):
            return (x_target, y_above)

        menu_local = QRect(
            host.mapFromGlobal(menu_global.topLeft()),
            menu_global.size(),
        )
        y_below_menu = menu_local.bottom() + margin
        if _panel_fits_y(
            y_below_menu, panel_height=panel_height, host_rect=host_rect, margin=margin
        ):
            x_menu = _clamp_panel_x(
                menu_local.center().x() - panel_width // 2,
                panel_width=panel_width,
                host_rect=host_rect,
                margin=margin,
            )
            return (x_menu, y_below_menu)

        return (
            _clamp_panel_x(
                (host_rect.width() - panel_width) // 2,
                panel_width=panel_width,
                host_rect=host_rect,
                margin=margin,
            ),
            max(margin, host_rect.height() - panel_height - margin),
        )

    y_below = local.bottom() + margin
    if _panel_fits_y(
        y_below, panel_height=panel_height, host_rect=host_rect, margin=margin
    ):
        return (x_target, y_below)

    y_above = local.top() - panel_height - margin
    if _panel_fits_y(
        y_above, panel_height=panel_height, host_rect=host_rect, margin=margin
    ):
        return (x_target, y_above)

    return (
        _clamp_panel_x(
            (host_rect.width() - panel_width) // 2,
            panel_width=panel_width,
            host_rect=host_rect,
            margin=margin,
        ),
        max(margin, host_rect.height() - panel_height - margin),
    )


def _close_active_dropdowns(target: QWidget | None = None) -> None:
    """Close an open QMenu popup so coach-panel clicks advance in one press."""
    app = QApplication.instance()
    if app is not None:
        popup = app.activePopupWidget()
        if popup is not None:
            popup.close()

    if target is not None and hasattr(target, "menu"):
        menu = target.menu()
        if menu is not None and menu.isVisible():
            menu.close()


def _panel_button_at_global(
    panel: OnboardingCoachPanel,
    global_pos: QPoint,
) -> QPushButton | None:
    local = panel.mapFromGlobal(global_pos)
    if not panel.rect().contains(local):
        return None
    child = panel.childAt(local)
    while child is not None and not isinstance(child, QPushButton):
        parent = child.parentWidget()
        if parent is None or parent is panel:
            return None
        child = parent
    if isinstance(child, QPushButton) and child.isEnabled():
        return child
    return None


class SpotlightOverlay(QWidget):
    """Fullscreen dim layer with a rounded cut-out around the active target."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._spotlight_global = QRect()
        self._ring_color = theme_for(is_dark=True).qcolor_role(ONBOARDING_SPOTLIGHT_RING)
        self._scrim_color = overlay_scrim_qcolor(is_dark=True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def apply_theme(self, is_dark: bool) -> None:
        theme = theme_for(is_dark=is_dark)
        self._ring_color = theme.qcolor_role(ONBOARDING_SPOTLIGHT_RING)
        self._scrim_color = overlay_scrim_qcolor(theme)
        self.update()
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

            painter.fillPath(overlay_path, self._scrim_color)

            if not hole_rect.isNull():
                painter.setPen(QPen(self._ring_color, 2))
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
        """Stable wrapped height — do not use heightForWidth after minimumHeight is set."""
        if not lbl.text().strip():
            return 0
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
            lbl.setMinimumHeight(0)
            lbl.setMaximumHeight(16_777_215)
            if lbl.isHidden() or not lbl.text().strip():
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
        theme = theme_for(is_dark=is_dark)
        self.setStyleSheet(theme.style(ONBOARDING_COACH_PANEL))


class _OnboardingTourInputHandler(QObject):
    """Captures Escape and coach-panel clicks while a popup menu is open."""

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

    def _forward_panel_click(self, global_pos: QPoint) -> None:
        btn = _panel_button_at_global(self._tour._panel, global_pos)
        if btn is not None:
            btn.click()

    def eventFilter(self, watched, event) -> bool:
        if not self._tour.is_active:
            return False

        if event.type() == QEvent.Type.MouseButtonPress and isinstance(
            event, QMouseEvent
        ):
            if event.button() != Qt.MouseButton.LeftButton:
                return False
            app = QApplication.instance()
            popup = app.activePopupWidget() if app is not None else None
            if popup is None:
                return False
            panel = self._tour._panel
            if not panel.isVisible():
                return False
            global_pos = event.globalPosition().toPoint()
            panel_rect = QRect(panel.mapToGlobal(QPoint(0, 0)), panel.size())
            if not panel_rect.contains(global_pos):
                return False
            popup.close()
            QTimer.singleShot(0, lambda gp=global_pos: self._forward_panel_click(gp))
            event.accept()
            return True

        if event.type() != QEvent.Type.KeyPress:
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
        self._input_handler = _OnboardingTourInputHandler(self)

        self._panel.escape_pressed.connect(self.skip)
        self._panel.skip_btn.clicked.connect(self.skip)
        self._panel.back_btn.clicked.connect(self.back)
        self._panel.next_btn.clicked.connect(self.next)

        self._refresh_timer = QTimer(host)
        self._refresh_timer.setInterval(250)
        self._refresh_timer.timeout.connect(self._refresh_step_ui)

        self._last_panel_content_key: tuple[str, ...] | None = None

    @property
    def is_active(self) -> bool:
        return self._active

    def start(self) -> None:
        if not self._steps:
            return
        self._active = True
        self._index = 0
        self._last_panel_content_key = None
        self._overlay.setGeometry(self._host.rect())
        self._overlay.show()
        self._panel.show()
        self._panel.raise_()
        self._overlay.raise_()
        self._panel.raise_()
        self._apply_theme()
        self._input_handler.install()
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
        self._last_panel_content_key = None
        self._refresh_timer.stop()
        self._input_handler.remove()
        self._overlay.hide()
        self._panel.hide()
        if self._on_finished:
            self._on_finished()

    def back(self) -> None:
        if not self._active or self._index <= 0:
            return
        step = self._steps[self._index]
        target = step.target_getter(self._host) if step.target_getter else None
        _close_active_dropdowns(target)
        self._index -= 1
        self._enter_step(self._index)

    def next(self) -> None:
        if not self._active:
            return
        step = self._steps[self._index]
        if step.predicate and not step.predicate(self._host):
            return
        target = step.target_getter(self._host) if step.target_getter else None
        _close_active_dropdowns(target)
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
        self._overlay.apply_theme(is_dark)

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
            _scroll_target_into_view(target)
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
        content_key = (
            self._panel.step_lbl.text(),
            self._panel.title_lbl.text(),
            self._panel.body_lbl.text(),
            self._panel.hint_lbl.text(),
            str(self._panel.hint_lbl.isHidden()),
        )
        if content_key != self._last_panel_content_key:
            self._last_panel_content_key = content_key
            self._panel.recalculate_content_size()
        self._position_panel(
            _global_widget_rect(target) if target else QRect(),
            target,
        )
        self._panel.raise_()

    def _position_panel(
        self,
        target_global: QRect,
        target: QWidget | None = None,
    ) -> None:
        margin = 16
        panel = self._panel
        x, y = _resolve_panel_position(
            self._host,
            target_global,
            target,
            panel_width=panel.width(),
            panel_height=panel.height(),
            margin=margin,
        )
        panel.move(x, y)
