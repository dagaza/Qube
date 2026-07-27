"""Bottom-right in-app notification toasts (updates, commands, release notes)."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QEvent
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


from core.app_notification_types import AppNotificationRequest
from core.theme.view_theme import view_resolved_theme
from core.theme.color_utils import with_alpha
from core.theme.svg_icons import themed_fa_icon, themed_fa_pixmap
from ui.components.brand_buttons import apply_brand_primary


def notification_toast_stylesheet(
    theme,
    *,
    has_countdown: bool = False,
) -> str:
    """QSS for :class:`AppNotificationToast` (testable without constructing widgets)."""
    accent = theme.link
    bg = theme.background
    fg = theme.text_primary
    sub = theme.text_secondary
    border = with_alpha(theme.link, 0.45) if theme.is_dark else theme.border
    countdown_chunk = ""
    if has_countdown:
        track = with_alpha(theme.text_muted, 0.25 if theme.is_dark else 0.35)
        fill = theme.success
        countdown_chunk = f"""
            QProgressBar#AppNotificationCountdown {{
                background-color: {track};
                border: none;
                border-radius: 2px;
            }}
            QProgressBar#AppNotificationCountdown::chunk {{
                background-color: {fill};
                border-radius: 2px;
            }}
            """
    return f"""
            QFrame#AppNotificationToast {{
                background-color: {bg};
                border: 1px solid {border};
                border-radius: 12px;
            }}
            QLabel#AppNotificationTitle {{
                color: {fg};
                font-size: 13px;
                font-weight: 700;
                background: transparent;
            }}
            QLabel#AppNotificationBody {{
                color: {sub};
                font-size: 12px;
                background: transparent;
            }}
            QPushButton#AppNotificationClose {{
                background: transparent;
                border: none;
                color: {sub};
            }}
            QPushButton#AppNotificationClose:hover {{
                color: {fg};
            }}
            {countdown_chunk}
            """


class AppNotificationToast(QFrame):
    """Single dismissible toast card."""

    dismissed = pyqtSignal(object)
    action_triggered = pyqtSignal(str, object)

    def __init__(self, request: AppNotificationRequest, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._request = request
        self.setObjectName("AppNotificationToast")
        self.setMinimumWidth(300)
        self.setMaximumWidth(420)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(8)
        self._icon = QLabel()
        self._icon.setFixedSize(16, 16)
        self._title = QLabel(request.title)
        self._title.setObjectName("AppNotificationTitle")
        self._title.setWordWrap(True)
        header.addWidget(self._icon, alignment=Qt.AlignmentFlag.AlignTop)
        header.addWidget(self._title, stretch=1)
        self._close_btn = QPushButton()
        self._close_btn.setObjectName("AppNotificationClose")
        self._close_btn.setFixedSize(22, 22)
        self._close_btn.setFlat(True)
        self._close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._close_btn.clicked.connect(self._emit_dismissed)
        header.addWidget(self._close_btn, alignment=Qt.AlignmentFlag.AlignTop)
        outer.addLayout(header)

        self._body = QLabel(request.body)
        self._body.setObjectName("AppNotificationBody")
        self._body.setWordWrap(True)
        outer.addWidget(self._body)

        self._countdown_bar: QProgressBar | None = None
        self._countdown_timer: QTimer | None = None
        self._countdown_started_ms = 0
        self._countdown_total_ms = 0
        if request.show_countdown and request.auto_dismiss_ms > 0:
            self._countdown_bar = QProgressBar()
            self._countdown_bar.setObjectName("AppNotificationCountdown")
            self._countdown_bar.setTextVisible(False)
            self._countdown_bar.setFixedHeight(3)
            self._countdown_bar.setRange(0, 1000)
            self._countdown_bar.setValue(1000)
            outer.addWidget(self._countdown_bar)

        self._action_btn: QPushButton | None = None
        if request.action_label and request.action_id:
            action_row = QHBoxLayout()
            action_row.addStretch(1)
            self._action_btn = QPushButton(request.action_label)
            self._action_btn.setObjectName("AppNotificationAction")
            apply_brand_primary(self._action_btn)
            self._action_btn.setAutoDefault(True)
            self._action_btn.setDefault(True)
            self._action_btn.clicked.connect(self._on_action)
            action_row.addWidget(self._action_btn)
            outer.addLayout(action_row)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._close_btn.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        if self._action_btn is not None:
            self._action_btn.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self._action_btn.installEventFilter(self)

        self._auto_timer = QTimer(self)
        self._auto_timer.setSingleShot(True)
        self._auto_timer.timeout.connect(self._emit_dismissed)
        if request.auto_dismiss_ms > 0:
            self._auto_timer.start(request.auto_dismiss_ms)
            if self._countdown_bar is not None:
                self._countdown_total_ms = request.auto_dismiss_ms
                self._countdown_started_ms = 0
                self._countdown_timer = QTimer(self)
                self._countdown_timer.setInterval(50)
                self._countdown_timer.timeout.connect(self._tick_countdown)
                self._countdown_timer.start()

        self.apply_theme(view_resolved_theme(self).is_dark)

    def restart_auto_dismiss(self, request: AppNotificationRequest) -> None:
        """Reset auto-dismiss and countdown when deduping the same toast."""
        self._request = request
        self._title.setText(request.title)
        self._body.setText(request.body)
        if request.auto_dismiss_ms > 0:
            self._auto_timer.start(request.auto_dismiss_ms)
            if self._countdown_bar is not None and request.show_countdown:
                self._countdown_total_ms = request.auto_dismiss_ms
                self._countdown_started_ms = 0
                self._countdown_bar.setValue(1000)
                if self._countdown_timer is None:
                    self._countdown_timer = QTimer(self)
                    self._countdown_timer.setInterval(50)
                    self._countdown_timer.timeout.connect(self._tick_countdown)
                if not self._countdown_timer.isActive():
                    self._countdown_timer.start()
        self.apply_theme(getattr(self, "_is_dark", True))

    def _tick_countdown(self) -> None:
        if self._countdown_bar is None or self._countdown_total_ms <= 0:
            return
        self._countdown_started_ms += 50
        remaining = max(0, self._countdown_total_ms - self._countdown_started_ms)
        value = int(1000 * remaining / self._countdown_total_ms)
        self._countdown_bar.setValue(value)
        if remaining <= 0 and self._countdown_timer is not None:
            self._countdown_timer.stop()

    def _emit_dismissed(self) -> None:
        self._auto_timer.stop()
        if self._countdown_timer is not None:
            self._countdown_timer.stop()
        self.dismissed.emit(self)

    def _on_action(self) -> None:
        if self._request.action_id:
            self.action_triggered.emit(self._request.action_id, self)
        self._emit_dismissed()

    def focus_primary_action(self) -> None:
        if self._action_btn is not None:
            self._action_btn.setFocus(Qt.FocusReason.OtherFocusReason)
        else:
            self.setFocus(Qt.FocusReason.OtherFocusReason)

    def activate_primary_action(self) -> None:
        if self._action_btn is not None:
            self._action_btn.click()
        else:
            self._emit_dismissed()

    def handle_key(self, event: QKeyEvent) -> bool:
        key = event.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.activate_primary_action()
            event.accept()
            return True
        if key == Qt.Key.Key_Escape:
            self._emit_dismissed()
            event.accept()
            return True
        return False

    def keyPressEvent(self, event) -> None:
        if self.handle_key(event):
            return
        super().keyPressEvent(event)

    def eventFilter(self, watched, event) -> bool:
        if (
            self._action_btn is not None
            and watched is self._action_btn
            and event.type() == QEvent.Type.KeyPress
            and self.handle_key(event)
        ):
            return True
        return super().eventFilter(watched, event)

    def apply_theme(self, is_dark: bool) -> None:
        self._is_dark = is_dark
        theme = view_resolved_theme(self, is_dark=is_dark)
        accent = theme.link
        sub = theme.text_muted if theme.is_dark else theme.text_secondary
        icon_name = self._request.icon_name or "fa5s.bell"
        self.setStyleSheet(
            notification_toast_stylesheet(
                theme,
                has_countdown=self._countdown_bar is not None,
            )
        )
        self._icon.setPixmap(themed_fa_pixmap(icon_name, accent, 16))
        self._close_btn.setIcon(themed_fa_icon("fa5s.times", sub, 12))
        self._close_btn.setIconSize(self._close_btn.size())


class AppNotificationCenter(QWidget):
    """Stacks toast notifications in the bottom-right of the host window."""

    action_triggered = pyqtSignal(str)

    _MARGIN = 18
    _SPACING = 10
    _MAX_VISIBLE = 4

    def __init__(self, host: QWidget) -> None:
        super().__init__(host)
        self._host = host
        self._is_dark = True
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("background: transparent;")

        self._stack = QVBoxLayout(self)
        self._stack.setContentsMargins(0, 0, 0, 0)
        self._stack.setSpacing(self._SPACING)
        self._stack.addStretch(1)

        self._toasts: list[AppNotificationToast] = []
        self._toast_by_dedupe: dict[str, AppNotificationToast] = {}
        self._app_filter_installed = False
        self.hide()
        self._host.installEventFilter(self)

    def has_visible_toasts(self) -> bool:
        return bool(self._toasts) and self.isVisible()

    def focus_top_toast(self) -> None:
        if not self._toasts:
            return
        self._toasts[-1].focus_primary_action()

    def handle_key(self, event: QKeyEvent) -> bool:
        if not self.has_visible_toasts():
            return False
        key = event.key()
        if key not in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Escape):
            return False
        return self._toasts[-1].handle_key(event)

    def _sync_app_event_filter(self) -> None:
        app = QApplication.instance()
        if app is None:
            return
        if self.has_visible_toasts():
            if not self._app_filter_installed:
                app.installEventFilter(self)
                self._app_filter_installed = True
        elif self._app_filter_installed:
            app.removeEventFilter(self)
            self._app_filter_installed = False

    def _toast_contains(self, widget: QWidget | None) -> bool:
        while widget is not None:
            if widget in self._toasts or widget is self:
                return True
            widget = widget.parentWidget()
        return False

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Type.KeyPress and self.has_visible_toasts():
            modal = QApplication.activeModalWidget()
            if modal is not None and not self._toast_contains(modal):
                return super().eventFilter(watched, event)
            if isinstance(event, QKeyEvent) and self.handle_key(event):
                return True
        return super().eventFilter(watched, event)

    def apply_theme(self, is_dark: bool) -> None:
        self._is_dark = is_dark
        for toast in self._toasts:
            toast.apply_theme(is_dark)

    def show_notification(self, request: AppNotificationRequest) -> None:
        if request.dedupe_key:
            existing = self._toast_by_dedupe.get(request.dedupe_key)
            if existing is not None and existing in self._toasts:
                existing.restart_auto_dismiss(request)
                self.relayout()
                self.show()
                self.raise_()
                return

        while len(self._toasts) >= self._MAX_VISIBLE:
            old = self._toasts.pop(0)
            self._unregister_dedupe(old)
            old.setParent(None)
            old.deleteLater()

        toast = AppNotificationToast(request, self)
        toast.apply_theme(self._is_dark)
        toast.dismissed.connect(self._remove_toast)
        toast.action_triggered.connect(self._forward_action)
        self._toasts.append(toast)
        if request.dedupe_key:
            self._toast_by_dedupe[request.dedupe_key] = toast
        self._stack.addWidget(toast)
        self.relayout()
        self.show()
        self.raise_()
        self._sync_app_event_filter()
        host_window = self._host.window()
        if host_window is not None:
            host_window.activateWindow()
        # Modal dialogs (e.g. PrestigeDialog) restore focus after exec(); defer focus.
        QTimer.singleShot(50, self.focus_top_toast)

    def _forward_action(self, action_id: str, _toast: AppNotificationToast) -> None:
        self.action_triggered.emit(action_id)

    def _unregister_dedupe(self, toast: AppNotificationToast) -> None:
        dedupe_key = getattr(toast._request, "dedupe_key", None)
        if dedupe_key and self._toast_by_dedupe.get(dedupe_key) is toast:
            self._toast_by_dedupe.pop(dedupe_key, None)

    def _remove_toast(self, toast: AppNotificationToast) -> None:
        if toast in self._toasts:
            self._toasts.remove(toast)
        self._unregister_dedupe(toast)
        toast.setParent(None)
        toast.deleteLater()
        if not self._toasts:
            self.hide()
            self._sync_app_event_filter()
        else:
            self.relayout()

    def relayout(self) -> None:
        if not self._host.isVisible():
            return
        host_w = self._host.width()
        host_h = self._host.height()
        width = min(420, max(300, host_w - 2 * self._MARGIN))
        self.setFixedWidth(width)
        self.adjustSize()
        height = self.sizeHint().height()
        x = host_w - width - self._MARGIN
        y = host_h - height - self._MARGIN
        self.setGeometry(max(0, x), max(0, y), width, height)
        self.raise_()
