"""Setup-tour-style coach panel for one-shot hidden feature discoveries."""

from __future__ import annotations

from typing import Callable

from PyQt6.QtCore import QTimer, Qt, QRect, QObject, QEvent
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QApplication, QWidget

from ui.components.celebration_burst import BorderFireworksHandle, show_border_fireworks
from ui.components.onboarding_tour import OnboardingCoachPanel, _global_widget_rect


def mention_popup_surface(popup: QWidget) -> QWidget:
    """Visual shell of the @-mention menu (rounded card), when available."""
    return getattr(popup, "_shell", popup)


class _HiddenFeatureCoachSession(QObject):
    def __init__(
        self,
        host: QWidget,
        target: QWidget,
        *,
        step_label: str,
        title: str,
        body: str,
        hint: str = "",
        on_dismiss: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(host)
        self._host = host
        self._target = target
        self._on_dismiss = on_dismiss
        self._active = False

        self._panel = OnboardingCoachPanel(host)
        self._panel.hide()
        self._panel.skip_btn.hide()
        self._panel.back_btn.hide()
        self._panel.next_btn.setText("Got it")
        self._panel.next_btn.setProperty("class", "PrimaryActionButton")
        self._panel.step_lbl.setText(step_label)
        self._panel.title_lbl.setText(title)
        self._panel.body_lbl.setText(body)
        if hint.strip():
            self._panel.hint_lbl.setText(hint)
            self._panel.hint_lbl.show()
        else:
            self._panel.hint_lbl.hide()
        self._panel.recalculate_content_size()

        self._panel.escape_pressed.connect(self.dismiss)
        self._panel.next_btn.clicked.connect(self.dismiss)

        self._refresh_timer = QTimer(host)
        self._refresh_timer.setInterval(250)
        self._refresh_timer.timeout.connect(self._refresh_layout)

    def present(self) -> None:
        if self._active:
            return
        self._active = True
        is_dark = getattr(self._host, "_is_dark_theme", True)
        self._panel.apply_theme(is_dark)
        self._panel.show()
        self._refresh_layout()
        self._panel.raise_()
        self._refresh_timer.start()
        self._host.installEventFilter(self)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        QTimer.singleShot(50, self._panel.setFocus)

    def dismiss(self) -> None:
        if not self._active:
            return
        self._active = False
        self._refresh_timer.stop()
        self._host.removeEventFilter(self)
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        self._panel.hide()
        if self._on_dismiss is not None:
            self._on_dismiss()

    def relayout(self) -> None:
        if self._active:
            self._refresh_layout()

    def eventFilter(self, watched, event) -> bool:
        if not self._active:
            return False
        if event.type() == QEvent.Type.KeyPress and isinstance(event, QKeyEvent):
            if event.key() == Qt.Key.Key_Escape:
                self.dismiss()
                event.accept()
                return True
        if watched is self._host and event.type() == QEvent.Type.Resize:
            QTimer.singleShot(0, self._refresh_layout)
        return False

    def _refresh_layout(self) -> None:
        if not self._active:
            return
        target_rect = _global_widget_rect(self._target, margin=8)
        self._position_panel(target_rect)
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


class _ComposerAtMentionDiscoveryPresentation(QObject):
    """Coordinates fireworks + coach while the @ menu is open."""

    def __init__(
        self,
        host: QWidget,
        mention_popup: QWidget,
        *,
        on_finished: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(host)
        self._host = host
        self._mention_popup = mention_popup
        self._surface = mention_popup_surface(mention_popup)
        self._on_finished = on_finished
        self._fireworks: BorderFireworksHandle | None = None
        self._coach: _HiddenFeatureCoachSession | None = None
        self._popup_watcher: _MentionPopupHideWatcher | None = None

    def start(self) -> None:
        if not self._mention_popup.isVisible():
            self._cleanup()
            return
        self._surface.updateGeometry()
        self._coach = _HiddenFeatureCoachSession(
            self._host,
            self._surface,
            step_label="SETUP TOUR",
            title="Hidden feature found",
            body=(
                "You unlocked the composer @ picker — release the modifier after @ "
                "to attach files, conversations, tools, skills, and commands. "
                "Type @@ for a literal @; add another @ for each extra escape.\n\n"
                "For the full guide (mixing limits, every skill, and token formats), "
                "open Settings → Help → Open @ Composer Guide."
            ),
            hint="Keep going — more secrets are tucked around Qube.",
            on_dismiss=self._on_coach_dismissed,
        )
        self._coach.present()
        self._fireworks = show_border_fireworks(
            self._surface,
            duration_ms=3200,
            on_finished=self._on_fireworks_finished,
        )
        self._popup_watcher = _MentionPopupHideWatcher(
            self._mention_popup,
            on_hidden=self._on_mention_menu_hidden,
        )

    def _on_mention_menu_hidden(self) -> None:
        self._stop_fireworks()
        if self._coach is not None:
            self._coach.relayout()

    def _on_fireworks_finished(self) -> None:
        self._fireworks = None

    def _stop_fireworks(self) -> None:
        if self._fireworks is not None:
            self._fireworks.stop()
            self._fireworks = None

    def _on_coach_dismissed(self) -> None:
        self._stop_fireworks()
        if self._popup_watcher is not None:
            self._popup_watcher.detach()
            self._popup_watcher = None
        self._coach = None
        setattr(self._host, "_composer_at_mention_discovery", None)
        if self._on_finished is not None:
            self._on_finished()

    def _cleanup(self) -> None:
        self._stop_fireworks()
        if self._popup_watcher is not None:
            self._popup_watcher.detach()
            self._popup_watcher = None
        if self._coach is not None:
            self._coach.dismiss()
            self._coach = None
        setattr(self._host, "_composer_at_mention_discovery", None)


class _MentionPopupHideWatcher(QObject):
    def __init__(self, popup: QWidget, *, on_hidden: Callable[[], None]) -> None:
        super().__init__(popup)
        self._popup = popup
        self._on_hidden = on_hidden
        popup.installEventFilter(self)

    def detach(self) -> None:
        self._popup.removeEventFilter(self)

    def eventFilter(self, watched, event) -> bool:
        if watched is self._popup and event.type() == QEvent.Type.Hide:
            self._on_hidden()
        return False


def present_composer_at_mention_discovery(
    host: QWidget,
    mention_popup: QWidget,
    *,
    on_finished: Callable[[], None] | None = None,
) -> None:
    """Fireworks around the @ menu while a setup-tour-style coach panel is shown."""
    presentation = _ComposerAtMentionDiscoveryPresentation(
        host,
        mention_popup,
        on_finished=on_finished,
    )
    setattr(host, "_composer_at_mention_discovery", presentation)
    QTimer.singleShot(80, presentation.start)
