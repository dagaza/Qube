"""Minimal startup splash shown before the heavy application module loads."""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import QEasingCurve, QObject, QPropertyAnimation, Qt, QTimer
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from core.paths import install_root
from core.platform.window_activation import activate_toplevel_window
from ui.branded_theme import splash_compact_card_qss
from ui.splash_widget import SplashCircleSpinner, _SplashCardChrome, resolve_splash_logo_path

logger = logging.getLogger("Qube.UI.EarlySplash")

_FADE_IN_MS = 220
_SPINNER_INTERVAL_MS = 16


class _EarlySplashShell(QWidget):
    def __init__(self, controller: "EarlySplashController") -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint,
        )
        self._controller = controller
        self.setWindowTitle("Qube")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background: transparent;")

    def closeEvent(self, event) -> None:  # noqa: N802
        event.accept()
        app = QApplication.instance()
        if app is not None and not self._controller.handoff_complete():
            logger.info("Early splash closed before startup completed; exiting.")
            app.quit()


class EarlySplashController(QObject):
    """Lightweight splash shown while ``main`` imports and initializes."""

    def __init__(self, *, repo_root: Path | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._repo_root = repo_root or install_root()
        self._shell = _EarlySplashShell(self)
        self._shell.setObjectName("QubeEarlySplashShell")
        self._shell.setWindowOpacity(0.0)

        outer = QVBoxLayout(self._shell)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setAlignment(Qt.AlignmentFlag.AlignCenter)

        card = _SplashCardChrome(self._shell)
        card.setObjectName("QubeEarlySplashCard")
        card.setFixedWidth(360)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(28, 24, 28, 22)
        card_layout.setSpacing(12)

        logo = QLabel()
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_path = resolve_splash_logo_path(self._repo_root)
        if logo_path is not None:
            pix = QPixmap(str(logo_path))
            if not pix.isNull():
                logo.setPixmap(pix.scaledToWidth(72, Qt.TransformationMode.SmoothTransformation))
        card_layout.addWidget(logo)

        title = QLabel("Qube")
        title.setObjectName("QubeEarlySplashTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont(title.font())
        title_font.setPointSize(22)
        title_font.setWeight(QFont.Weight.ExtraBold)
        title.setFont(title_font)
        card_layout.addWidget(title)

        self._status = QLabel("Starting Qube…")
        self._status.setObjectName("QubeEarlySplashStatus")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setWordWrap(True)
        card_layout.addWidget(self._status)

        self._spinner = SplashCircleSpinner(size=40, parent=card)
        card_layout.addWidget(self._spinner, 0, Qt.AlignmentFlag.AlignHCenter)

        card.setStyleSheet(splash_compact_card_qss().strip())
        outer.addWidget(card)

        self._fade_in_anim: QPropertyAnimation | None = None
        self._dismissed = False
        self._handoff_complete = False

        self._spinner_timer = QTimer(self)
        self._spinner_timer.setInterval(_SPINNER_INTERVAL_MS)
        self._spinner_timer.timeout.connect(self._advance_spinner)

    def handoff_complete(self) -> bool:
        return self._handoff_complete

    def present(self) -> None:
        self._recenter_on_primary_screen()
        self._shell.show()
        self._shell.raise_()
        self._spinner_timer.start()
        QTimer.singleShot(0, self._start_fade_in)
        logger.info("Early splash presented.")

    def dismiss(self) -> None:
        if self._dismissed:
            return
        self._dismissed = True
        self._handoff_complete = True
        self._spinner_timer.stop()
        self._shell.hide()
        self._shell.deleteLater()
        logger.info("Early splash dismissed.")

    def request_activation(self) -> None:
        activate_toplevel_window(self._shell)

    def _advance_spinner(self) -> None:
        self._spinner.advance(_SPINNER_INTERVAL_MS)

    def _start_fade_in(self) -> None:
        self._fade_in_anim = QPropertyAnimation(self._shell, b"windowOpacity")
        self._fade_in_anim.setDuration(_FADE_IN_MS)
        self._fade_in_anim.setStartValue(0.0)
        self._fade_in_anim.setEndValue(1.0)
        self._fade_in_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._fade_in_anim.start()

    def _recenter_on_primary_screen(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        self._shell.adjustSize()
        frame = self._shell.frameGeometry()
        frame.moveCenter(available.center())
        left = max(
            available.left(),
            min(frame.left(), available.right() - frame.width() + 1),
        )
        top = max(
            available.top(),
            min(frame.top(), available.bottom() - frame.height() + 1),
        )
        self._shell.move(left, top)
