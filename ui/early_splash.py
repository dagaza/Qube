"""Minimal startup splash shown before the heavy application module loads.

Kept intentionally static: ``import main`` blocks the GUI thread, so timer-driven
spinners and opacity animations cannot run reliably. A fully opaque card with the
Qube logo and a Loading label is the one-shot certain presentation.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import QObject, Qt
from PyQt6.QtGui import QFont, QPixmap, QShowEvent
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from core.paths import install_root
from core.platform.window_activation import (
    activate_toplevel_window,
    center_widget_on_screen,
    splash_window_flags,
)
from core.startup_exit import arm_force_process_exit, mark_startup_exit_requested
from ui.branded_theme import apply_splash_label_styles, early_splash_card_qss
from ui.splash_widget import _SplashCardChrome, resolve_splash_logo_path

logger = logging.getLogger("Qube.UI.EarlySplash")

_LOGO_WIDTH_PX = 96


class _EarlySplashShell(QWidget):
    def __init__(self, controller: "EarlySplashController") -> None:
        super().__init__(None, splash_window_flags())
        self._controller = controller
        self.setWindowTitle("Qube")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background: transparent;")

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        activate_toplevel_window(self)

    def closeEvent(self, event) -> None:  # noqa: N802
        event.accept()
        app = QApplication.instance()
        if app is not None and not self._controller.handoff_complete():
            logger.info("Early splash closed before startup completed; exiting.")
            mark_startup_exit_requested()
            app.quit()
            arm_force_process_exit()


class EarlySplashController(QObject):
    """Lightweight splash shown while ``main`` imports and initializes."""

    def __init__(self, *, repo_root: Path | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._repo_root = repo_root or install_root()
        self._shell = _EarlySplashShell(self)
        self._shell.setObjectName("QubeEarlySplashShell")
        # Fully opaque from the first paint — fade/timers cannot run during blocked import.
        self._shell.setWindowOpacity(1.0)

        outer = QVBoxLayout(self._shell)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setAlignment(Qt.AlignmentFlag.AlignCenter)

        card = _SplashCardChrome(self._shell)
        card.setObjectName("QubeEarlySplashCard")
        card.setFixedWidth(360)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(32, 28, 32, 28)
        card_layout.setSpacing(14)

        self._logo = QLabel()
        self._logo.setObjectName("QubeEarlySplashLogo")
        self._logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_path = resolve_splash_logo_path(self._repo_root)
        if logo_path is not None:
            pix = QPixmap(str(logo_path))
            if not pix.isNull():
                self._logo.setPixmap(
                    pix.scaledToWidth(
                        _LOGO_WIDTH_PX,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
        card_layout.addWidget(self._logo)

        title = QLabel("Qube")
        title.setObjectName("QubeEarlySplashTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont(title.font())
        title_font.setPointSize(22)
        title_font.setWeight(QFont.Weight.ExtraBold)
        title.setFont(title_font)
        card_layout.addWidget(title)

        self._status = QLabel("Loading…")
        self._status.setObjectName("QubeEarlySplashStatus")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setWordWrap(True)
        card_layout.addWidget(self._status)

        # Widget-level QSS plus per-label styles (Windows/macOS ignore rgba() in QSS).
        card.setStyleSheet(early_splash_card_qss().strip())
        apply_splash_label_styles(title=title, status=self._status)
        outer.addWidget(card)

        self._dismissed = False
        self._handoff_complete = False

    def handoff_complete(self) -> bool:
        return self._handoff_complete

    def present(self) -> None:
        self._recenter_on_screen()
        self._shell.show()
        activate_toplevel_window(self._shell)
        app = QApplication.instance()
        if app is not None:
            # Flush the first paint before the GUI thread blocks on ``import main``.
            app.processEvents()
        logger.info("Early splash presented.")

    def dismiss(self) -> None:
        if self._dismissed:
            return
        self._dismissed = True
        self._handoff_complete = True
        self._shell.hide()
        self._shell.deleteLater()
        logger.info("Early splash dismissed.")

    def request_activation(self) -> bool:
        if self._handoff_complete or self._dismissed:
            return False
        try:
            activate_toplevel_window(self._shell)
            return bool(self._shell.isVisible())
        except RuntimeError:
            return False

    def _recenter_on_screen(self) -> None:
        center_widget_on_screen(self._shell)
