"""Scrollable Prestige dialog for the composer @-mention user guide."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QShowEvent
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
)

from core.composer_mention_guide import build_composer_mention_guide_text
from core.platform.frameless_window import (
    apply_frameless_dialog_chrome,
    configure_frameless_dialog_host,
)
from core.theme.view_theme import view_resolved_theme
from core.theme.widget_styles import (
    PRESTIGE_ACCENT_LABEL,
    PRESTIGE_BODY_LABEL,
    PRESTIGE_GHOST_BUTTON,
    PRESTIGE_SOURCE_CONTAINER,
    PRESTIGE_TEXT_VIEW,
)
from ui.components.prestige_dialog import _resolve_is_dark_from_parent


class ComposerMentionGuideDialog(QDialog):
    """Read-only guide viewer with Prestige frameless chrome."""

    def __init__(self, parent=None, *, is_dark: bool | None = None) -> None:
        super().__init__(parent)
        if is_dark is None:
            is_dark = _resolve_is_dark_from_parent(parent)
        theme = view_resolved_theme(parent, is_dark=is_dark)

        configure_frameless_dialog_host(self)
        self.setMinimumSize(640, 520)
        self.resize(720, 620)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)

        container = QFrame()
        container.setObjectName("ComposerMentionGuideContainer")
        container.setStyleSheet(
            theme.style(
                PRESTIGE_SOURCE_CONTAINER,
                accent=theme.link,
                object_name="ComposerMentionGuideContainer",
            )
        )

        inner = QVBoxLayout(container)
        inner.setContentsMargins(28, 26, 28, 22)
        inner.setSpacing(14)

        header = QLabel("COMPOSER GUIDE")
        header.setStyleSheet(
            theme.style(PRESTIGE_ACCENT_LABEL, accent=theme.link, font_size="11px")
        )
        title = QLabel("@ mentions in chat")
        title.setWordWrap(True)
        title.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        title.setStyleSheet(theme.style(PRESTIGE_BODY_LABEL, font_size="16px"))

        intro = QLabel(
            "Attach files, tools, skills, and more from the composer palette. "
            "Scroll for mixing rules and limits."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(theme.style(PRESTIGE_BODY_LABEL, font_size="13px", font_weight="400"))

        inner.addWidget(header)
        inner.addWidget(title)
        inner.addWidget(intro)

        self.viewer = QTextEdit()
        self.viewer.setReadOnly(True)
        self.viewer.setPlainText(build_composer_mention_guide_text())
        self.viewer.setMinimumHeight(320)
        self.viewer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.viewer.setStyleSheet(theme.style(PRESTIGE_TEXT_VIEW))
        inner.addWidget(self.viewer, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("CLOSE")
        close_btn.setStyleSheet(theme.style(PRESTIGE_GHOST_BUTTON))
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        inner.addLayout(btn_row)

        outer.addWidget(container)

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        apply_frameless_dialog_chrome(self)


def show_composer_mention_guide(parent=None, *, is_dark: bool | None = None) -> None:
    """Modal guide dialog; safe to call from Settings, onboarding, or main window."""
    ComposerMentionGuideDialog(parent, is_dark=is_dark).exec()
