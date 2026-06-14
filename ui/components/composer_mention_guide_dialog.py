"""Scrollable Prestige dialog for the composer @-mention user guide."""

from __future__ import annotations

from PyQt6.QtCore import Qt
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
from ui.components.prestige_dialog import _resolve_is_dark_from_parent


class ComposerMentionGuideDialog(QDialog):
    """Read-only guide viewer with Prestige frameless chrome."""

    def __init__(self, parent=None, *, is_dark: bool | None = None) -> None:
        super().__init__(parent)
        if is_dark is None:
            is_dark = _resolve_is_dark_from_parent(parent)

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumSize(640, 520)
        self.resize(720, 620)

        bg, fg = ("#1e1e2e", "#cdd6f4") if is_dark else ("#ffffff", "#1e293b")
        accent = "#89b4fa"
        border = "rgba(255, 255, 255, 0.1)" if is_dark else "#cbd5e1"
        surface = "#313244" if is_dark else "#f8fafc"

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)

        container = QFrame()
        container.setObjectName("ComposerMentionGuideContainer")
        container.setStyleSheet(
            f"""
            QFrame#ComposerMentionGuideContainer {{
                background: {bg};
                border: 2px solid {accent};
                border-radius: 20px;
            }}
        """
        )

        inner = QVBoxLayout(container)
        inner.setContentsMargins(28, 26, 28, 22)
        inner.setSpacing(14)

        header = QLabel("COMPOSER GUIDE")
        header.setStyleSheet(
            f"color: {accent}; font-weight: bold; font-size: 11px; letter-spacing: 2px;"
        )
        title = QLabel("@ mentions in chat")
        title.setWordWrap(True)
        title.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        title.setStyleSheet(f"color: {fg}; font-size: 16px; font-weight: bold;")

        intro = QLabel(
            "Attach files, tools, skills, and more from the composer palette. "
            "Scroll for mixing rules and limits."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {fg}; font-size: 13px;")

        inner.addWidget(header)
        inner.addWidget(title)
        inner.addWidget(intro)

        self.viewer = QTextEdit()
        self.viewer.setReadOnly(True)
        self.viewer.setPlainText(build_composer_mention_guide_text())
        self.viewer.setMinimumHeight(320)
        self.viewer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.viewer.setStyleSheet(
            f"""
            QTextEdit {{
                background: {surface};
                color: {fg};
                border: 1px solid {border};
                border-radius: 12px;
                padding: 14px 16px;
                font-size: 13px;
                line-height: 1.55;
                font-family: "Inter", sans-serif;
            }}
        """
        )
        inner.addWidget(self.viewer, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("CLOSE")
        close_btn.setStyleSheet(
            f"""
            QPushButton {{
                padding: 12px 22px;
                min-height: 32px;
                border-radius: 12px;
                font-weight: bold;
                font-size: 12px;
                letter-spacing: 1px;
                color: {fg};
                border: 1px solid {border};
                background: transparent;
            }}
            QPushButton:hover {{
                background: rgba(255, 255, 255, 0.05);
            }}
        """
        )
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        inner.addLayout(btn_row)

        outer.addWidget(container)


def show_composer_mention_guide(parent=None, *, is_dark: bool | None = None) -> None:
    """Modal guide dialog; safe to call from Settings, onboarding, or main window."""
    ComposerMentionGuideDialog(parent, is_dark=is_dark).exec()
