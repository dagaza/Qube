"""Frameless Prestige-styled dialogs shared across Qube (dark/light aware)."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTextEdit,
)
from PyQt6.QtCore import Qt


def _resolve_is_dark_from_parent(parent) -> bool:
    w = parent.window() if parent else None
    return getattr(w, "_is_dark_theme", True) if w else True


class PrestigeDialog(QDialog):
    def __init__(self, parent, title, message, is_dark=True, is_input=False, default_text=""):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.setMinimumWidth(450)

        self.result_text = None
        bg, fg = ("#1e1e2e", "#cdd6f4") if is_dark else ("#ffffff", "#1e293b")
        accent = "#f38ba8" if "Delete" in title else "#89b4fa"
        border = "rgba(255, 255, 255, 0.1)" if is_dark else "#cbd5e1"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetFixedSize)

        self.container = QFrame()
        self.container.setObjectName("DialogContainer")
        self.container.setStyleSheet(
            f"""
            QFrame#DialogContainer {{
                background: {bg};
                border: 2px solid {accent};
                border-radius: 20px;
            }}
            QLabel {{ color: {fg}; border: none; background: transparent; }}
        """
        )

        c_layout = QVBoxLayout(self.container)
        c_layout.setContentsMargins(30, 30, 30, 25)
        c_layout.setSpacing(20)

        t_lbl = QLabel(title.upper())
        t_lbl.setStyleSheet(f"color: {accent}; font-weight: bold; font-size: 12px; letter-spacing: 2px;")

        m_lbl = QLabel(message)
        m_lbl.setWordWrap(True)
        m_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        m_lbl.setMinimumWidth(0)
        m_lbl.setStyleSheet(f"color: {fg}; font-size: 15px; line-height: 1.4;")

        c_layout.addWidget(t_lbl)
        c_layout.addWidget(m_lbl)

        self.field = None
        if is_input:
            self.field = QLineEdit(default_text)
            self.field.setMinimumHeight(45)
            self.field.setStyleSheet(
                f"""
                QLineEdit {{
                    background: {'#313244' if is_dark else '#f8fafc'};
                    color: {fg};
                    border-radius: 10px;
                    padding: 10px 15px;
                    border: 1px solid {accent};
                    font-size: 14px;
                }}
            """
            )
            c_layout.addWidget(self.field)
            self.field.setFocus()

        btns = QHBoxLayout()
        btns.setSpacing(15)

        cancel_btn = QPushButton("CANCEL")
        con_b = QPushButton("CONFIRM")

        btn_style = """
            QPushButton {
                padding: 15px 15px;
                min-height: 30px;
                border-radius: 12px;
                font-weight: bold;
                font-size: 12px;
                letter-spacing: 1px;
            }
        """

        cancel_btn.setStyleSheet(
            btn_style
            + f"""
            QPushButton {{
                color: {fg};
                border: 1px solid {border};
                background: transparent;
            }}
            QPushButton:hover {{
                background: rgba(255, 255, 255, 0.05);
            }}
        """
        )

        con_b.setStyleSheet(
            btn_style
            + f"""
            QPushButton {{
                background: {accent};
                color: #11111b;
                border: none;
            }}
            QPushButton:hover {{
                background: {accent};
                opacity: 0.9;
            }}
        """
        )

        cancel_btn.clicked.connect(self.reject)
        con_b.clicked.connect(self.accept)

        btns.addStretch()
        btns.addWidget(cancel_btn)
        btns.addWidget(con_b)
        c_layout.addLayout(btns)

        layout.addWidget(self.container)

    def exec(self):
        """Returns the input text if Accepted and is_input=True, otherwise True/None."""
        if super().exec():
            if self.field:
                self.result_text = self.field.text().strip()
                return self.result_text
            return True
        return None

    def accept_action(self):
        if getattr(self, "field", None):
            self.result_text = self.field.text()
        self.accept()


class SourcePreviewer(QDialog):
    """
    Read-only document viewer for citation sources: Prestige frameless chrome, theme-aware.
    """

    def __init__(self, filename: str, content: str, parent=None, *, is_dark: bool | None = None):
        super().__init__(parent)
        if is_dark is None:
            is_dark = _resolve_is_dark_from_parent(parent)

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowTitle(f"Source — {filename}")
        self.setMinimumSize(600, 500)
        self.resize(720, 560)

        bg, fg = ("#1e1e2e", "#cdd6f4") if is_dark else ("#ffffff", "#1e293b")
        accent = "#89b4fa"
        border = "rgba(255, 255, 255, 0.1)" if is_dark else "#cbd5e1"
        surface = "#313244" if is_dark else "#f8fafc"

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)

        container = QFrame()
        container.setObjectName("SourcePreviewContainer")
        container.setStyleSheet(
            f"""
            QFrame#SourcePreviewContainer {{
                background: {bg};
                border: 2px solid {accent};
                border-radius: 20px;
            }}
        """
        )

        inner = QVBoxLayout(container)
        inner.setContentsMargins(28, 26, 28, 22)
        inner.setSpacing(14)

        header = QLabel("SOURCE PREVIEW")
        header.setStyleSheet(
            f"color: {accent}; font-weight: bold; font-size: 11px; letter-spacing: 2px;"
        )
        title = QLabel(filename)
        title.setWordWrap(True)
        title.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        title.setStyleSheet(f"color: {fg}; font-size: 16px; font-weight: bold;")

        inner.addWidget(header)
        inner.addWidget(title)

        self.viewer = QTextEdit()
        self.viewer.setReadOnly(True)
        self.viewer.setPlainText(content)
        self.viewer.setMinimumHeight(280)
        self.viewer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.viewer.setStyleSheet(
            f"""
            QTextEdit {{
                background: {surface};
                color: {fg};
                border: 1px solid {border};
                border-radius: 12px;
                padding: 14px 16px;
                font-size: 14px;
                line-height: 1.55;
            }}
        """
        )
        inner.addWidget(self.viewer, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("CLOSE")
        btn_style = f"""
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
        close_btn.setStyleSheet(btn_style)
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        inner.addLayout(btn_row)

        outer.addWidget(container)
