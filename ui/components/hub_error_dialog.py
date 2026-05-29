"""Prestige-styled dialog for Hugging Face errors with optional Retry / status link."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QSizePolicy

from core.hf_hub_errors import HF_STATUS_URL, HubErrorInfo


class HubErrorDialog(QDialog):
    """Modal for Hub failures. Returns True when the user chooses Retry."""

    def __init__(
        self,
        parent,
        info: HubErrorInfo,
        *,
        is_dark: bool = True,
        show_retry: bool | None = None,
        show_status: bool | None = None,
    ):
        super().__init__(parent)
        self._info = info
        self._retry_chosen = False
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        retry = info.retryable if show_retry is None else bool(show_retry)
        status = info.show_status_link if show_status is None else bool(show_status)

        bg, fg = ("#1e1e2e", "#cdd6f4") if is_dark else ("#ffffff", "#1e293b")
        accent = "#89b4fa"
        border = "rgba(255, 255, 255, 0.1)" if is_dark else "#cbd5e1"

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)

        container = QFrame()
        container.setObjectName("HubErrorDialogContainer")
        container.setStyleSheet(
            f"""
            QFrame#HubErrorDialogContainer {{
                background: {bg};
                border: 2px solid {accent};
                border-radius: 20px;
            }}
            QLabel {{ color: {fg}; border: none; background: transparent; }}
            """
        )
        container.setMinimumWidth(420)
        c_layout = QVBoxLayout(container)
        c_layout.setContentsMargins(30, 30, 30, 25)
        c_layout.setSpacing(20)

        title_lbl = QLabel(str(info.title or "Hugging Face error").upper())
        title_lbl.setStyleSheet(
            f"color: {accent}; font-weight: bold; font-size: 12px; letter-spacing: 2px;"
        )
        msg_lbl = QLabel(info.dialog_message())
        msg_lbl.setWordWrap(True)
        msg_lbl.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        msg_lbl.setStyleSheet(f"color: {fg}; font-size: 15px; line-height: 1.4;")

        c_layout.addWidget(title_lbl)
        c_layout.addWidget(msg_lbl)

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
        btns = QHBoxLayout()
        btns.setSpacing(15)

        dismiss_btn = QPushButton("DISMISS")
        dismiss_btn.setStyleSheet(
            btn_style
            + f"""
            QPushButton {{
                color: {fg};
                border: 1px solid {border};
                background: transparent;
            }}
            QPushButton:hover {{ background: rgba(255, 255, 255, 0.05); }}
            """
        )
        dismiss_btn.clicked.connect(self.reject)
        btns.addWidget(dismiss_btn)

        if status:
            status_btn = QPushButton("CHECK STATUS")
            status_btn.setStyleSheet(
                btn_style
                + f"""
                QPushButton {{
                    color: {fg};
                    border: 1px solid {border};
                    background: transparent;
                }}
                QPushButton:hover {{ background: rgba(255, 255, 255, 0.05); }}
                """
            )

            def _open_status() -> None:
                QDesktopServices.openUrl(QUrl(HF_STATUS_URL))

            status_btn.clicked.connect(_open_status)
            btns.addWidget(status_btn)

        if retry:
            retry_btn = QPushButton("RETRY")
            retry_btn.setStyleSheet(
                btn_style
                + f"""
                QPushButton {{
                    background: {accent};
                    color: #11111b;
                    border: none;
                }}
                QPushButton:hover {{ opacity: 0.9; }}
                """
            )
            retry_btn.clicked.connect(self._choose_retry)
            btns.addWidget(retry_btn)

        btns.addStretch()
        c_layout.addLayout(btns)
        outer.addWidget(container)

    def _choose_retry(self) -> None:
        self._retry_chosen = True
        self.accept()

    def exec_retry(self) -> bool:
        """Run the dialog; return True if the user tapped Retry."""
        self.exec()
        return self._retry_chosen
