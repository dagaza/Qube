"""Prestige-styled dialog for Hugging Face errors with optional Retry / status link."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QSizePolicy

from core.hf_hub_errors import HF_STATUS_URL, HubErrorInfo
from core.theme.accessors import theme_for
from core.theme.color_utils import with_alpha
from core.theme.widget_styles import (
    PRESTIGE_ACCENT_LABEL,
    PRESTIGE_BODY_LABEL,
    PRESTIGE_DIALOG_CONFIRM,
    PRESTIGE_GHOST_BUTTON,
    PRESTIGE_SOURCE_CONTAINER,
)


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
        theme = theme_for(is_dark=is_dark)
        hover_bg = with_alpha(theme.text_primary, 0.05)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)

        container = QFrame()
        container.setObjectName("HubErrorDialogContainer")
        container.setStyleSheet(
            theme.style(
                PRESTIGE_SOURCE_CONTAINER,
                accent=theme.link,
                object_name="HubErrorDialogContainer",
            )
            + "QLabel { border: none; background: transparent; }"
        )
        container.setMinimumWidth(420)
        c_layout = QVBoxLayout(container)
        c_layout.setContentsMargins(30, 30, 30, 25)
        c_layout.setSpacing(20)

        title_lbl = QLabel(str(info.title or "Hugging Face error").upper())
        title_lbl.setStyleSheet(
            theme.style(PRESTIGE_ACCENT_LABEL, accent=theme.link, font_size="12px")
        )
        msg_lbl = QLabel(info.dialog_message())
        msg_lbl.setWordWrap(True)
        msg_lbl.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        msg_lbl.setStyleSheet(theme.style(PRESTIGE_BODY_LABEL, font_size="15px", font_weight="400"))

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
            + theme.style(PRESTIGE_GHOST_BUTTON)
            + f"QPushButton:hover {{ background: {hover_bg}; }}"
        )
        dismiss_btn.clicked.connect(self.reject)
        btns.addWidget(dismiss_btn)

        if status:
            status_btn = QPushButton("CHECK STATUS")
            status_btn.setStyleSheet(
                btn_style
                + theme.style(PRESTIGE_GHOST_BUTTON)
                + f"QPushButton:hover {{ background: {hover_bg}; }}"
            )

            def _open_status() -> None:
                QDesktopServices.openUrl(QUrl(HF_STATUS_URL))

            status_btn.clicked.connect(_open_status)
            btns.addWidget(status_btn)

        if retry:
            retry_btn = QPushButton("RETRY")
            retry_btn.setStyleSheet(
                btn_style
                + theme.style(
                    PRESTIGE_DIALOG_CONFIRM,
                    accent=theme.link,
                    confirm_fg=theme.text_on_accent,
                )
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
