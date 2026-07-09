"""Prestige-styled provider credential configure dialog (Settings → Knowledge)."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QShowEvent
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.knowledge.provider_credentials import get_provider_credential_spec
from ui.components.prestige_dialog import _center_dialog_on_host, _resolve_is_dark_from_parent
from ui.views.settings.sections.knowledge_provider_credentials import (
    build_provider_credential_card,
    sync_provider_credential_rows,
)
from ui.views.settings.sections.knowledge_provider_status import sync_provider_status_panel

_DIALOG_WIDTH = 520
_DIALOG_MIN_HEIGHT = 340
_CONTENT_MARGIN = 28


class ProviderCredentialDialog(QDialog):
    """Modal for one knowledge provider's API key configuration."""

    def __init__(
        self,
        host,
        provider_id: str,
        *,
        is_dark: bool | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        if is_dark is None:
            is_dark = _resolve_is_dark_from_parent(parent)

        spec = get_provider_credential_spec(provider_id)
        if spec is None:
            raise ValueError(f"Unknown provider: {provider_id!r}")

        self._host = host
        self._provider_id = provider_id

        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedWidth(_DIALOG_WIDTH)
        self.setMinimumHeight(_DIALOG_MIN_HEIGHT)

        bg = "#1e1e2e" if is_dark else "#ffffff"
        fg = "#cdd6f4" if is_dark else "#1e293b"
        accent = "#89b4fa"
        border = "rgba(255, 255, 255, 0.1)" if is_dark else "#cbd5e1"

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)

        container = QFrame()
        container.setObjectName("ProviderCredentialDialogContainer")
        container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        container.setStyleSheet(
            f"""
            QFrame#ProviderCredentialDialogContainer {{
                background: {bg};
                border: 2px solid {accent};
                border-radius: 20px;
            }}
            QLabel {{
                color: {fg};
                background: transparent;
                border: none;
            }}
        """
        )

        inner = QVBoxLayout(container)
        inner.setContentsMargins(_CONTENT_MARGIN, 26, _CONTENT_MARGIN, 22)
        inner.setSpacing(14)

        header = QLabel(spec.label.upper())
        header.setStyleSheet(
            f"color: {accent}; font-weight: bold; font-size: 11px; letter-spacing: 2px;"
        )
        inner.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        scroll.setStyleSheet(
            """
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollArea > QWidget > QWidget {
                background: transparent;
            }
        """
        )
        scroll.viewport().setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        scroll.viewport().setStyleSheet("background: transparent;")

        body_host = QWidget()
        body_host.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        body_host.setStyleSheet("background: transparent;")
        body_layout = QVBoxLayout(body_host)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        if not hasattr(host, "knowledge_provider_key_fields"):
            host.knowledge_provider_key_fields = {}
        if not hasattr(host, "knowledge_provider_status_labels"):
            host.knowledge_provider_status_labels = {}

        card = build_provider_credential_card(
            host,
            spec,
            include_title=False,
            for_dialog=True,
        )
        card.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        body_layout.addWidget(card)
        scroll.setWidget(body_host)
        inner.addWidget(scroll, stretch=1)

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

        sync_provider_credential_rows(host)

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802 — Qt API
        super().showEvent(event)
        _center_dialog_on_host(self)

    def done(self, result: int) -> None:  # noqa: N802 — Qt API
        super().done(result)
        try:
            from ui.views.settings.sections.knowledge_sources import sync_live_source_rows

            sync_live_source_rows(self._host)
        except ImportError:
            pass
        sync_provider_status_panel(self._host)


def open_provider_credential_dialog(
    host,
    provider_id: str,
    *,
    is_dark: bool | None = None,
    parent=None,
) -> None:
    """Show configure modal for one provider; no-op when spec is unknown."""
    pid = (provider_id or "").strip().lower()
    if get_provider_credential_spec(pid) is None:
        return
    dlg = ProviderCredentialDialog(
        host,
        pid,
        is_dark=is_dark,
        parent=parent or (host.window() if hasattr(host, "window") else None),
    )
    dlg.exec()
