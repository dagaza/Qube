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
from core.theme.accessors import theme_for
from core.theme.color_utils import with_alpha
from core.theme.widget_styles import (
    PRESTIGE_ACCENT_LABEL,
    PRESTIGE_GHOST_BUTTON,
    PRESTIGE_SOURCE_CONTAINER,
)
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
        theme = theme_for(is_dark=is_dark)
        hover_bg = with_alpha(theme.text_primary, 0.05)

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

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)

        container = QFrame()
        container.setObjectName("ProviderCredentialDialogContainer")
        container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        container.setStyleSheet(
            theme.style(
                PRESTIGE_SOURCE_CONTAINER,
                accent=theme.link,
                object_name="ProviderCredentialDialogContainer",
            )
            + f"""
            QLabel {{
                color: {theme.text_primary};
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
            theme.style(PRESTIGE_ACCENT_LABEL, accent=theme.link, font_size="11px")
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
            theme.style(PRESTIGE_GHOST_BUTTON)
            + f"""
            QPushButton:hover {{
                background: {hover_bg};
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
