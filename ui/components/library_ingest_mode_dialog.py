"""Library import indexing mode chooser."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QLabel,
    QPushButton,
)
from PyQt6.QtGui import QShowEvent

from core.library_ingest_modes import (
    INGEST_MODE_PRECISION,
    INGEST_MODE_STANDARD,
    precision_ingest_license_message,
)
from core.library_pro_features import default_import_ingest_mode, user_has_pro_library_ingest
from core.platform.frameless_window import (
    apply_frameless_dialog_chrome,
    configure_frameless_dialog_host,
)
from core.theme.svg_icons import themed_fa_icon
from ui.components.prestige_dialog import (
    _center_dialog_on_host,
    _dialog_theme,
    _PRESTIGE_BTN_BASE,
)
from core.theme.widget_styles import (
    PRESTIGE_DIALOG_CONFIRM,
    PRESTIGE_DIALOG_CONTAINER,
    PRESTIGE_DIALOG_MESSAGE,
    PRESTIGE_DIALOG_MODE_OPTION,
    PRESTIGE_DIALOG_TITLE,
    PRESTIGE_GHOST_BUTTON,
    prestige_accent_colors,
)


def _apply_mode_button_tooltip(button: QPushButton, tooltip: str) -> None:
    """Hover tooltips do not fire on disabled QPushButtons — keep enabled + WA_Hover."""
    button.setToolTip(tooltip)
    button.setAttribute(Qt.WidgetAttribute.WA_Hover, True)


class LibraryIngestModeDialog(QDialog):
    """Choose standard or precision indexing before Library ingest starts."""

    def __init__(
        self,
        parent,
        *,
        file_count: int | None = None,
        is_dark: bool | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        configure_frameless_dialog_host(self)
        self.setMinimumWidth(480)

        self._selected_mode: str | None = None
        if is_dark is None:
            is_dark = getattr(parent.window() if parent else None, "_is_dark_theme", True)

        theme = _dialog_theme(parent, is_dark)
        accent, confirm_fg = prestige_accent_colors(theme, tone="default", title="Import")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        container = QFrame()
        container.setObjectName("DialogContainer")
        container.setStyleSheet(
            theme.style(PRESTIGE_DIALOG_CONTAINER, accent=accent, object_name="DialogContainer")
        )
        c_layout = QVBoxLayout(container)
        c_layout.setContentsMargins(30, 30, 30, 25)
        c_layout.setSpacing(18)

        title = QLabel("CHOOSE INDEXING MODE")
        title.setStyleSheet(theme.style(PRESTIGE_DIALOG_TITLE, accent=accent))
        message = QLabel(self._message_text(file_count))
        message.setWordWrap(True)
        message.setStyleSheet(theme.style(PRESTIGE_DIALOG_MESSAGE))

        c_layout.addWidget(title)
        c_layout.addWidget(message)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(12)

        self._normal_btn = QPushButton("NORMAL INDEXING")
        self._precision_btn = QPushButton("  PRECISION INDEXING")
        self._normal_btn.setMinimumHeight(44)
        self._precision_btn.setMinimumHeight(44)
        self._normal_btn.setStyleSheet(
            theme.style(
                PRESTIGE_DIALOG_MODE_OPTION,
                btn_base=_PRESTIGE_BTN_BASE,
            )
        )

        licensed = user_has_pro_library_ingest()
        if licensed:
            self._precision_btn.setStyleSheet(
                theme.style(
                    PRESTIGE_DIALOG_CONFIRM,
                    btn_base=_PRESTIGE_BTN_BASE,
                    accent=accent,
                    confirm_fg=confirm_fg,
                )
            )
            self._precision_btn.setIcon(
                themed_fa_icon("fa5s.gem", confirm_fg, 14)
            )
            self._precision_btn.setIconSize(QSize(14, 14))
            _apply_mode_button_tooltip(
                self._precision_btn,
                "Semantic breakpoint chunking for maximum citation accuracy.",
            )
        else:
            from core.theme.widget_styles import MUTED_STATUS

            self._precision_btn.setStyleSheet(
                theme.style(
                    PRESTIGE_DIALOG_MODE_OPTION,
                    btn_base=_PRESTIGE_BTN_BASE,
                    inactive=True,
                )
            )
            muted = theme.color(MUTED_STATUS)
            self._precision_btn.setIcon(
                themed_fa_icon("fa5s.gem", muted, 14)
            )
            self._precision_btn.setIconSize(QSize(14, 14))
            self._precision_btn.setCursor(Qt.CursorShape.ForbiddenCursor)
            _apply_mode_button_tooltip(
                self._precision_btn,
                precision_ingest_license_message(),
            )
        self._normal_btn.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

        mode_row.addWidget(self._normal_btn, stretch=1)
        mode_row.addWidget(self._precision_btn, stretch=1)
        c_layout.addLayout(mode_row)

        cancel_row = QHBoxLayout()
        cancel_btn = QPushButton("CANCEL")
        cancel_btn.setStyleSheet(
            theme.style(PRESTIGE_GHOST_BUTTON, btn_base=_PRESTIGE_BTN_BASE, accent=accent)
        )
        cancel_btn.clicked.connect(self.reject)
        cancel_row.addStretch()
        cancel_row.addWidget(cancel_btn)
        c_layout.addLayout(cancel_row)

        self._normal_btn.clicked.connect(self._choose_normal)
        self._precision_btn.clicked.connect(self._on_precision_clicked)

        layout.addWidget(container)

        if default_import_ingest_mode() == INGEST_MODE_PRECISION and licensed:
            self._precision_btn.setDefault(True)
            self._precision_btn.setFocus()
        else:
            self._normal_btn.setDefault(True)
            self._normal_btn.setFocus()

    @staticmethod
    def _message_text(file_count: int | None) -> str:
        body = (
            "Normal indexing uses fast structural chunking.\n\n"
            "Precision indexing (Pro) uses embedding-similarity breakpoints for "
            "denser documents. It can take much longer (often 10–100× more embedding work), "
            "but may be more accurate for citation extraction. This may be worth it to you if "
            "you are a library, publisher, or other individual or organization for which maximum citation accuracy is critical."
        )
        if file_count and file_count > 0:
            noun = "file" if file_count == 1 else "files"
            return f"You are importing {file_count} {noun}.\n\n{body}"
        return f"Choose how to index the files you select next.\n\n{body}"

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        apply_frameless_dialog_chrome(self)
        _center_dialog_on_host(self)
        self.raise_()
        self.activateWindow()

    def selected_mode(self) -> str | None:
        return self._selected_mode

    def _choose_normal(self) -> None:
        self._selected_mode = INGEST_MODE_STANDARD
        self.accept()

    def _on_precision_clicked(self) -> None:
        if not user_has_pro_library_ingest():
            from core.qube_tooltip import QubeToolTipController

            btn = self._precision_btn
            QubeToolTipController.instance().show_tip(
                btn,
                btn.mapToGlobal(btn.rect().center()),
                precision_ingest_license_message(),
            )
            return
        self._choose_precision()

    def _choose_precision(self) -> None:
        self._selected_mode = INGEST_MODE_PRECISION
        self.accept()
