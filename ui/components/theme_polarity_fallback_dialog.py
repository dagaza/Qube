"""Dialog when nav polarity toggle has no sibling variant in the active family."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QShowEvent
from PyQt6.QtWidgets import QDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from core.platform.frameless_window import (
    apply_frameless_dialog_chrome,
    configure_frameless_dialog_host,
)

from core.theme.families_policy import (
    NAV_FALLBACK_CANCEL_ACTION,
    NAV_FALLBACK_CHOOSE_THEME_ACTION,
    NAV_FALLBACK_MODAL_TITLE,
    NAV_FALLBACK_MODAL_TITLE_DARK,
)
from core.theme.polarity_toggle import PolarityToggleAction, PolarityToggleRequest
from core.theme.tokens import ThemeMode
from ui.components.prestige_dialog import _PRESTIGE_BTN_BASE, _center_dialog_on_host, _dialog_theme
from core.theme.widget_styles import (
    PRESTIGE_DIALOG_CANCEL,
    PRESTIGE_DIALOG_CONFIRM,
    PRESTIGE_DIALOG_CONTAINER,
    PRESTIGE_DIALOG_MESSAGE,
    PRESTIGE_DIALOG_TITLE,
    PRESTIGE_GHOST_BUTTON,
    prestige_accent_colors,
)


def prompt_theme_polarity_fallback(
    parent,
    request: PolarityToggleRequest,
    *,
    is_dark: bool,
) -> PolarityToggleAction:
    """Ask how to proceed when the current theme family lacks the target polarity."""
    dialog = QDialog(parent)
    dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
    configure_frameless_dialog_host(dialog)

    def _on_show(event: QShowEvent) -> None:
        QDialog.showEvent(dialog, event)
        apply_frameless_dialog_chrome(dialog)
        _center_dialog_on_host(dialog)

    dialog.showEvent = _on_show  # type: ignore[method-assign]

    theme = _dialog_theme(parent, is_dark)
    accent, confirm_fg = prestige_accent_colors(theme, tone="default", title="Theme")

    root = QVBoxLayout(dialog)
    root.setContentsMargins(10, 10, 10, 10)

    container = QFrame()
    container.setObjectName("DialogContainer")
    container.setStyleSheet(
        theme.style(PRESTIGE_DIALOG_CONTAINER, accent=accent, object_name="DialogContainer")
    )
    layout = QVBoxLayout(container)
    layout.setContentsMargins(30, 30, 30, 25)
    layout.setSpacing(20)

    title = (
        NAV_FALLBACK_MODAL_TITLE
        if request.target_mode is ThemeMode.LIGHT
        else NAV_FALLBACK_MODAL_TITLE_DARK
    )
    title_label = QLabel(title.upper())
    title_label.setStyleSheet(theme.style(PRESTIGE_DIALOG_TITLE, accent=accent))

    message_label = QLabel(request.message)
    message_label.setWordWrap(True)
    message_label.setStyleSheet(theme.style(PRESTIGE_DIALOG_MESSAGE))

    layout.addWidget(title_label)
    layout.addWidget(message_label)

    action_row = QHBoxLayout()
    action_row.setSpacing(12)

    cancel_btn = QPushButton(NAV_FALLBACK_CANCEL_ACTION)
    choose_btn = QPushButton(NAV_FALLBACK_CHOOSE_THEME_ACTION)
    fallback_btn = QPushButton(request.primary_action_label)

    cancel_btn.setStyleSheet(theme.style(PRESTIGE_DIALOG_CANCEL, btn_base=_PRESTIGE_BTN_BASE))
    choose_btn.setStyleSheet(theme.style(PRESTIGE_GHOST_BUTTON, btn_base=_PRESTIGE_BTN_BASE))
    fallback_btn.setStyleSheet(
        theme.style(
            PRESTIGE_DIALOG_CONFIRM,
            btn_base=_PRESTIGE_BTN_BASE,
            accent=accent,
            confirm_fg=confirm_fg,
        )
    )

    result = PolarityToggleAction.CANCEL

    def _set(action: PolarityToggleAction) -> None:
        nonlocal result
        result = action
        dialog.accept()

    cancel_btn.clicked.connect(lambda: _set(PolarityToggleAction.CANCEL))
    choose_btn.clicked.connect(lambda: _set(PolarityToggleAction.CHOOSE_THEME))
    fallback_btn.clicked.connect(lambda: _set(PolarityToggleAction.APPLY_FALLBACK))

    action_row.addWidget(cancel_btn)
    action_row.addStretch()
    action_row.addWidget(choose_btn)
    action_row.addWidget(fallback_btn)
    layout.addLayout(action_row)

    root.addWidget(container)
    dialog.adjustSize()
    dialog.exec()
    return result
