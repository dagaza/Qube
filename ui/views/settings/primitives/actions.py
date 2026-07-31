"""Compact action buttons used in Settings rich rows and nested cards."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QSizePolicy, QWidget

from core.theme.widget_styles import KNOWLEDGE_ACTION_BUTTON
from ui.views.settings.primitives.theme import repolish_widget, settings_theme

STATUS_COLUMN_WIDTH_PX = 168
ACTION_COLUMN_WIDTH_PX = 92
ACTION_CONTROL_HEIGHT_PX = 32
ACTION_ROW_BOTTOM_INSET_PX = 4


def make_settings_action_row(button: QPushButton) -> QWidget:
    """Left-aligned action row with bottom inset for nested card layouts."""
    row = QWidget()
    row_layout = QHBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, ACTION_ROW_BOTTOM_INSET_PX)
    row_layout.setSpacing(0)
    row_layout.addWidget(
        button,
        alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
    )
    row_layout.addStretch(1)
    row.setMinimumHeight(button.sizeHint().height() + ACTION_ROW_BOTTOM_INSET_PX)
    return row


def style_settings_action_button(
    button: QPushButton,
    *,
    variant: str,
    is_dark: bool,
    host=None,
    object_name: str = "KnowledgeConfigureButton",
) -> None:
    theme = settings_theme(is_dark=is_dark, host=host)
    button.setObjectName(object_name)
    button.setFixedWidth(ACTION_COLUMN_WIDTH_PX)
    button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    button.setStyleSheet(
        theme.style(KNOWLEDGE_ACTION_BUTTON, variant=variant, object_name=object_name)
    )
    repolish_widget(button)
    button.setFixedHeight(ACTION_CONTROL_HEIGHT_PX)


def style_settings_configure_button(
    button: QPushButton,
    *,
    is_dark: bool,
    host=None,
) -> None:
    style_settings_action_button(
        button,
        variant="configure",
        is_dark=is_dark,
        host=host,
        object_name="KnowledgeConfigureButton",
    )


def style_settings_free_button(
    button: QPushButton,
    *,
    is_dark: bool,
    host=None,
) -> None:
    style_settings_action_button(
        button,
        variant="free",
        is_dark=is_dark,
        host=host,
        object_name="KnowledgeFreeButton",
    )
