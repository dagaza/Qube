"""Backward-compatible aliases for Live sources access styling.

Prefer ``ui.views.settings.primitives`` for new Settings UI work.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QPushButton, QSizePolicy, QWidget

from core.theme.widget_styles import (
    KNOWLEDGE_SETUP_CALLOUT,
    KNOWLEDGE_SETUP_CALLOUT_BODY,
    KNOWLEDGE_SETUP_CALLOUT_DISMISS,
    KNOWLEDGE_SETUP_CALLOUT_TITLE,
)
from ui.views.settings.primitives import (
    ACTION_COLUMN_WIDTH_PX,
    ACTION_CONTROL_HEIGHT_PX,
    ACTION_ROW_BOTTOM_INSET_PX,
    STATUS_COLUMN_WIDTH_PX,
    apply_settings_callout_theme,
    coalesce_settings_is_dark,
    make_settings_action_row as make_knowledge_configure_action_row,
    repolish_widget,
    resolve_settings_is_dark,
    settings_theme,
    style_settings_access_hint as style_access_hint,
    style_settings_configure_button as style_configure_button,
    style_settings_free_button as style_free_action_button,
    style_settings_status_chip as style_access_badge,
)

__all__ = [
    "ACTION_COLUMN_WIDTH_PX",
    "ACTION_CONTROL_HEIGHT_PX",
    "ACTION_ROW_BOTTOM_INSET_PX",
    "STATUS_COLUMN_WIDTH_PX",
    "apply_setup_callout_theme",
    "coalesce_settings_is_dark",
    "make_knowledge_configure_action_row",
    "resolve_settings_is_dark",
    "style_access_badge",
    "style_access_hint",
    "style_configure_button",
    "style_free_action_button",
    "style_setup_callout_body",
    "style_setup_callout_card",
    "style_setup_callout_dismiss",
    "style_setup_callout_title",
]


def style_setup_callout_card(card: QWidget, *, is_dark: bool, host=None) -> None:
    theme = settings_theme(is_dark=is_dark, host=host)
    card.setObjectName("KnowledgeSetupCallout")
    card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    card.setStyleSheet(theme.style(KNOWLEDGE_SETUP_CALLOUT))
    repolish_widget(card)


def style_setup_callout_title(label: QLabel, *, is_dark: bool, host=None) -> None:
    theme = settings_theme(is_dark=is_dark, host=host)
    label.setObjectName("KnowledgeSetupCalloutTitle")
    label.setStyleSheet(theme.style(KNOWLEDGE_SETUP_CALLOUT_TITLE))
    repolish_widget(label)


def style_setup_callout_body(label: QLabel, *, is_dark: bool, host=None) -> None:
    theme = settings_theme(is_dark=is_dark, host=host)
    label.setObjectName("KnowledgeSetupCalloutBody")
    label.setStyleSheet(theme.style(KNOWLEDGE_SETUP_CALLOUT_BODY))
    repolish_widget(label)


def style_setup_callout_dismiss(button: QPushButton, *, is_dark: bool, host=None) -> None:
    theme = settings_theme(is_dark=is_dark, host=host)
    button.setObjectName("KnowledgeSetupCalloutDismiss")
    button.setFixedHeight(ACTION_CONTROL_HEIGHT_PX)
    button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    button.setStyleSheet(theme.style(KNOWLEDGE_SETUP_CALLOUT_DISMISS))
    repolish_widget(button)


def apply_setup_callout_theme(callout, *, is_dark: bool, host=None) -> None:
    from ui.views.settings.primitives.callouts import SettingsCallout

    if isinstance(callout, SettingsCallout):
        apply_settings_callout_theme(callout, is_dark=is_dark, host=host)
        return
    style_setup_callout_card(callout, is_dark=is_dark, host=host)
    style_setup_callout_title(callout.title_label, is_dark=is_dark, host=host)
    style_setup_callout_body(callout.body_label, is_dark=is_dark, host=host)
    style_setup_callout_dismiss(callout.dismiss_btn, is_dark=is_dark, host=host)
