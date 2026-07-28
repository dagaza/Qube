"""Widget-level styling for Live sources access status and action controls."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QWidget

from core.theme.accessors import theme_for
from core.theme.view_theme import view_resolved_theme
from core.theme.widget_styles import (
    KNOWLEDGE_ACCESS_BADGE,
    KNOWLEDGE_ACCESS_HINT,
    KNOWLEDGE_ACTION_BUTTON,
    KNOWLEDGE_SETUP_CALLOUT,
    KNOWLEDGE_SETUP_CALLOUT_BODY,
    KNOWLEDGE_SETUP_CALLOUT_DISMISS,
    KNOWLEDGE_SETUP_CALLOUT_TITLE,
)

STATUS_COLUMN_WIDTH_PX = 168
ACTION_COLUMN_WIDTH_PX = 92
ACTION_CONTROL_HEIGHT_PX = 32
ACTION_ROW_BOTTOM_INSET_PX = 4


def _settings_theme(host, *, is_dark: bool | None = None):
    return view_resolved_theme(host, is_dark=is_dark)


def make_knowledge_configure_action_row(button: QPushButton) -> QWidget:
    """Left-aligned configure row with bottom inset for SettingsLogCard layouts."""
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


def coalesce_settings_is_dark(host, *, is_dark: bool | None = None) -> bool:
    """Return the active settings theme, preferring MainWindow over stale cache."""
    window = host.window() if hasattr(host, "window") else None
    if window is not None and hasattr(window, "_is_dark_theme"):
        resolved = bool(window._is_dark_theme)
    elif is_dark is not None:
        resolved = bool(is_dark)
    else:
        resolved = bool(getattr(host, "_settings_ui_is_dark", True))
    host._settings_ui_is_dark = resolved
    return resolved


def resolve_settings_is_dark(host) -> bool:
    """Backward-compatible alias — always syncs from the window when possible."""
    return coalesce_settings_is_dark(host)


def _repolish_widget(widget: QWidget) -> None:
    widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    widget.setAutoFillBackground(False)
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


def style_access_badge(label: QLabel, access: str, *, is_dark: bool, host=None) -> None:
    """Apply theme-stable pill styling on the status badge label."""
    theme = _settings_theme(host, is_dark=is_dark) if host is not None else theme_for(is_dark=is_dark)
    label.setObjectName("KnowledgeAccessBadge")
    label.setProperty("access", access)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
    label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    label.setStyleSheet(theme.style(KNOWLEDGE_ACCESS_BADGE, access=access))
    label.setFixedHeight(ACTION_CONTROL_HEIGHT_PX)
    _repolish_widget(label)


def style_access_hint(label: QLabel, *, is_dark: bool, host=None) -> None:
    theme = _settings_theme(host, is_dark=is_dark) if host is not None else theme_for(is_dark=is_dark)
    label.setObjectName("KnowledgeAccessHint")
    label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    label.setWordWrap(True)
    label.setStyleSheet(theme.style(KNOWLEDGE_ACCESS_HINT))
    _repolish_widget(label)


def style_configure_button(button: QPushButton, *, is_dark: bool, host=None) -> None:
    _apply_action_button_style(
        button,
        is_dark=is_dark,
        host=host,
        variant="configure",
        object_name="KnowledgeConfigureButton",
    )


def style_free_action_button(button: QPushButton, *, is_dark: bool, host=None) -> None:
    _apply_action_button_style(
        button,
        is_dark=is_dark,
        host=host,
        variant="free",
        object_name="KnowledgeFreeButton",
    )


def _apply_action_button_style(
    button: QPushButton,
    *,
    is_dark: bool,
    host=None,
    variant: str,
    object_name: str,
) -> None:
    theme = _settings_theme(host, is_dark=is_dark) if host is not None else theme_for(is_dark=is_dark)
    button.setObjectName(object_name)
    button.setFixedWidth(ACTION_COLUMN_WIDTH_PX)
    button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    button.setStyleSheet(
        theme.style(KNOWLEDGE_ACTION_BUTTON, variant=variant, object_name=object_name)
    )
    _repolish_widget(button)
    button.setFixedHeight(ACTION_CONTROL_HEIGHT_PX)


def style_setup_callout_card(card: QWidget, *, is_dark: bool, host=None) -> None:
    theme = _settings_theme(host, is_dark=is_dark) if host is not None else theme_for(is_dark=is_dark)
    card.setObjectName("KnowledgeSetupCallout")
    card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    card.setStyleSheet(theme.style(KNOWLEDGE_SETUP_CALLOUT))
    _repolish_widget(card)


def style_setup_callout_title(label: QLabel, *, is_dark: bool, host=None) -> None:
    theme = _settings_theme(host, is_dark=is_dark) if host is not None else theme_for(is_dark=is_dark)
    label.setObjectName("KnowledgeSetupCalloutTitle")
    label.setStyleSheet(theme.style(KNOWLEDGE_SETUP_CALLOUT_TITLE))
    _repolish_widget(label)


def style_setup_callout_body(label: QLabel, *, is_dark: bool, host=None) -> None:
    theme = _settings_theme(host, is_dark=is_dark) if host is not None else theme_for(is_dark=is_dark)
    label.setObjectName("KnowledgeSetupCalloutBody")
    label.setStyleSheet(theme.style(KNOWLEDGE_SETUP_CALLOUT_BODY))
    _repolish_widget(label)


def style_setup_callout_dismiss(button: QPushButton, *, is_dark: bool, host=None) -> None:
    theme = _settings_theme(host, is_dark=is_dark) if host is not None else theme_for(is_dark=is_dark)
    button.setObjectName("KnowledgeSetupCalloutDismiss")
    button.setFixedHeight(ACTION_CONTROL_HEIGHT_PX)
    button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    button.setStyleSheet(theme.style(KNOWLEDGE_SETUP_CALLOUT_DISMISS))
    _repolish_widget(button)


def apply_setup_callout_theme(callout, *, is_dark: bool, host=None) -> None:
    """Re-apply Prestige styling on the recommended-setup callout."""
    style_setup_callout_card(callout, is_dark=is_dark, host=host)
    style_setup_callout_title(callout.title_label, is_dark=is_dark, host=host)
    style_setup_callout_body(callout.body_label, is_dark=is_dark, host=host)
    style_setup_callout_dismiss(callout.dismiss_btn, is_dark=is_dark, host=host)
