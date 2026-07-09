"""Widget-level styling for Live sources access status and action controls."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QPushButton, QSizePolicy, QWidget

STATUS_COLUMN_WIDTH_PX = 168
ACTION_COLUMN_WIDTH_PX = 92
ACTION_CONTROL_HEIGHT_PX = 28

_ACCESS_BADGE_STYLES: dict[str, dict[str, tuple[str, str]]] = {
    "optional_key": {
        "dark": ("#f9e2af", "rgba(249, 226, 175, 0.18)"),
        "light": ("#b45309", "rgba(180, 83, 9, 0.12)"),
    },
    "key_required": {
        "dark": ("#f38ba8", "rgba(243, 139, 168, 0.18)"),
        "light": ("#be123c", "rgba(190, 18, 60, 0.12)"),
    },
    "connected": {
        "dark": ("#a6e3a1", "rgba(166, 227, 161, 0.18)"),
        "light": ("#15803d", "rgba(21, 128, 61, 0.12)"),
    },
    "env_override": {
        "dark": ("#89b4fa", "rgba(137, 180, 250, 0.18)"),
        "light": ("#1d4ed8", "rgba(29, 78, 216, 0.12)"),
    },
    "coming_soon": {
        "dark": ("#6c7086", "rgba(108, 112, 134, 0.22)"),
        "light": ("#94a3b8", "rgba(148, 163, 184, 0.18)"),
    },
}


def resolve_settings_is_dark(host) -> bool:
    window = host.window() if hasattr(host, "window") else None
    return getattr(window, "_is_dark_theme", True)


def style_access_badge(label: QLabel, access: str, *, is_dark: bool) -> None:
    """Apply theme-stable pill styling on the status badge label."""
    theme_key = "dark" if is_dark else "light"
    fg, bg = _ACCESS_BADGE_STYLES.get(access, _ACCESS_BADGE_STYLES["coming_soon"])[theme_key]
    label.setObjectName("KnowledgeAccessBadge")
    label.setProperty("access", access)
    label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    label.setStyleSheet(
        f"""
        QLabel#KnowledgeAccessBadge {{
            padding: 3px 10px;
            border-radius: 10px;
            font-size: 11px;
            font-weight: 600;
            color: {fg};
            background-color: {bg};
        }}
    """
    )


def style_access_hint(label: QLabel, *, is_dark: bool) -> None:
    color = "#6c7086" if is_dark else "#64748b"
    label.setObjectName("KnowledgeAccessHint")
    label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    label.setWordWrap(True)
    label.setStyleSheet(
        f"""
        QLabel#KnowledgeAccessHint {{
            color: {color};
            font-size: 11px;
            font-weight: normal;
            background: transparent;
            border: none;
            padding: 0;
        }}
    """
    )


def style_configure_button(button: QPushButton, *, is_dark: bool) -> None:
    fg = "#89b4fa" if is_dark else "#2563eb"
    border = "rgba(137, 180, 250, 0.35)" if is_dark else "rgba(37, 99, 235, 0.25)"
    hover_bg = "rgba(137, 180, 250, 0.12)" if is_dark else "rgba(37, 99, 235, 0.08)"
    _apply_action_button_style(
        button,
        text_color=fg,
        border_color=border,
        background="transparent",
        hover_background=hover_bg,
        object_name="KnowledgeConfigureButton",
    )


def style_free_action_button(button: QPushButton, *, is_dark: bool) -> None:
    fg = "#a6e3a1" if is_dark else "#15803d"
    border = "rgba(166, 227, 161, 0.35)" if is_dark else "rgba(21, 128, 61, 0.28)"
    bg = "rgba(166, 227, 161, 0.08)" if is_dark else "rgba(21, 128, 61, 0.06)"
    _apply_action_button_style(
        button,
        text_color=fg,
        border_color=border,
        background=bg,
        hover_background=bg,
        object_name="KnowledgeFreeButton",
    )


def _apply_action_button_style(
    button: QPushButton,
    *,
    text_color: str,
    border_color: str,
    background: str,
    hover_background: str,
    object_name: str,
) -> None:
    button.setObjectName(object_name)
    button.setFixedSize(ACTION_COLUMN_WIDTH_PX, ACTION_CONTROL_HEIGHT_PX)
    button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    button.setStyleSheet(
        f"""
        QPushButton#{object_name} {{
            padding: 4px 10px;
            border-radius: 8px;
            font-size: 11px;
            font-weight: 600;
            color: {text_color};
            border: 1px solid {border_color};
            background: {background};
        }}
        QPushButton#{object_name}:hover {{
            background: {hover_background};
        }}
        QPushButton#{object_name}:disabled {{
            color: {text_color};
            border: 1px solid {border_color};
            background: {background};
        }}
    """
    )


_SETUP_CALLOUT_STYLES: dict[str, dict[str, str]] = {
    "dark": {
        "card_bg": "rgba(249, 226, 175, 0.1)",
        "card_border": "rgba(249, 226, 175, 0.32)",
        "title": "#f9e2af",
        "body": "#bac2de",
        "dismiss_fg": "#a6adc8",
        "dismiss_border": "rgba(255, 255, 255, 0.1)",
        "dismiss_hover": "rgba(255, 255, 255, 0.06)",
    },
    "light": {
        "card_bg": "rgba(251, 191, 36, 0.1)",
        "card_border": "rgba(180, 83, 9, 0.22)",
        "title": "#b45309",
        "body": "#475569",
        "dismiss_fg": "#64748b",
        "dismiss_border": "#e2e8f0",
        "dismiss_hover": "rgba(148, 163, 184, 0.12)",
    },
}


def style_setup_callout_card(card: QWidget, *, is_dark: bool) -> None:
    theme = _SETUP_CALLOUT_STYLES["dark" if is_dark else "light"]
    card.setObjectName("KnowledgeSetupCallout")
    card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    card.setStyleSheet(
        f"""
        QWidget#KnowledgeSetupCallout {{
            background-color: {theme["card_bg"]};
            border: 1px solid {theme["card_border"]};
            border-radius: 8px;
        }}
    """
    )


def style_setup_callout_title(label: QLabel, *, is_dark: bool) -> None:
    theme = _SETUP_CALLOUT_STYLES["dark" if is_dark else "light"]
    label.setObjectName("KnowledgeSetupCalloutTitle")
    label.setStyleSheet(
        f"""
        QLabel#KnowledgeSetupCalloutTitle {{
            color: {theme["title"]};
            font-size: 11px;
            font-weight: 600;
            background: transparent;
            border: none;
            padding: 0;
        }}
    """
    )


def style_setup_callout_body(label: QLabel, *, is_dark: bool) -> None:
    theme = _SETUP_CALLOUT_STYLES["dark" if is_dark else "light"]
    label.setObjectName("KnowledgeSetupCalloutBody")
    label.setStyleSheet(
        f"""
        QLabel#KnowledgeSetupCalloutBody {{
            color: {theme["body"]};
            font-size: 12px;
            font-weight: 400;
            background: transparent;
            border: none;
            padding: 0;
        }}
    """
    )


def style_setup_callout_dismiss(button: QPushButton, *, is_dark: bool) -> None:
    theme = _SETUP_CALLOUT_STYLES["dark" if is_dark else "light"]
    button.setObjectName("KnowledgeSetupCalloutDismiss")
    button.setFixedHeight(ACTION_CONTROL_HEIGHT_PX)
    button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    button.setStyleSheet(
        f"""
        QPushButton#KnowledgeSetupCalloutDismiss {{
            padding: 4px 12px;
            border-radius: 8px;
            font-size: 11px;
            font-weight: 600;
            color: {theme["dismiss_fg"]};
            border: 1px solid {theme["dismiss_border"]};
            background: transparent;
        }}
        QPushButton#KnowledgeSetupCalloutDismiss:hover {{
            background: {theme["dismiss_hover"]};
        }}
    """
    )


def apply_setup_callout_theme(callout, *, is_dark: bool) -> None:
    """Re-apply Prestige styling on the recommended-setup callout."""
    style_setup_callout_card(callout, is_dark=is_dark)
    style_setup_callout_title(callout.title_label, is_dark=is_dark)
    style_setup_callout_body(callout.body_label, is_dark=is_dark)
    style_setup_callout_dismiss(callout.dismiss_btn, is_dark=is_dark)
