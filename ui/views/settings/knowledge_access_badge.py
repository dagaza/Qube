"""Widget-level styling for Live sources access status and action controls."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QWidget

STATUS_COLUMN_WIDTH_PX = 168
ACTION_COLUMN_WIDTH_PX = 92
ACTION_CONTROL_HEIGHT_PX = 32
ACTION_ROW_BOTTOM_INSET_PX = 4

# Opaque elevated surfaces so chips/buttons read clearly above section cards
# (#E9EFF5 light / #232337 dark). Semantic meaning comes from text + border only.
_LIGHT_SURFACE = "#ffffff"
_DARK_SURFACE = "#2e3048"

# (foreground, background, border)
_ACCESS_BADGE_STYLES: dict[str, dict[str, tuple[str, str, str]]] = {
    "free": {
        "dark": ("#bac2de", _DARK_SURFACE, "#5c6078"),
        "light": ("#475569", _LIGHT_SURFACE, "#94a3b8"),
    },
    "optional_key": {
        "dark": ("#f9e2af", _DARK_SURFACE, "#9a7b3c"),
        "light": ("#b45309", _LIGHT_SURFACE, "#f59e0b"),
    },
    "key_required": {
        "dark": ("#f38ba8", _DARK_SURFACE, "#9a4a62"),
        "light": ("#be123c", _LIGHT_SURFACE, "#f43f5e"),
    },
    "connected": {
        "dark": ("#a6e3a1", _DARK_SURFACE, "#4f7a55"),
        "light": ("#15803d", _LIGHT_SURFACE, "#22c55e"),
    },
    "env_override": {
        "dark": ("#89b4fa", _DARK_SURFACE, "#4f6a9a"),
        "light": ("#1d4ed8", _LIGHT_SURFACE, "#3b82f6"),
    },
    "coming_soon": {
        "dark": ("#a6adc8", _DARK_SURFACE, "#5c6078"),
        "light": ("#64748b", _LIGHT_SURFACE, "#94a3b8"),
    },
}


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


def style_access_badge(label: QLabel, access: str, *, is_dark: bool) -> None:
    """Apply theme-stable pill styling on the status badge label."""
    theme_key = "dark" if is_dark else "light"
    fg, bg, border = _ACCESS_BADGE_STYLES.get(
        access, _ACCESS_BADGE_STYLES["coming_soon"]
    )[theme_key]
    label.setObjectName("KnowledgeAccessBadge")
    label.setProperty("access", access)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
    label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    label.setStyleSheet(
        f"""
        QLabel#KnowledgeAccessBadge {{
            padding: 0 10px;
            border-radius: 10px;
            font-size: 11px;
            font-weight: 600;
            color: {fg} !important;
            background-color: {bg} !important;
            border: 1px solid {border} !important;
        }}
    """
    )
    label.setFixedHeight(ACTION_CONTROL_HEIGHT_PX)
    _repolish_widget(label)


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
    _repolish_widget(label)


def style_configure_button(button: QPushButton, *, is_dark: bool) -> None:
    if is_dark:
        fg = "#89b4fa"
        border = "#5b7ec8"
        background = _DARK_SURFACE
        hover_bg = "#363a52"
    else:
        fg = "#1d4ed8"
        border = "#3b82f6"
        background = _LIGHT_SURFACE
        hover_bg = "#eff6ff"
    _apply_action_button_style(
        button,
        text_color=fg,
        border_color=border,
        background=background,
        hover_background=hover_bg,
        object_name="KnowledgeConfigureButton",
    )


def style_free_action_button(button: QPushButton, *, is_dark: bool) -> None:
    if is_dark:
        fg = "#a6e3a1"
        border = "#4f7a55"
        background = _DARK_SURFACE
        hover_bg = "#363a52"
    else:
        fg = "#15803d"
        border = "#22c55e"
        background = _LIGHT_SURFACE
        hover_bg = "#f0fdf4"
    _apply_action_button_style(
        button,
        text_color=fg,
        border_color=border,
        background=background,
        hover_background=hover_bg,
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
    button.setFixedWidth(ACTION_COLUMN_WIDTH_PX)
    button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    button.setStyleSheet(
        f"""
        QPushButton#{object_name} {{
            padding: 4px 10px;
            border-radius: 8px;
            font-size: 11px;
            font-weight: 600;
            color: {text_color} !important;
            border: 1px solid {border_color} !important;
            background-color: {background} !important;
        }}
        QPushButton#{object_name}:hover {{
            background-color: {hover_background} !important;
        }}
        QPushButton#{object_name}:disabled {{
            color: {text_color} !important;
            border: 1px solid {border_color} !important;
            background-color: {background} !important;
        }}
    """
    )
    _repolish_widget(button)
    button.setFixedHeight(ACTION_CONTROL_HEIGHT_PX)


_SETUP_CALLOUT_STYLES: dict[str, dict[str, str]] = {
    "dark": {
        "card_bg": _DARK_SURFACE,
        "card_border": "#9a7b3c",
        "title": "#f9e2af",
        "body": "#bac2de",
        "dismiss_fg": "#a6adc8",
        "dismiss_border": "#5c6078",
        "dismiss_hover": "#363a52",
    },
    "light": {
        "card_bg": _LIGHT_SURFACE,
        "card_border": "#f59e0b",
        "title": "#b45309",
        "body": "#475569",
        "dismiss_fg": "#64748b",
        "dismiss_border": "#cbd5e1",
        "dismiss_hover": "#f8fafc",
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
    _repolish_widget(card)


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
    _repolish_widget(label)


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
    _repolish_widget(label)


def style_setup_callout_dismiss(button: QPushButton, *, is_dark: bool) -> None:
    theme = _SETUP_CALLOUT_STYLES["dark" if is_dark else "light"]
    button.setObjectName("KnowledgeSetupCalloutDismiss")
    button.setFixedHeight(ACTION_CONTROL_HEIGHT_PX)
    button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    dismiss_bg = _DARK_SURFACE if is_dark else _LIGHT_SURFACE
    button.setStyleSheet(
        f"""
        QPushButton#KnowledgeSetupCalloutDismiss {{
            padding: 4px 12px;
            border-radius: 8px;
            font-size: 11px;
            font-weight: 600;
            color: {theme["dismiss_fg"]};
            border: 1px solid {theme["dismiss_border"]};
            background-color: {dismiss_bg};
        }}
        QPushButton#KnowledgeSetupCalloutDismiss:hover {{
            background-color: {theme["dismiss_hover"]};
        }}
    """
    )
    _repolish_widget(button)


def apply_setup_callout_theme(callout, *, is_dark: bool) -> None:
    """Re-apply Prestige styling on the recommended-setup callout."""
    style_setup_callout_card(callout, is_dark=is_dark)
    style_setup_callout_title(callout.title_label, is_dark=is_dark)
    style_setup_callout_body(callout.body_label, is_dark=is_dark)
    style_setup_callout_dismiss(callout.dismiss_btn, is_dark=is_dark)
