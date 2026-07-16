"""Theme-aware styling for Web search discovery cards (Settings → Knowledge)."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QSizePolicy, QWidget

_ROLE_KEYS = {
    "primary": "primary",
    "fallback": "fallback",
    "optional": "optional",
}

_ROLE_ACCENTS: dict[str, dict[str, tuple[str, str, str]]] = {
    "primary": {
        "dark": ("#89b4fa", "rgba(137, 180, 250, 0.22)", "rgba(137, 180, 250, 0.35)"),
        "light": ("#2563eb", "rgba(37, 99, 235, 0.12)", "rgba(37, 99, 235, 0.28)"),
    },
    "fallback": {
        "dark": ("#f9e2af", "rgba(249, 226, 175, 0.2)", "rgba(249, 226, 175, 0.38)"),
        "light": ("#b45309", "rgba(180, 83, 9, 0.12)", "rgba(180, 83, 9, 0.26)"),
    },
    "optional": {
        "dark": ("#cba6f7", "rgba(203, 166, 247, 0.2)", "rgba(203, 166, 247, 0.38)"),
        "light": ("#7c3aed", "rgba(124, 58, 237, 0.1)", "rgba(124, 58, 237, 0.24)"),
    },
}

_INFO_VARIANTS: dict[str, dict[str, tuple[str, str, str]]] = {
    "privacy": {
        "dark": ("#a6e3a1", "rgba(166, 227, 161, 0.12)", "rgba(166, 227, 161, 0.32)"),
        "light": ("#15803d", "rgba(21, 128, 61, 0.08)", "rgba(21, 128, 61, 0.22)"),
    },
    "policy": {
        "dark": ("#89b4fa", "rgba(137, 180, 250, 0.12)", "rgba(137, 180, 250, 0.32)"),
        "light": ("#2563eb", "rgba(37, 99, 235, 0.08)", "rgba(37, 99, 235, 0.22)"),
    },
}


def _theme_key(is_dark: bool) -> str:
    return "dark" if is_dark else "light"


def _repolish_widget(widget: QWidget) -> None:
    widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    widget.setAutoFillBackground(False)
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


def _normalize_role(role_label: str) -> str:
    return _ROLE_KEYS.get(role_label.strip().lower(), "fallback")


def apply_discovery_provider_card_theme(
    card: QWidget, *, role_label: str, is_dark: bool
) -> None:
    role = _normalize_role(role_label)
    accent, bg_tint, border = _ROLE_ACCENTS[role][_theme_key(is_dark)]
    card.setObjectName("DiscoveryProviderCard")
    card.setProperty("discovery_role", role)
    card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    # Opaque surfaces: semi-transparent shells looked washed out on section cards
    # and briefly kept the previous theme tint after a light/dark toggle.
    shell_bg = "#313244" if is_dark else "#ffffff"
    card.setStyleSheet(
        f"""
        QWidget#DiscoveryProviderCard {{
            background-color: {shell_bg};
            border: 1px solid {border};
            border-left: 3px solid {accent};
            border-radius: 10px;
        }}
    """
    )
    _repolish_widget(card)


def style_discovery_role_chip(label: QLabel, *, role_label: str, is_dark: bool) -> None:
    role = _normalize_role(role_label)
    accent, bg_tint, border = _ROLE_ACCENTS[role][_theme_key(is_dark)]
    background = bg_tint if is_dark else "#ffffff"
    label.setObjectName("DiscoveryCardRoleChip")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    label.setStyleSheet(
        f"""
        QLabel#DiscoveryCardRoleChip {{
            color: {accent} !important;
            background-color: {background} !important;
            border: 1px solid {border} !important;
            border-radius: 6px;
            padding: 3px 8px;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.08em;
        }}
    """
    )
    _repolish_widget(label)


def style_discovery_provider_name(label: QLabel, *, is_dark: bool) -> None:
    color = "#cdd6f4" if is_dark else "#0f172a"
    label.setObjectName("DiscoveryCardProviderName")
    label.setStyleSheet(
        f"""
        QLabel#DiscoveryCardProviderName {{
            color: {color};
            font-size: 15px;
            font-weight: 600;
            background: transparent;
            border: none;
            padding: 0;
        }}
    """
    )


def style_discovery_privacy_chip(label: QLabel, *, is_dark: bool) -> None:
    fg = "#a6adc8" if is_dark else "#64748b"
    bg = "rgba(166, 173, 200, 0.14)" if is_dark else "#ffffff"
    border = "rgba(166, 173, 200, 0.22)" if is_dark else "rgba(148, 163, 184, 0.28)"
    label.setObjectName("DiscoveryCardPrivacyChip")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    label.setStyleSheet(
        f"""
        QLabel#DiscoveryCardPrivacyChip {{
            color: {fg} !important;
            background-color: {bg} !important;
            border: 1px solid {border} !important;
            border-radius: 6px;
            padding: 2px 8px;
            font-size: 10px;
            font-weight: 500;
        }}
    """
    )
    _repolish_widget(label)


def style_discovery_body_text(label: QLabel, *, is_dark: bool) -> None:
    color = "#bac2de" if is_dark else "#475569"
    label.setObjectName("DiscoveryCardBody")
    label.setStyleSheet(
        f"""
        QLabel#DiscoveryCardBody {{
            color: {color};
            font-size: 12px;
            font-weight: 400;
            line-height: 1.45;
            background: transparent;
            border: none;
            padding: 0;
        }}
    """
    )


def build_discovery_divider(*, is_dark: bool) -> QWidget:
    line = QWidget()
    line.setFixedHeight(1)
    line.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    color = "rgba(255, 255, 255, 0.08)" if is_dark else "rgba(148, 163, 184, 0.28)"
    line.setStyleSheet(f"background-color: {color}; border: none;")
    return line


def apply_discovery_info_card_theme(
    card: QWidget, *, variant: str, is_dark: bool
) -> None:
    key = variant if variant in _INFO_VARIANTS else "policy"
    accent, bg_tint, border = _INFO_VARIANTS[key][_theme_key(is_dark)]
    card.setObjectName("DiscoveryInfoCard")
    card.setProperty("discovery_info_variant", key)
    card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    card.setStyleSheet(
        f"""
        QWidget#DiscoveryInfoCard {{
            background-color: {bg_tint};
            border: 1px solid {border};
            border-top: 2px solid {accent};
            border-radius: 10px;
        }}
    """
    )


def style_discovery_info_title(label: QLabel, *, variant: str, is_dark: bool) -> None:
    key = variant if variant in _INFO_VARIANTS else "policy"
    accent, _, _ = _INFO_VARIANTS[key][_theme_key(is_dark)]
    label.setObjectName("DiscoveryInfoCardTitle")
    label.setStyleSheet(
        f"""
        QLabel#DiscoveryInfoCardTitle {{
            color: {accent};
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.1em;
            background: transparent;
            border: none;
            padding: 0;
        }}
    """
    )


def style_discovery_info_highlight(label: QLabel, *, is_dark: bool) -> None:
    fg = "#cdd6f4" if is_dark else "#0f172a"
    bg = "rgba(137, 180, 250, 0.1)" if is_dark else "rgba(37, 99, 235, 0.06)"
    border = "rgba(137, 180, 250, 0.2)" if is_dark else "rgba(37, 99, 235, 0.14)"
    label.setObjectName("DiscoveryInfoHighlight")
    label.setStyleSheet(
        f"""
        QLabel#DiscoveryInfoHighlight {{
            color: {fg};
            background-color: {bg};
            border: 1px solid {border};
            border-radius: 8px;
            padding: 8px 10px;
            font-size: 12px;
            font-weight: 600;
        }}
    """
    )


def style_discovery_info_kv_key(label: QLabel, *, is_dark: bool) -> None:
    color = "#6c7086" if is_dark else "#64748b"
    label.setObjectName("DiscoveryInfoKvKey")
    label.setStyleSheet(
        f"""
        QLabel#DiscoveryInfoKvKey {{
            color: {color};
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.04em;
            background: transparent;
            border: none;
            padding: 0;
        }}
    """
    )


def style_discovery_info_kv_value(label: QLabel, *, is_dark: bool) -> None:
    color = "#cdd6f4" if is_dark else "#1e293b"
    label.setObjectName("DiscoveryInfoKvValue")
    label.setStyleSheet(
        f"""
        QLabel#DiscoveryInfoKvValue {{
            color: {color};
            font-size: 12px;
            font-weight: 500;
            background: transparent;
            border: none;
            padding: 0;
        }}
    """
    )


def style_discovery_info_bullet(label: QLabel, *, is_dark: bool) -> None:
    color = "#a6adc8" if is_dark else "#475569"
    label.setObjectName("DiscoveryInfoBullet")
    label.setStyleSheet(
        f"""
        QLabel#DiscoveryInfoBullet {{
            color: {color};
            font-size: 12px;
            font-weight: 400;
            background: transparent;
            border: none;
            padding: 2px 0 2px 2px;
        }}
    """
    )


def style_discovery_info_status(label: QLabel, *, is_dark: bool) -> None:
    color = "#6c7086" if is_dark else "#64748b"
    label.setObjectName("DiscoveryInfoStatus")
    label.setStyleSheet(
        f"""
        QLabel#DiscoveryInfoStatus {{
            color: {color};
            font-size: 11px;
            font-weight: 400;
            font-style: italic;
            background: transparent;
            border: none;
            padding: 1px 0;
        }}
    """
    )
