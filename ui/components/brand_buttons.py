"""Prestige brand-action button styling helper.

Widget-level QSS drives rendering from ``ResolvedTheme`` semantic tokens.
Logo/identity colors live in ``core.brand_identity`` and are not user-customizable.
"""

from __future__ import annotations

from typing import Optional

import qtawesome as qta
from PyQt6.QtWidgets import QPushButton

from core.theme.accessors import theme_for
from core.theme.color_utils import adjust_lightness, with_alpha
from core.theme.tokens import ResolvedTheme

BRAND_PRIMARY = "primary"
BRAND_SUCCESS = "success"
BRAND_DANGER = "danger"
BRAND_CAUTION = "caution"

_BRAND_VARIANTS = frozenset(
    {BRAND_PRIMARY, BRAND_SUCCESS, BRAND_DANGER, BRAND_CAUTION}
)


def brand_fg_color(
    variant: str,
    *,
    theme: ResolvedTheme | None = None,
    is_dark: bool = True,
) -> str:
    if variant not in _BRAND_VARIANTS:
        raise ValueError(
            f"Unknown brand variant: {variant!r}. "
            f"Expected one of: {sorted(_BRAND_VARIANTS)}"
        )
    return theme_for(is_dark=is_dark, resolved=theme).brand_fg


def _brand_disabled_qss(theme: ResolvedTheme) -> str:
    return (
        f"background-color: {theme.brand_disabled_bg};"
        f" color: {theme.brand_disabled_fg};"
        f" border: 1px solid {with_alpha(theme.border, 0.35)};"
    )


def _semantic_button_qss(
    theme: ResolvedTheme,
    *,
    base: str,
    hover: str,
    pressed: str,
    border: str,
) -> str:
    fg = theme.brand_fg
    disabled = _brand_disabled_qss(theme)
    return f"""
        QPushButton {{
            background-color: {base};
            color: {fg};
            border: 1px solid {border};
            border-radius: 6px;
            padding: 8px 15px;
            font-weight: 700;
        }}
        QPushButton:hover {{
            background-color: {hover};
            border: 1px solid {hover};
        }}
        QPushButton:pressed {{
            background-color: {pressed};
            border: 1px solid {pressed};
        }}
        QPushButton:disabled {{
            {disabled}
        }}
    """


def brand_qss_for_variant(variant: str, theme: ResolvedTheme) -> str:
    disabled = _brand_disabled_qss(theme)
    if variant == BRAND_PRIMARY:
        return f"""
        QPushButton {{
            background-color: {theme.accent};
            color: {theme.brand_fg};
            border: 1px solid {theme.accent};
            border-radius: 6px;
            padding: 8px 15px;
            font-weight: 700;
        }}
        QPushButton:hover {{
            background-color: {theme.accent_hover};
            border: 1px solid {theme.accent_hover};
        }}
        QPushButton:pressed {{
            background-color: {theme.accent_pressed};
            border: 1px solid {theme.accent_pressed};
        }}
        QPushButton:disabled {{
            {disabled}
        }}
    """
    if variant == BRAND_SUCCESS:
        return _semantic_button_qss(
            theme,
            base=theme.success,
            hover=adjust_lightness(theme.success, -0.08),
            pressed=adjust_lightness(theme.success, -0.16),
            border=adjust_lightness(theme.success, -0.12),
        )
    if variant == BRAND_DANGER:
        return _semantic_button_qss(
            theme,
            base=theme.error,
            hover=adjust_lightness(theme.error, -0.08),
            pressed=adjust_lightness(theme.error, -0.16),
            border=adjust_lightness(theme.error, -0.12),
        )
    if variant == BRAND_CAUTION:
        return _semantic_button_qss(
            theme,
            base=theme.warning,
            hover=adjust_lightness(theme.warning, -0.08),
            pressed=adjust_lightness(theme.warning, -0.16),
            border=adjust_lightness(theme.warning, -0.12),
        )
    raise ValueError(f"Unknown brand variant: {variant!r}")


def apply_brand_style(
    button: QPushButton,
    variant: str,
    icon_name: Optional[str] = None,
    *,
    theme: ResolvedTheme | None = None,
    is_dark: bool | None = None,
) -> None:
    resolved = theme_for(is_dark=is_dark if is_dark is not None else True, resolved=theme)
    qss = brand_qss_for_variant(variant, resolved)

    if variant == BRAND_PRIMARY:
        class_tag = "PrimaryActionButton BrandPrimaryButton"
    elif variant == BRAND_SUCCESS:
        class_tag = "PrimaryActionButton BrandSuccessButton"
    elif variant == BRAND_CAUTION:
        class_tag = "PrimaryActionButton BrandCautionButton"
    else:
        class_tag = "PrimaryActionButton BrandDangerButton"

    button.setProperty("class", class_tag)
    button.setStyleSheet(qss)

    if icon_name is not None:
        button.setIcon(
            qta.icon(
                icon_name,
                color=resolved.brand_fg,
                color_disabled=resolved.brand_disabled_fg,
            )
        )

    style = button.style()
    if style is not None:
        style.unpolish(button)
        style.polish(button)
    button.update()


def apply_brand_primary(
    button: QPushButton,
    icon_name: Optional[str] = None,
    *,
    theme: ResolvedTheme | None = None,
    is_dark: bool | None = None,
) -> None:
    apply_brand_style(
        button, BRAND_PRIMARY, icon_name=icon_name, theme=theme, is_dark=is_dark
    )


def apply_brand_success(
    button: QPushButton,
    icon_name: Optional[str] = None,
    *,
    theme: ResolvedTheme | None = None,
    is_dark: bool | None = None,
) -> None:
    apply_brand_style(
        button, BRAND_SUCCESS, icon_name=icon_name, theme=theme, is_dark=is_dark
    )


def apply_brand_danger(
    button: QPushButton,
    icon_name: Optional[str] = None,
    *,
    theme: ResolvedTheme | None = None,
    is_dark: bool | None = None,
) -> None:
    apply_brand_style(
        button, BRAND_DANGER, icon_name=icon_name, theme=theme, is_dark=is_dark
    )


def apply_brand_caution(
    button: QPushButton,
    icon_name: Optional[str] = None,
    *,
    theme: ResolvedTheme | None = None,
    is_dark: bool | None = None,
) -> None:
    apply_brand_style(
        button, BRAND_CAUTION, icon_name=icon_name, theme=theme, is_dark=is_dark
    )
