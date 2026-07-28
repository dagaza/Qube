"""Prestige brand-action button styling helper.

Widget-level QSS drives rendering from ``ResolvedTheme`` semantic tokens.
Logo/identity colors live in ``core.brand_identity`` and are not user-customizable.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QPushButton

from core.theme.accessors import theme_for
from core.theme.color_utils import adjust_lightness, with_alpha
from core.theme.tokens import ResolvedTheme

BRAND_PRIMARY = "primary"
BRAND_SUCCESS = "success"
BRAND_DANGER = "danger"
BRAND_CAUTION = "caution"
BRAND_SECONDARY = "secondary"

_BRAND_VARIANTS = frozenset(
    {BRAND_PRIMARY, BRAND_SUCCESS, BRAND_DANGER, BRAND_CAUTION, BRAND_SECONDARY}
)
_BRAND_ICON_SIZE = 16

def brand_label_color(
    variant: str,
    theme: ResolvedTheme,
    *,
    disabled: bool = False,
) -> str:
    """Foreground color for brand button labels and matching qtawesome icon tints."""
    if variant not in _BRAND_VARIANTS:
        raise ValueError(
            f"Unknown brand variant: {variant!r}. "
            f"Expected one of: {sorted(_BRAND_VARIANTS)}"
        )
    if disabled:
        return _brand_disabled_colors(theme, variant)[1]
    if variant == BRAND_SECONDARY:
        return theme.text_primary
    return theme.brand_fg


def brand_fg_color(
    variant: str,
    *,
    theme: ResolvedTheme | None = None,
    is_dark: bool = True,
    disabled: bool = False,
) -> str:
    resolved = theme_for(is_dark=is_dark, resolved=theme)
    return brand_label_color(variant, resolved, disabled=disabled)


def _brand_button_icon(icon_name: str, fg: str, disabled_fg: str) -> QIcon:
    """Bake qtawesome glyphs into static pixmaps so QSS/platform styles cannot retint them."""
    from core.theme.svg_icons import themed_fa_icon

    return themed_fa_icon(
        icon_name,
        fg,
        _BRAND_ICON_SIZE,
        disabled_color=disabled_fg,
    )


def _brand_disabled_colors(
    theme: ResolvedTheme, variant: str
) -> tuple[str, str, str]:
    """Return ``(background, foreground, border)`` for a disabled brand button."""
    if variant == BRAND_PRIMARY:
        bg = with_alpha(theme.accent, 0.38)
        fg = with_alpha(theme.brand_fg, 0.65)
        border = with_alpha(theme.accent, 0.45)
    elif variant == BRAND_SUCCESS:
        bg = with_alpha(theme.success, 0.38)
        fg = with_alpha(theme.brand_fg, 0.65)
        border = with_alpha(theme.success, 0.45)
    elif variant == BRAND_DANGER:
        bg = with_alpha(theme.error, 0.38)
        fg = with_alpha(theme.brand_fg, 0.65)
        border = with_alpha(theme.error, 0.45)
    elif variant == BRAND_CAUTION:
        bg = with_alpha(theme.warning, 0.38)
        fg = with_alpha(theme.brand_fg, 0.65)
        border = with_alpha(theme.warning, 0.45)
    elif variant == BRAND_SECONDARY:
        border = theme.border_subtle if theme.is_dark else theme.border
        bg = with_alpha("#000000", 0.12) if theme.is_dark else with_alpha(theme.surface_elevated, 0.85)
        fg = with_alpha(theme.text_primary, 0.55)
        border = with_alpha(border, 0.55)
    else:
        bg = theme.brand_disabled_bg
        fg = theme.brand_disabled_fg
        border = with_alpha(theme.border, 0.35)
    return bg, fg, border


def _brand_disabled_qss(theme: ResolvedTheme, variant: str) -> str:
    """Muted variant tint when disabled — keeps semantic color visible (not flat gray)."""
    bg, fg, border = _brand_disabled_colors(theme, variant)
    return (
        f"background-color: {bg} !important;"
        f" color: {fg} !important;"
        f" border: 1px solid {border} !important;"
    )


def _brand_button_selector(button: QPushButton) -> str:
    """Prefer #objectName selectors — they beat app-level QPushButton rules reliably."""
    object_name = button.objectName()
    if object_name:
        return f"QPushButton#{object_name}"
    return "QPushButton"


def _brand_class_tag(variant: str) -> str:
    """Only primary carries PrimaryActionButton (app-level disabled gray targets it)."""
    if variant == BRAND_PRIMARY:
        return "PrimaryActionButton BrandPrimaryButton"
    if variant == BRAND_SUCCESS:
        return "BrandSuccessButton"
    if variant == BRAND_CAUTION:
        return "BrandCautionButton"
    if variant == BRAND_SECONDARY:
        return "BrandSecondaryButton"
    return "BrandDangerButton"


def _scoped_brand_qss(selector: str, template: str) -> str:
    return template.replace("QPushButton", selector)


def _semantic_button_qss(
    theme: ResolvedTheme,
    *,
    variant: str,
    selector: str,
    base: str,
    hover: str,
    pressed: str,
    border: str,
) -> str:
    fg = brand_label_color(variant, theme)
    disabled = _brand_disabled_qss(theme, variant)
    return _scoped_brand_qss(
        selector,
        f"""
        QPushButton {{
            background-color: {base} !important;
            color: {fg} !important;
            border: 1px solid {border} !important;
            border-radius: 6px;
            padding: 10px 20px;
            font-weight: 700;
        }}
        QPushButton:hover {{
            background-color: {hover} !important;
            border: 1px solid {hover} !important;
        }}
        QPushButton:pressed {{
            background-color: {pressed} !important;
            border: 1px solid {pressed} !important;
        }}
        QPushButton:disabled {{
            {disabled}
        }}
    """,
    )


def brand_qss_for_variant(
    variant: str,
    theme: ResolvedTheme,
    *,
    selector: str = "QPushButton",
) -> str:
    if variant == BRAND_PRIMARY:
        disabled = _brand_disabled_qss(theme, variant)
        return _scoped_brand_qss(
            selector,
            f"""
        QPushButton {{
            background-color: {theme.accent} !important;
            color: {theme.brand_fg} !important;
            border: 1px solid {theme.accent} !important;
            border-radius: 6px;
            padding: 10px 20px;
            font-weight: 700;
        }}
        QPushButton:hover {{
            background-color: {theme.accent_hover} !important;
            border: 1px solid {theme.accent_hover} !important;
        }}
        QPushButton:pressed {{
            background-color: {theme.accent_pressed} !important;
            border: 1px solid {theme.accent_pressed} !important;
        }}
        QPushButton:disabled {{
            {disabled}
        }}
    """,
        )
    if variant == BRAND_SUCCESS:
        return _semantic_button_qss(
            theme,
            variant=variant,
            selector=selector,
            base=theme.success,
            hover=adjust_lightness(theme.success, -0.08),
            pressed=adjust_lightness(theme.success, -0.16),
            border=adjust_lightness(theme.success, -0.12),
        )
    if variant == BRAND_DANGER:
        return _semantic_button_qss(
            theme,
            variant=variant,
            selector=selector,
            base=theme.error,
            hover=adjust_lightness(theme.error, -0.08),
            pressed=adjust_lightness(theme.error, -0.16),
            border=adjust_lightness(theme.error, -0.12),
        )
    if variant == BRAND_CAUTION:
        return _semantic_button_qss(
            theme,
            variant=variant,
            selector=selector,
            base=theme.warning,
            hover=adjust_lightness(theme.warning, -0.08),
            pressed=adjust_lightness(theme.warning, -0.16),
            border=adjust_lightness(theme.warning, -0.12),
        )
    if variant == BRAND_SECONDARY:
        border = theme.border_subtle if theme.is_dark else theme.border
        bg = with_alpha("#000000", 0.2) if theme.is_dark else theme.surface_elevated
        hover = theme.surface_hover
        pressed = theme.surface_pressed
        fg = brand_label_color(BRAND_SECONDARY, theme)
        disabled = _brand_disabled_qss(theme, variant)
        return _scoped_brand_qss(
            selector,
            f"""
        QPushButton {{
            background-color: {bg} !important;
            color: {fg} !important;
            border: 1px solid {border} !important;
            border-radius: 6px;
            padding: 10px 20px;
            font-weight: 700;
        }}
        QPushButton:hover {{
            background-color: {hover} !important;
            border: 1px solid {border} !important;
        }}
        QPushButton:pressed {{
            background-color: {pressed} !important;
            border: 1px solid {border} !important;
        }}
        QPushButton:disabled {{
            {disabled}
        }}
    """,
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
    selector = _brand_button_selector(button)
    qss = brand_qss_for_variant(variant, resolved, selector=selector)

    button.setProperty("class", _brand_class_tag(variant))
    button.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    button.setStyleSheet(qss)

    style = button.style()
    if style is not None:
        style.unpolish(button)
        style.polish(button)

    if icon_name is not None:
        fg = brand_label_color(variant, resolved)
        disabled_fg = brand_label_color(variant, resolved, disabled=True)
        button.setIconSize(QSize(_BRAND_ICON_SIZE, _BRAND_ICON_SIZE))
        button.setIcon(_brand_button_icon(icon_name, fg, disabled_fg))

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


def apply_brand_secondary(
    button: QPushButton,
    icon_name: Optional[str] = None,
    *,
    theme: ResolvedTheme | None = None,
    is_dark: bool | None = None,
) -> None:
    apply_brand_style(
        button, BRAND_SECONDARY, icon_name=icon_name, theme=theme, is_dark=is_dark
    )
