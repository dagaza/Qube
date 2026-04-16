"""Prestige brand-action button styling helper.

Why this exists
---------------
The app's reusable "primary / brand" action buttons (Wakeword Test Lab,
Use Selected, Delete, Retest, Apply, Model Manager Download, etc.) were
originally styled through a dynamic `class` property and matching
`QPushButton[class~="..."]` rules in `assets/styles/base.qss` +
`assets/styles/light.qss`. In practice that approach produced subtle but
persistent rendering bugs on some Qt builds / platform styles — most
visibly in light theme, where the generic `QPushButton` rule leaked
through and left buttons rendered as white-on-off-white even with
`!important` on the attribute-selector Brand rules. The specificity of
dynamic-property selectors and Qt's honoring of `!important` is not
consistently enforced by every QStyle implementation, and adding more
`!important` or higher-specificity app-level rules did not reliably fix
it.

Widget-level `setStyleSheet(...)` always wins: it has the highest
specificity of any QSS source in Qt, is re-polished automatically on
every repaint, and cannot be overridden by the app-level sheet or the
widget's QPalette. That is exactly the guarantee we need for buttons
whose accent colors are theme-independent by design (brand purple, brand
red, brand green are the same in dark + light themes).

This module centralizes the three canonical brand styles so every
call-site uses the same copy of the QSS. Callers pass a
`QPushButton`, pick a variant, and get consistent bg / border / hover
/ disabled rendering regardless of the active theme or platform style.
"""

from __future__ import annotations

from typing import Optional

import qtawesome as qta
from PyQt6.QtWidgets import QPushButton


BRAND_PRIMARY = "primary"
BRAND_SUCCESS = "success"
BRAND_DANGER = "danger"

# Canonical foreground (text / icon) color per brand variant. All three share
# the same near-white so the icon/text pair renders consistently; expose it as
# a mapping so callers who set icons separately (qtawesome, custom pixmaps)
# can read the correct tint without hard-coding `#f8fafc` across views.
BRAND_FG_COLOR: dict[str, str] = {
    BRAND_PRIMARY: "#f8fafc",
    BRAND_SUCCESS: "#f8fafc",
    BRAND_DANGER: "#f8fafc",
}

# Canonical disabled-state foreground color (used when the icon tint should
# follow the button into its disabled state). Matches the :disabled color in
# `_BRAND_QSS` below.
BRAND_DISABLED_FG_COLOR = "rgba(148, 163, 184, 0.85)"


def brand_fg_color(variant: str) -> str:
    """Return the canonical text/icon color for a brand variant."""
    try:
        return BRAND_FG_COLOR[variant]
    except KeyError as exc:
        raise ValueError(
            f"Unknown brand variant: {variant!r}. "
            f"Expected one of: {sorted(BRAND_FG_COLOR)}"
        ) from exc


_BRAND_QSS: dict[str, str] = {
    BRAND_PRIMARY: """
        QPushButton {
            background-color: #8b5cf6;
            color: #f8fafc;
            border: 1px solid #8b5cf6;
            border-radius: 6px;
            padding: 8px 15px;
            font-weight: 700;
        }
        QPushButton:hover {
            background-color: #7c3aed;
            border: 1px solid #7c3aed;
        }
        QPushButton:pressed {
            background-color: #6d28d9;
            border: 1px solid #6d28d9;
        }
        QPushButton:disabled {
            background-color: rgba(100, 116, 139, 0.22);
            color: rgba(148, 163, 184, 0.85);
            border: 1px solid rgba(148, 163, 184, 0.35);
        }
    """,
    BRAND_SUCCESS: """
        QPushButton {
            background-color: #16a34a;
            color: #f8fafc;
            border: 1px solid #15803d;
            border-radius: 6px;
            padding: 8px 15px;
            font-weight: 700;
        }
        QPushButton:hover {
            background-color: #15803d;
            border: 1px solid #15803d;
        }
        QPushButton:pressed {
            background-color: #166534;
            border: 1px solid #166534;
        }
        QPushButton:disabled {
            background-color: rgba(100, 116, 139, 0.22);
            color: rgba(148, 163, 184, 0.85);
            border: 1px solid rgba(148, 163, 184, 0.35);
        }
    """,
    BRAND_DANGER: """
        QPushButton {
            background-color: #dc2626;
            color: #f8fafc;
            border: 1px solid #b91c1c;
            border-radius: 6px;
            padding: 8px 15px;
            font-weight: 700;
        }
        QPushButton:hover {
            background-color: #b91c1c;
            border: 1px solid #b91c1c;
        }
        QPushButton:pressed {
            background-color: #991b1b;
            border: 1px solid #991b1b;
        }
        QPushButton:disabled {
            background-color: rgba(100, 116, 139, 0.22);
            color: rgba(148, 163, 184, 0.85);
            border: 1px solid rgba(148, 163, 184, 0.35);
        }
    """,
}


def apply_brand_style(
    button: QPushButton,
    variant: str,
    icon_name: Optional[str] = None,
) -> None:
    """Apply the brand style (widget-level QSS) and optionally tint an icon.

    Widget-level QSS is what actually drives the visible render: it has the
    highest specificity of any QSS source in Qt and will defeat any
    app-level `QPushButton { background-color: ... }` rule on every render
    path. The dynamic `class` property is kept in sync for backward
    compatibility with any code that still matches on `class~=` and for
    future migrations.

    If `icon_name` is provided it is rendered through `qtawesome` using the
    variant's canonical foreground color (`BRAND_FG_COLOR[variant]`) so the
    icon always matches the button's text color. Pass `icon_name=None` for
    text-only brand buttons or when the caller manages the icon separately
    (for example when stacking multiple icons via `qta.icon(..., options=...)`
    or switching icons across runtime states with state-specific names — in
    that case, read the tint with `brand_fg_color(variant)` at the call site
    and pass it as the `color` kwarg).
    """
    try:
        qss = _BRAND_QSS[variant]
    except KeyError as exc:
        raise ValueError(
            f"Unknown brand variant: {variant!r}. "
            f"Expected one of: {sorted(_BRAND_QSS)}"
        ) from exc

    if variant == BRAND_PRIMARY:
        class_tag = "PrimaryActionButton BrandPrimaryButton"
    elif variant == BRAND_SUCCESS:
        class_tag = "PrimaryActionButton BrandSuccessButton"
    else:
        class_tag = "PrimaryActionButton BrandDangerButton"

    button.setProperty("class", class_tag)
    button.setStyleSheet(qss)

    if icon_name is not None:
        button.setIcon(
            qta.icon(
                icon_name,
                color=BRAND_FG_COLOR[variant],
                color_disabled=BRAND_DISABLED_FG_COLOR,
            )
        )

    style = button.style()
    if style is not None:
        style.unpolish(button)
        style.polish(button)
    button.update()


def apply_brand_primary(
    button: QPushButton, icon_name: Optional[str] = None
) -> None:
    apply_brand_style(button, BRAND_PRIMARY, icon_name=icon_name)


def apply_brand_success(
    button: QPushButton, icon_name: Optional[str] = None
) -> None:
    apply_brand_style(button, BRAND_SUCCESS, icon_name=icon_name)


def apply_brand_danger(
    button: QPushButton, icon_name: Optional[str] = None
) -> None:
    apply_brand_style(button, BRAND_DANGER, icon_name=icon_name)
