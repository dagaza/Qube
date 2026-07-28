"""Shared QSS for A− / A+ readability controls (Conversations + Library preview toolbars)."""

from __future__ import annotations

from core.theme.accessors import theme_for
from core.theme.tokens import ResolvedTheme
from core.theme.widget_styles import READABILITY_FONT_PAIR


def readability_font_pair_stylesheet(
    *,
    is_dark: bool | None = None,
    theme: ResolvedTheme | None = None,
    button_px: int = 30,
) -> str:
    """Theme-stable stylesheet for the font size pair; not coupled to LLM/TTS state."""
    resolved = theme_for(is_dark=is_dark if is_dark is not None else True, resolved=theme)
    return resolved.style(READABILITY_FONT_PAIR, button_px=button_px)
