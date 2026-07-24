"""Theme token models — core primitives and resolved semantic outputs."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Any


class ThemeMode(str, Enum):
    DARK = "dark"
    LIGHT = "light"

    @property
    def is_dark(self) -> bool:
        return self is ThemeMode.DARK


CORE_TOKEN_KEYS: tuple[str, ...] = (
    "background",
    "surface",
    "sidebar_surface",
    "surface_elevated",
    "text_primary",
    "text_secondary",
    "border",
    "accent",
    "success",
    "warning",
    "error",
    "info",
)


@dataclass(frozen=True)
class CoreTokenSet:
    background: str
    surface: str
    sidebar_surface: str
    surface_elevated: str
    text_primary: str
    text_secondary: str
    border: str
    accent: str
    success: str
    warning: str
    error: str
    info: str

    def as_dict(self) -> dict[str, str]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> CoreTokenSet:
        missing = [key for key in CORE_TOKEN_KEYS if key not in values]
        if missing:
            raise ValueError(f"Missing core tokens: {', '.join(missing)}")
        extra = [key for key in values if key not in CORE_TOKEN_KEYS]
        if extra:
            raise ValueError(f"Unknown core tokens: {', '.join(extra)}")
        return cls(**{key: str(values[key]) for key in CORE_TOKEN_KEYS})


@dataclass(frozen=True)
class ResolvedTheme:
    """Fully derived theme: core primitives plus semantic outputs."""

    scheme_id: str
    scheme_name: str
    mode: ThemeMode
    algorithm: str

    # Core primitives (editable by users)
    background: str
    surface: str
    sidebar_surface: str
    surface_elevated: str
    text_primary: str
    text_secondary: str
    border: str
    accent: str
    success: str
    warning: str
    error: str
    info: str

    # Semantic outputs (derived only)
    accent_hover: str
    accent_pressed: str
    accent_muted_bg: str
    accent_secondary: str
    selection: str
    selection_border: str
    selection_bg: str
    link: str
    link_visited: str
    surface_hover: str
    surface_pressed: str
    surface_selected: str
    border_subtle: str
    overlay_pane: str
    modal_scrim: str
    text_muted: str
    text_on_accent: str
    text_on_surface_elevated: str
    scrollbar_thumb: str
    scrollbar_thumb_hover: str
    tooltip_bg: str
    tooltip_border: str
    chat_user_bubble: str
    chat_user_text: str
    chat_agent_text: str
    chat_header: str
    brand_fg: str
    brand_disabled_bg: str
    brand_disabled_fg: str
    list_row_title_selected: str

    @property
    def is_dark(self) -> bool:
        return self.mode.is_dark

    def core_tokens(self) -> CoreTokenSet:
        return CoreTokenSet(
            background=self.background,
            surface=self.surface,
            sidebar_surface=self.sidebar_surface,
            surface_elevated=self.surface_elevated,
            text_primary=self.text_primary,
            text_secondary=self.text_secondary,
            border=self.border,
            accent=self.accent,
            success=self.success,
            warning=self.warning,
            error=self.error,
            info=self.info,
        )

    def style(self, role: str, **kwargs) -> str:
        """Return QSS for a named widget style role."""
        from core.theme.widget_styles import theme_style

        return theme_style(self, role, **kwargs)

    def color(self, role: str) -> str:
        """Return a single color token for icon tinting etc."""
        from core.theme.widget_styles import theme_color

        return theme_color(self, role)

    def qcolor(self, color: str):
        """Return ``QColor`` parsed from a theme color string."""
        from core.theme.color_utils import theme_qcolor

        return theme_qcolor(color)

    def qcolor_role(self, role: str):
        """Return ``QColor`` for a named color role."""
        return self.qcolor(self.color(role))

    def apply_style(self, widget, role: str, **kwargs) -> None:
        """Apply ``style(role)`` to a widget."""
        widget.setStyleSheet(self.style(role, **kwargs))
