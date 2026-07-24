"""Default semantic token derivation (generic HSL adjustments)."""

from __future__ import annotations

from core.theme.color_utils import adjust_lightness, with_alpha
from core.theme.tokens import CoreTokenSet, ResolvedTheme, ThemeMode


def derive_semantic_tokens(
    core: CoreTokenSet,
    *,
    scheme_id: str,
    scheme_name: str,
    mode: ThemeMode,
    algorithm: str,
    accent_secondary: str | None = None,
    link: str | None = None,
    link_visited: str | None = None,
    text_muted: str | None = None,
    overlay_pane: str | None = None,
    modal_scrim: str | None = None,
    chat_user_bubble: str | None = None,
    chat_user_text: str | None = None,
    chat_header: str | None = None,
    scrollbar_thumb: str | None = None,
    scrollbar_thumb_hover: str | None = None,
    list_row_title_selected: str | None = None,
) -> ResolvedTheme:
    is_dark = mode.is_dark
    accent_secondary = accent_secondary or (
        "#89b4fa" if is_dark else "#3b82f6"
    )
    link = link or "#3b82f6"
    link_visited = link_visited or core.accent
    text_muted = text_muted or ("#6c7086" if is_dark else "#64748b")
    overlay_pane = overlay_pane or (
        "rgba(0,0,0,0.15)" if is_dark else "rgba(241,245,249,0.9)"
    )
    modal_scrim = modal_scrim or (
        "rgba(0,0,0,175)" if is_dark else "rgba(30,41,59,110)"
    )
    chat_user_bubble = chat_user_bubble or ("#89b4fa" if is_dark else "#0f172a")
    chat_user_text = chat_user_text or ("#11111b" if is_dark else "#ffffff")
    chat_header = chat_header or accent_secondary
    scrollbar_thumb = scrollbar_thumb or ("#45475a" if is_dark else "#cbd5e1")
    scrollbar_thumb_hover = scrollbar_thumb_hover or (
        "#585b70" if is_dark else "#94a3b8"
    )
    list_row_title_selected = list_row_title_selected or (
        "#ffffff" if is_dark else core.text_primary
    )

    surface_hover = adjust_lightness(core.surface, 0.04 if is_dark else -0.02)
    surface_pressed = adjust_lightness(core.surface, 0.07 if is_dark else -0.04)
    surface_selected = adjust_lightness(core.surface, 0.05 if is_dark else -0.03)
    border_subtle = (
        with_alpha(core.text_primary, 0.05)
        if is_dark
        else adjust_lightness(core.border, 0.06)
    )

    return ResolvedTheme(
        scheme_id=scheme_id,
        scheme_name=scheme_name,
        mode=mode,
        algorithm=algorithm,
        background=core.background,
        surface=core.surface,
        sidebar_surface=core.sidebar_surface,
        surface_elevated=core.surface_elevated,
        text_primary=core.text_primary,
        text_secondary=core.text_secondary,
        border=core.border,
        accent=core.accent,
        success=core.success,
        warning=core.warning,
        error=core.error,
        info=core.info,
        accent_hover=adjust_lightness(core.accent, -0.08),
        accent_pressed=adjust_lightness(core.accent, -0.16),
        accent_muted_bg=with_alpha(core.accent, 0.22),
        accent_secondary=accent_secondary,
        selection=core.accent,
        selection_border=core.accent,
        selection_bg=with_alpha(core.accent, 0.22),
        link=link,
        link_visited=link_visited,
        surface_hover=surface_hover,
        surface_pressed=surface_pressed,
        surface_selected=surface_selected,
        border_subtle=border_subtle,
        overlay_pane=overlay_pane,
        modal_scrim=modal_scrim,
        text_muted=text_muted,
        text_on_accent="#f8fafc",
        text_on_surface_elevated=core.text_primary,
        scrollbar_thumb=scrollbar_thumb,
        scrollbar_thumb_hover=scrollbar_thumb_hover,
        tooltip_bg=core.background if is_dark else core.surface_elevated,
        tooltip_border=accent_secondary if is_dark else core.border,
        chat_user_bubble=chat_user_bubble,
        chat_user_text=chat_user_text,
        chat_agent_text=core.text_primary,
        chat_header=chat_header,
        brand_fg="#f8fafc",
        brand_disabled_bg="rgba(100,116,139,0.22)",
        brand_disabled_fg="rgba(148,163,184,0.85)",
        list_row_title_selected=list_row_title_selected,
    )


class DefaultThemeStrategy:
    def derive(
        self,
        core: CoreTokenSet,
        *,
        scheme_id: str,
        scheme_name: str,
        mode: ThemeMode,
        algorithm: str,
    ) -> ResolvedTheme:
        return derive_semantic_tokens(
            core,
            scheme_id=scheme_id,
            scheme_name=scheme_name,
            mode=mode,
            algorithm=algorithm,
        )
