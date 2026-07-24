"""Catppuccin semantic derivation — Mocha dark overlays; Latte uses light defaults."""

from __future__ import annotations

from core.theme.strategies.default import DefaultThemeStrategy, derive_semantic_tokens
from core.theme.tokens import CoreTokenSet, ResolvedTheme, ThemeMode


class CatppuccinThemeStrategy(DefaultThemeStrategy):
    def derive(
        self,
        core: CoreTokenSet,
        *,
        scheme_id: str,
        scheme_name: str,
        mode: ThemeMode,
        algorithm: str,
    ) -> ResolvedTheme:
        if mode.is_dark:
            return derive_semantic_tokens(
                core,
                scheme_id=scheme_id,
                scheme_name=scheme_name,
                mode=mode,
                algorithm=algorithm,
                accent_secondary="#89b4fa",
                text_muted="rgba(205,214,244,0.5)",
                overlay_pane="rgba(0,0,0,0.15)",
                modal_scrim="rgba(0,0,0,175)",
                chat_user_bubble="#89b4fa",
                chat_user_text="#11111b",
                chat_header="#8b5cf6",
                scrollbar_thumb="#45475a",
                scrollbar_thumb_hover="#585b70",
            )
        return derive_semantic_tokens(
            core,
            scheme_id=scheme_id,
            scheme_name=scheme_name,
            mode=mode,
            algorithm=algorithm,
            chat_user_bubble="#89b4fa",
            chat_user_text="#11111b",
            chat_header=core.accent,
        )
