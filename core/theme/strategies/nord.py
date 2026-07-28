"""Nord semantic derivation — cooler secondary/link accents."""

from __future__ import annotations

from core.theme.strategies.default import derive_semantic_tokens
from core.theme.tokens import CoreTokenSet, ResolvedTheme, ThemeMode


class NordThemeStrategy:
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
            accent_secondary="#81a1c1",
            link="#88c0d0",
            link_visited="#5e81ac",
            text_muted="#4c566a",
            overlay_pane="rgba(0,0,0,0.18)",
            chat_header="#88c0d0",
            scrollbar_thumb="#434c5e",
            scrollbar_thumb_hover="#4c566a",
        )
