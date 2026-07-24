"""Helpers for resolving ``ResolvedTheme`` in widget code during migration."""

from __future__ import annotations

from core.theme.resolver import ThemeResolver
from core.theme.schemes import BUILTIN_SCHEMES, default_scheme_id_for_mode
from core.theme.tokens import ResolvedTheme, ThemeMode


def theme_for(
    *,
    is_dark: bool | None = None,
    resolved: ResolvedTheme | None = None,
    scheme_id: str | None = None,
) -> ResolvedTheme:
    """Return ``resolved`` when provided, else built-in scheme for ``is_dark``."""
    if resolved is not None:
        return resolved
    mode = ThemeMode.DARK if is_dark else ThemeMode.LIGHT
    sid = scheme_id or default_scheme_id_for_mode(mode.value)
    return ThemeResolver(BUILTIN_SCHEMES).resolve(mode=mode, scheme_id=sid)
