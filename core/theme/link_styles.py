"""Hyperlink anchor CSS fragments for QTextDocument stylesheets."""

from __future__ import annotations

from core.theme.tokens import ResolvedTheme


def link_anchor_css(resolved: ResolvedTheme) -> str:
    """Return ``a`` / ``a:link`` / ``a:visited`` rules for the given theme."""
    u, v = resolved.link, resolved.link_visited
    return (
        f"a:link {{ color: {u}; text-decoration: none; }}"
        f"a {{ color: {u}; text-decoration: none; }}"
        f"a:visited {{ color: {v}; text-decoration: none; }}"
    )
