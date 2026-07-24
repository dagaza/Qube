"""QTextDocument default stylesheets and application-wide hyperlink colors for Qt widgets."""

from __future__ import annotations

from PyQt6.QtGui import QColor, QGuiApplication, QPalette

from core.theme.accessors import theme_for
from core.theme.color_utils import theme_qcolor
from core.theme.link_styles import link_anchor_css as _link_anchor_css
from core.theme.tokens import ResolvedTheme, ThemeMode

# Legacy constants — prefer ``ResolvedTheme.link`` / ``link_visited``.
LINK_COLOR_UNVISITED = "#3b82f6"
LINK_COLOR_VISITED = "#8b5cf6"


def link_anchor_css(theme: ResolvedTheme | None = None, *, is_dark: bool = True) -> str:
    resolved = theme_for(is_dark=is_dark, resolved=theme)
    return _link_anchor_css(resolved)


def markdown_document_stylesheet(
    is_dark: bool | None = None,
    *,
    theme: ResolvedTheme | None = None,
) -> str:
    """QTextDocument default stylesheet for Markdown → HTML."""
    resolved = theme_for(
        is_dark=is_dark if is_dark is not None else True,
        resolved=theme,
    )
    fg = resolved.text_primary
    border = resolved.border_subtle if resolved.is_dark else resolved.border
    code_bg = resolved.surface_elevated if resolved.is_dark else "#f1f5f9"
    return (
        f"body, p, span, div, li, ul, ol, dd, dt, "
        f"table, thead, tbody, tr, th, td, "
        f"blockquote, pre, code, "
        f"h1, h2, h3, h4, h5, h6, strong, em {{ color: {fg}; }}"
        + _link_anchor_css(resolved)
        + f"table {{ border-color: {border}; }}"
        + f"th, td {{ border-color: {border}; border-width: 1px; border-style: solid; padding: 4px; }}"
        + f"code, pre {{ background-color: {code_bg}; }}"
        + f"hr {{ border-color: {border}; color: {border}; }}"
        + "h1 { font-size: 1.35em; font-weight: 700; margin-top: 0.45em; margin-bottom: 0.2em; }"
        + "h2 { font-size: 1.2em; font-weight: 600; margin-top: 0.4em; margin-bottom: 0.18em; }"
        + "h3 { font-size: 1.1em; font-weight: 600; margin-top: 0.35em; margin-bottom: 0.15em; }"
        + "h4, h5, h6 { font-size: 1.05em; font-weight: 600; margin-top: 0.3em; margin-bottom: 0.12em; }"
        + "div.hub-readme { margin: 0; }"
        + "div.hub-readme p { margin-top: 0.35em; margin-bottom: 0.35em; }"
        + "div.hub-readme h1, div.hub-readme h2, div.hub-readme h3 "
        + "{ margin-top: 0.6em; margin-bottom: 0.35em; }"
    )


def apply_app_link_palette(
    app: QGuiApplication | None = None,
    *,
    theme: ResolvedTheme | None = None,
    is_dark: bool | None = None,
) -> None:
    """Set QPalette Link / LinkVisited from the active theme."""
    app = app or QGuiApplication.instance()
    if app is None:
        return
    resolved = theme_for(
        is_dark=is_dark if is_dark is not None else True,
        resolved=theme,
    )
    pal = app.palette()
    pal.setColor(QPalette.ColorRole.Link, theme_qcolor(resolved.link))
    pal.setColor(QPalette.ColorRole.LinkVisited, theme_qcolor(resolved.link_visited))
    app.setPalette(pal)


def link_colors_for_mode(mode: ThemeMode) -> tuple[str, str]:
    resolved = theme_for(is_dark=mode.is_dark)
    return resolved.link, resolved.link_visited
