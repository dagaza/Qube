"""QTextDocument default stylesheets and application-wide hyperlink colors for Qt widgets."""

from __future__ import annotations

from PyQt6.QtGui import QColor, QGuiApplication, QPalette

# App-wide: unvisited vs visited (used in QTextDocument CSS and QPalette Link roles)
LINK_COLOR_UNVISITED = "#3b82f6"
LINK_COLOR_VISITED = "#8b5cf6"


def link_anchor_css() -> str:
    """CSS fragment for <a> in QTextDocument.setDefaultStyleSheet (Markdown → HTML)."""
    u, v = LINK_COLOR_UNVISITED, LINK_COLOR_VISITED
    # Put a:visited after generic `a` so visited wins when supported.
    return (
        f"a:link {{ color: {u}; text-decoration: none; }}"
        f"a {{ color: {u}; text-decoration: none; }}"
        f"a:visited {{ color: {v}; text-decoration: none; }}"
    )


def markdown_document_stylesheet(is_dark: bool) -> str:
    """
    QTextDocument default stylesheet for Markdown → HTML. Must be set *before* setMarkdown()
    so nested elements inherit foreground (see AgentMessageLabel).
    """
    fg = "#cdd6f4" if is_dark else "#1e293b"
    border = "#585b70" if is_dark else "#cbd5e1"
    code_bg = "#313244" if is_dark else "#f1f5f9"
    return (
        f"body, p, span, div, li, ul, ol, dd, dt, "
        f"table, thead, tbody, tr, th, td, "
        f"blockquote, pre, code, "
        f"h1, h2, h3, h4, h5, h6, strong, em {{ color: {fg}; }}"
        + link_anchor_css()
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


def apply_app_link_palette(app: QGuiApplication | None = None) -> None:
    """Set QPalette Link / LinkVisited so QLabel rich text and other palette-driven links match."""
    app = app or QGuiApplication.instance()
    if app is None:
        return
    pal = app.palette()
    pal.setColor(QPalette.ColorRole.Link, QColor(LINK_COLOR_UNVISITED))
    pal.setColor(QPalette.ColorRole.LinkVisited, QColor(LINK_COLOR_VISITED))
    app.setPalette(pal)
