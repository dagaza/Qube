"""Theme helpers for canonical trace diff debugger UI."""

from __future__ import annotations

from PyQt6.QtGui import QColor

from core.theme.accessors import theme_for
from core.theme.color_utils import adjust_lightness, parse_color, with_alpha
from core.theme.tokens import ResolvedTheme

DiffStatus = str  # match | modified | missing | extra
RailLevel = str  # REQUEST | PROMPT | OUTPUT
CollapseRisk = str  # LOW | MEDIUM | HIGH


def resolve_trace_diff_theme(
    *,
    is_dark: bool = True,
    theme: ResolvedTheme | None = None,
) -> ResolvedTheme:
    return theme if theme is not None else theme_for(is_dark=is_dark)


def trace_diff_html_stylesheet(theme: ResolvedTheme) -> str:
    """Inline CSS for word/sentence diff HTML rendered in QTextBrowser."""
    t = theme
    return f"""
.diff-match {{ color: {t.success}; }}
.diff-mod {{ color: {t.warning}; background: {with_alpha(t.warning, 0.12)}; }}
.diff-miss {{ color: {t.error}; background: {with_alpha(t.error, 0.15)}; }}
.diff-extra {{ color: {t.link}; background: {with_alpha(t.link, 0.12)}; }}
.diff-truncated {{ color: {t.text_muted}; font-style: italic; }}
.divergence-marker {{ color: {t.warning}; font-weight: 600; }}
"""


def trace_diff_status_colors(theme: ResolvedTheme) -> dict[DiffStatus, str]:
    """Darker tints for tree row backgrounds keyed by diff status."""
    t = theme
    delta = -0.12 if t.is_dark else -0.08
    return {
        "match": adjust_lightness(t.success, delta),
        "modified": adjust_lightness(t.warning, delta),
        "missing": adjust_lightness(t.error, delta),
        "extra": adjust_lightness(t.link, delta),
    }


def trace_diff_status_fallback(theme: ResolvedTheme) -> str:
    t = theme
    return adjust_lightness(t.text_muted, -0.05 if t.is_dark else 0.0)


def trace_diff_qt_color(color: str, alpha: float) -> QColor:
    rgba = parse_color(color)
    qt = QColor(rgba.r, rgba.g, rgba.b)
    qt.setAlphaF(alpha)
    return qt


def trace_diff_window_stylesheet(theme: ResolvedTheme) -> str:
    border = with_alpha(theme.text_primary, 0.08 if theme.is_dark else 0.12)
    surface_bg = with_alpha(theme.surface, 0.2 if theme.is_dark else 0.5)
    summary_bg = with_alpha(theme.text_primary, 0.04 if theme.is_dark else 0.06)
    return f"""
#CanonicalTraceDiffWindow {{
    background-color: {theme.background};
}}
#CanonicalTraceDiffSurface {{
    background-color: {surface_bg};
    border: 1px solid {border};
    border-radius: 12px;
}}
#CanonicalTraceDiffSummary {{
    background-color: {summary_bg};
    border: 1px solid {border};
    border-radius: 8px;
}}
#CanonicalTraceDiffWindow QLabel,
#CanonicalTraceDiffWindow QGroupBox,
#CanonicalTraceDiffWindow QPushButton {{
    color: {theme.text_primary};
}}
"""


def trace_diff_rail_background(theme: ResolvedTheme) -> QColor:
    rgba = parse_color(theme.background)
    return QColor(rgba.r, rgba.g, rgba.b)


def trace_diff_rail_level_colors(theme: ResolvedTheme) -> dict[RailLevel, QColor]:
    return {
        "REQUEST": trace_diff_qt_color(theme.error, 1.0),
        "PROMPT": trace_diff_qt_color(theme.warning, 1.0),
        "OUTPUT": trace_diff_qt_color(theme.link, 1.0),
    }


def trace_diff_rail_fallback(theme: ResolvedTheme) -> QColor:
    return trace_diff_qt_color(theme.text_muted, 1.0)


def scenario_workflow_surface_stylesheet(theme: ResolvedTheme) -> str:
    return f"""
    QFrame#ScenarioWorkflowSurface {{
        background: {theme.background};
        border: 1px solid {theme.link};
        border-radius: 12px;
        padding: 8px;
    }}
    QLabel {{ color: {theme.text_primary}; background: transparent; }}
    """


def collapse_risk_chip_stylesheet(
    theme: ResolvedTheme,
    risk: CollapseRisk | str,
    *,
    selected: bool,
) -> tuple[str, str, str]:
    """Return frame, title, and subtitle QSS for a collapse turn chip."""
    level = str(risk or "LOW").upper()
    delta = -0.12 if theme.is_dark else -0.08
    if level == "HIGH":
        fg = adjust_lightness(theme.error, delta)
        bg = with_alpha(theme.error, 0.15 if theme.is_dark else 0.12)
    elif level == "MEDIUM":
        fg = adjust_lightness(theme.warning, delta)
        bg = with_alpha(theme.warning, 0.15 if theme.is_dark else 0.12)
    else:
        fg = adjust_lightness(theme.success, delta)
        bg = with_alpha(theme.success, 0.15 if theme.is_dark else 0.12)

    border = theme.link if selected else fg
    frame = (
        f"QFrame#CollapseTurnChip {{"
        f"background: {bg}; color: {fg}; border: 2px solid {border};"
        f"border-radius: 8px; padding: 4px;"
        f"}}"
    )
    label = f"color: {fg};"
    title = label + " font-weight: 700;"
    subtitle = label + " font-size: 11px;"
    return frame, title, subtitle
