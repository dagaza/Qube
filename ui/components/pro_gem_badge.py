"""Pro-tier gem badge used for paid-feature affordances."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import QLabel, QWidget

from core.theme.view_theme import view_resolved_theme
from core.theme.widget_styles import ACCENT_ICON
from core.theme.svg_icons import themed_fa_icon

PRECISION_INGEST_BADGE_TOOLTIP = (
    "Indexed with precision ingest (Qube Pro). "
    "Semantic chunking for maximum citation accuracy."
)


def pro_tier_gem_color(theme) -> str:
    return theme.color(ACCENT_ICON)


def make_pro_gem_badge(
    parent: QWidget | None,
    *,
    tooltip: str = PRECISION_INGEST_BADGE_TOOLTIP,
    size: int = 14,
) -> QLabel:
    """Small gem icon marking a Pro-indexed Library document."""
    badge = QLabel(parent)
    badge.setObjectName("ProGemBadge")
    badge.setFixedSize(size + 4, size + 4)
    badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
    badge.setToolTip(tooltip)
    badge.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
    apply_pro_gem_badge_theme(badge, parent=parent, size=size, tooltip=tooltip)
    return badge


def apply_pro_gem_badge_theme(
    badge: QLabel,
    *,
    parent: QWidget | None,
    size: int = 14,
    tooltip: str | None = None,
) -> None:
    is_dark = getattr(parent.window() if parent else None, "_is_dark_theme", True)
    theme = view_resolved_theme(parent, is_dark=is_dark)
    color = pro_tier_gem_color(theme)
    badge.setPixmap(
        themed_fa_icon("fa5s.gem", color, size).pixmap(QSize(size, size))
    )
    if tooltip:
        badge.setToolTip(tooltip)
