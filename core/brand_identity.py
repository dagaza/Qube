"""Fixed Qube brand identity colors — outside user theme customization.

Logo stroke, celebration confetti, and other trademark visuals must not read
``ResolvedTheme`` user overrides (accent/background pickers).
"""

from __future__ import annotations

from PyQt6.QtGui import QColor

BRAND_LOGO_STROKE_HEX = "#8b5cf6"
BRAND_LOGO_STROKE_COLOR = QColor(BRAND_LOGO_STROKE_HEX)

BRAND_CELEBRATION_PALETTE: tuple[str, ...] = (
    "#f9e2af",
    "#fab387",
    "#89b4fa",
    "#cba6f7",
    "#a6e3a1",
    "#f38ba8",
)

# Fixed telemetry legend / sidebar mini-metrics (pre-theme-migration values on ``main``).
BRAND_TELEMETRY_CPU_HEX = "#10b981"
BRAND_TELEMETRY_RAM_HEX = "#3b82f6"
BRAND_TELEMETRY_GPU_HEX = "#8b5cf6"

# Top-bar WEB / HYBRID standby indicators (fixed on ``main``; not scheme warning-derived).
BRAND_WEB_INDICATOR_STANDBY_HEX = "#c2410c"

# Model Manager hub row "Official" publisher label (readable on light + dark hub rows).
BRAND_HUB_OFFICIAL_BADGE_FG_DARK = "#cdd6f4"
BRAND_HUB_OFFICIAL_BADGE_FG_LIGHT = "#1e3a8a"
