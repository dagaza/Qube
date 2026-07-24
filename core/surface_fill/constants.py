"""Surface fill identifiers and enumerated values."""

from __future__ import annotations

from typing import Literal

SURFACE_CHAT_TRANSCRIPT = "chat_transcript"
SURFACE_LIBRARY_PREVIEW = "library_preview"

V2_SURFACES: frozenset[str] = frozenset(
    {
        SURFACE_CHAT_TRANSCRIPT,
        SURFACE_LIBRARY_PREVIEW,
    }
)

OverlayStrength = Literal["subtle", "balanced", "vivid"]
OVERLAY_STRENGTHS: frozenset[str] = frozenset({"subtle", "balanced", "vivid"})

GradientDirection = Literal["vertical", "horizontal", "diagonal_down", "diagonal_up"]
GRADIENT_DIRECTIONS: frozenset[str] = frozenset(
    {"vertical", "horizontal", "diagonal_down", "diagonal_up"}
)

WallpaperKind = Literal[
    "none",
    "theme_default",
    "preset",
    "solid",
    "gradient",
    "image",
]
