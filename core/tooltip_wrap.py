"""Tooltip helpers (QSS-driven; no runtime monkey-patching)."""

from __future__ import annotations

from typing import Optional

def install_wrapping_tooltips(max_width_px: Optional[int] = None) -> None:
    """
    Backward-compatible no-op.
    Tooltip rendering is fully QSS-driven to avoid rich-text rendering artifacts.
    """
    _ = max_width_px


def apply_tooltip_theme(is_dark: bool) -> None:
    """
    Backward-compatible no-op.
    Tooltip visuals come only from active theme QSS.
    """
    _ = is_dark
