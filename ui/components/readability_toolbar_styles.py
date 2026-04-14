"""Shared QSS for A− / A+ readability controls (Conversations + Library preview toolbars)."""

from __future__ import annotations


def readability_font_pair_stylesheet(*, is_dark: bool, button_px: int = 30) -> str:
    """Theme-stable stylesheet for the font size pair; not coupled to LLM/TTS state."""
    base_icon_color = "#8b5cf6" if is_dark else "#1e293b"
    hover_bg = "rgba(255, 255, 255, 0.08)" if is_dark else "rgba(0, 0, 0, 0.05)"
    dis = "#6c7086" if is_dark else "#94a3b8"
    return f"""
                QPushButton {{
                    background: transparent;
                    border: none;
                    border-radius: 6px;
                    padding: 2px 4px;
                    color: {base_icon_color};
                    font-weight: 700;
                    font-size: 13px;
                    min-width: {button_px}px;
                    max-width: {button_px}px;
                    min-height: {button_px}px;
                    max-height: {button_px}px;
                }}
                QPushButton:hover {{ background-color: {hover_bg}; }}
                QPushButton:disabled {{ color: {dis}; }}
            """
