"""Color parsing and manipulation for theme derivation."""

from __future__ import annotations

import re
from typing import NamedTuple


class RGBA(NamedTuple):
    r: int
    g: int
    b: int
    a: int = 255

    def to_hex(self) -> str:
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"

    def to_rgba(self) -> str:
        alpha = round(self.a / 255, 2)
        if alpha >= 1.0:
            return self.to_hex()
        return f"rgba({self.r},{self.g},{self.b},{alpha})"


_HEX_RE = re.compile(r"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")
_RGBA_RE = re.compile(
    r"^rgba\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([0-9.]+)\s*\)$",
    re.IGNORECASE,
)


def parse_color(value: str) -> RGBA:
    raw = str(value or "").strip()
    if _RGBA_RE.match(raw):
        r, g, b, a = _RGBA_RE.match(raw).groups()
        return RGBA(int(r), int(g), int(b), int(round(float(a) * 255)))
    if _HEX_RE.match(raw):
        hex_body = raw[1:]
        if len(hex_body) == 3:
            hex_body = "".join(ch * 2 for ch in hex_body)
        if len(hex_body) == 6:
            return RGBA(
                int(hex_body[0:2], 16),
                int(hex_body[2:4], 16),
                int(hex_body[4:6], 16),
            )
        return RGBA(
            int(hex_body[0:2], 16),
            int(hex_body[2:4], 16),
            int(hex_body[4:6], 16),
            int(hex_body[6:8], 16),
        )
    raise ValueError(f"Unsupported color format: {value!r}")


def _rgb_to_hsl(r: int, g: int, b: int) -> tuple[float, float, float]:
    r_f, g_f, b_f = r / 255.0, g / 255.0, b / 255.0
    mx = max(r_f, g_f, b_f)
    mn = min(r_f, g_f, b_f)
    lightness = (mx + mn) / 2.0
    if mx == mn:
        return 0.0, 0.0, lightness
    delta = mx - mn
    saturation = delta / (2.0 - mx - mn) if lightness > 0.5 else delta / (mx + mn)
    if mx == r_f:
        hue = ((g_f - b_f) / delta) % 6.0
    elif mx == g_f:
        hue = ((b_f - r_f) / delta) + 2.0
    else:
        hue = ((r_f - g_f) / delta) + 4.0
    return hue / 6.0, saturation, lightness


def _hsl_to_rgb(h: float, s: float, l: float) -> tuple[int, int, int]:
    if s == 0.0:
        v = int(round(l * 255))
        return v, v, v

    def _hue_to_rgb(p: float, q: float, t: float) -> float:
        if t < 0.0:
            t += 1.0
        if t > 1.0:
            t -= 1.0
        if t < 1.0 / 6.0:
            return p + (q - p) * 6.0 * t
        if t < 1.0 / 2.0:
            return q
        if t < 2.0 / 3.0:
            return p + (q - p) * (2.0 / 3.0 - t) * 6.0
        return p

    q = l * (1.0 + s) if l < 0.5 else l + s - l * s
    p = 2.0 * l - q
    r = _hue_to_rgb(p, q, h + 1.0 / 3.0)
    g = _hue_to_rgb(p, q, h)
    b = _hue_to_rgb(p, q, h - 1.0 / 3.0)
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))


def adjust_lightness(value: str, delta: float) -> str:
    """Adjust HSL lightness by ``delta`` in [-1, 1]. Preserves alpha for rgba inputs."""
    rgba = parse_color(value)
    h, s, l = _rgb_to_hsl(rgba.r, rgba.g, rgba.b)
    l = max(0.0, min(1.0, l + delta))
    r, g, b = _hsl_to_rgb(h, s, l)
    return RGBA(r, g, b, rgba.a).to_rgba()


def with_alpha(value: str, alpha: float) -> str:
    rgba = parse_color(value)
    return RGBA(rgba.r, rgba.g, rgba.b, int(round(max(0.0, min(1.0, alpha)) * 255))).to_rgba()


def rgba_tuple(value: str) -> tuple[int, int, int, int]:
    """Return ``(r, g, b, a)`` with alpha 0–255 for Qt / pyqtgraph APIs."""
    rgba = parse_color(value)
    return rgba.r, rgba.g, rgba.b, rgba.a


def theme_qcolor(value: str):
    """Build ``QColor`` from a theme token string (``#hex`` or ``rgba(...)`` CSS)."""
    from PyQt6.QtGui import QColor

    rgba = parse_color(value)
    return QColor(rgba.r, rgba.g, rgba.b, rgba.a)


def relative_luminance(value: str) -> float:
    rgba = parse_color(value)

    def _channel(c: int) -> float:
        s = c / 255.0
        return s / 12.92 if s <= 0.03928 else ((s + 0.055) / 1.055) ** 2.4

    r, g, b = _channel(rgba.r), _channel(rgba.g), _channel(rgba.b)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(foreground: str, background: str) -> float:
    l1 = relative_luminance(foreground)
    l2 = relative_luminance(background)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def adjust_text_for_contrast(
    text: str,
    background: str,
    *,
    target: float = 4.5,
    max_steps: int = 48,
) -> str:
    """Nudge ``text`` lightness until contrast against ``background`` reaches ``target``."""
    if contrast_ratio(text, background) >= target:
        return text
    direction = 0.04 if relative_luminance(background) < 0.45 else -0.04
    current = text
    for _ in range(max_steps):
        if contrast_ratio(current, background) >= target:
            return current
        current = adjust_lightness(current, direction)
    return current


def contrasting_label_color(
    background: str,
    *,
    dark: str = "#11111b",
    light: str = "#f8fafc",
) -> str:
    """Pick readable label text on top of ``background``."""
    if contrast_ratio(dark, background) >= contrast_ratio(light, background):
        return dark
    return light
