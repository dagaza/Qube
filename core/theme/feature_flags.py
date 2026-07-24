"""Theme system feature flags."""

from __future__ import annotations

import os


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def is_static_theme_enabled() -> bool:
    """When true, load ``base.qss`` / ``light.qss`` verbatim instead of rendering."""
    return _env_truthy("QUBE_STATIC_THEME")


def is_generated_theme_enabled() -> bool:
    """When true, ``ThemeApplicator`` renders QSS from ``ResolvedTheme`` tokens.

    Generated themes are the default (Phase 3+). Set ``QUBE_STATIC_THEME=1`` to
    force the legacy static stylesheet path.
    """
    if is_static_theme_enabled():
        return False
    explicit = os.environ.get("QUBE_GENERATED_THEME", "").strip().lower()
    if explicit in ("0", "false", "no", "off"):
        return False
    if explicit in ("1", "true", "yes", "on"):
        return True
    return True
