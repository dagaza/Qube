"""Runtime platform tier detection for the desktop companion overlay."""

from __future__ import annotations

import os
import sys
from enum import Enum


class CompanionPlatformTier(str, Enum):
    """Capability tier for floating companion overlays."""

    FULL = "full"
    LIMITED = "limited"
    DEGRADED = "degraded"
    NONE = "none"


def _session_type() -> str:
    if sys.platform.startswith("linux"):
        return os.environ.get("XDG_SESSION_TYPE", "").lower()
    return ""


def detect_companion_platform_tier() -> CompanionPlatformTier:
    """Best-effort runtime tier for companion overlay support."""
    if os.environ.get("QUBE_COMPANION_FORCE_TIER"):
        raw = os.environ.get("QUBE_COMPANION_FORCE_TIER", "").strip().lower()
        for tier in CompanionPlatformTier:
            if tier.value == raw:
                return tier

    if sys.platform == "win32":
        return CompanionPlatformTier.FULL
    if sys.platform == "darwin":
        return CompanionPlatformTier.FULL

    if sys.platform.startswith("linux"):
        session = _session_type()
        if session == "wayland":
            return CompanionPlatformTier.DEGRADED
        if session in ("x11", "xcb", ""):
            return CompanionPlatformTier.LIMITED
        return CompanionPlatformTier.DEGRADED

    return CompanionPlatformTier.LIMITED


def default_companion_enabled_for_tier(tier: CompanionPlatformTier) -> bool:
    """Conservative default: opt-in everywhere; Wayland starts disabled."""
    if tier == CompanionPlatformTier.DEGRADED:
        return False
    return False


def tier_display_name(tier: CompanionPlatformTier) -> str:
    names = {
        CompanionPlatformTier.FULL: "Full overlay support",
        CompanionPlatformTier.LIMITED: "Limited overlay support (X11)",
        CompanionPlatformTier.DEGRADED: "Degraded (Wayland — dock/tray recommended)",
        CompanionPlatformTier.NONE: "Unavailable",
    }
    return names.get(tier, tier.value)
