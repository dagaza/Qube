"""Sidecar expression capability tier resolution."""

from __future__ import annotations

import json
import logging
from enum import Enum
from pathlib import Path

from core.companion_cognition.types import ExpressionLevel

logger = logging.getLogger("Qube.CompanionVerbal")


class ExpressionCapabilityTier(str, Enum):
    MICRO = "micro"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


_TIER_MAX_LEVEL: dict[ExpressionCapabilityTier, ExpressionLevel] = {
    ExpressionCapabilityTier.MICRO: ExpressionLevel.TEMPLATE,
    ExpressionCapabilityTier.SMALL: ExpressionLevel.TEMPLATE,
    ExpressionCapabilityTier.MEDIUM: ExpressionLevel.SIDECAR_REWRITE,
    ExpressionCapabilityTier.LARGE: ExpressionLevel.FULL_GENERATE,
}

_TIER_ALLOWS_L2: dict[ExpressionCapabilityTier, bool] = {
    ExpressionCapabilityTier.MICRO: False,
    ExpressionCapabilityTier.SMALL: True,
    ExpressionCapabilityTier.MEDIUM: True,
    ExpressionCapabilityTier.LARGE: True,
}

_TIER_ALLOWS_L3: dict[ExpressionCapabilityTier, bool] = {
    ExpressionCapabilityTier.MICRO: False,
    ExpressionCapabilityTier.SMALL: False,
    ExpressionCapabilityTier.MEDIUM: False,
    ExpressionCapabilityTier.LARGE: True,
}


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def bundled_capability_tiers_path() -> Path:
    return _project_root() / "assets" / "companion" / "capability_tiers.json"


def _load_basename_tiers() -> dict[str, str]:
    path = bundled_capability_tiers_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        mapping = data.get("basename_patterns") or data.get("patterns") or {}
        if isinstance(mapping, dict):
            return {str(k).lower(): str(v).lower() for k, v in mapping.items()}
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("[CompanionCognition] capability tiers load failed: %s", e)
    return {}


def _tier_from_basename(basename: str, patterns: dict[str, str]) -> ExpressionCapabilityTier | None:
    low = basename.lower()
    for pattern, tier in patterns.items():
        if pattern in low:
            try:
                return ExpressionCapabilityTier(tier)
            except ValueError:
                continue
    return None


def _telemetry_downgrade(current: ExpressionCapabilityTier) -> ExpressionCapabilityTier:
    try:
        from core.sidecar_telemetry import get_sidecar_telemetry_brain

        brain = get_sidecar_telemetry_brain()
        stats = brain.companion_line_stats(window=20)
        total = stats.get("total", 0)
        if total < 5:
            return current
        ok_rate = stats.get("ok_rate", 1.0)
        if float(ok_rate) < 0.6:
            order = [
                ExpressionCapabilityTier.LARGE,
                ExpressionCapabilityTier.MEDIUM,
                ExpressionCapabilityTier.SMALL,
                ExpressionCapabilityTier.MICRO,
            ]
            idx = order.index(current)
            if idx < len(order) - 1:
                return order[idx + 1]
    except Exception:
        pass
    return current


def _freedom_override_tier() -> ExpressionCapabilityTier | None:
    from core import app_settings

    mode = app_settings.get_companion_expression_freedom()
    if mode == "conservative":
        return ExpressionCapabilityTier.MICRO
    if mode == "expressive":
        return ExpressionCapabilityTier.LARGE
    return None


def resolve_expression_capability(
    *,
    sidecar_basename: str = "",
) -> ExpressionCapabilityTier:
    override = _freedom_override_tier()
    if override is not None:
        return override

    patterns = _load_basename_tiers()
    basename = sidecar_basename
    if not basename:
        try:
            from core.auxiliary_cognition import active_cognition_basename

            basename = active_cognition_basename()
        except Exception:
            basename = ""

    tier = _tier_from_basename(basename, patterns) if basename else None
    if tier is None:
        tier = ExpressionCapabilityTier.SMALL

    return _telemetry_downgrade(tier)


def max_expression_level(tier: ExpressionCapabilityTier) -> ExpressionLevel:
    return _TIER_MAX_LEVEL.get(tier, ExpressionLevel.TEMPLATE)


def allows_sidecar_rewrite(tier: ExpressionCapabilityTier) -> bool:
    return _TIER_ALLOWS_L2.get(tier, False)


def allows_full_generate(tier: ExpressionCapabilityTier, *, trigger: str = "") -> bool:
    if trigger == "test":
        from core import app_settings

        if app_settings.get_companion_expression_freedom() == "expressive":
            return True
    return _TIER_ALLOWS_L3.get(tier, False)
