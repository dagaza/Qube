"""ObservationEngine — deterministic trigger → observation mapping."""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

from core import app_settings
from core.companion_cognition.ambient_context import resolve_daypart, resolve_season
from core.companion_cognition.types import CompanionObservation, CompanionTriggerEvent
from core.companion_verbal_policy import CompanionVerbalGateContext

logger = logging.getLogger("Qube.CompanionVerbal")

# Allowlisted fact keys per observation type (privacy gate).
ALLOWED_FACT_KEYS: dict[str, frozenset[str]] = {
    "quiet_period": frozenset({"idle_sec", "main_hidden", "daypart", "season"}),
    "library_update_completed": frozenset({"file_count"}),
    "model_download_completed": frozenset({"basename"}),
    "settings_preview": frozenset({"daypart"}),
    "model_ready": frozenset({"basename"}),
    "companion_startup": frozenset({"session_index", "daypart", "season"}),
    "focus_detected": frozenset({"idle_sec", "main_hidden", "daypart", "season"}),
    "system_resumed": frozenset({"gap_sec", "daypart", "season"}),
    "usage_milestone": frozenset({"milestone_id"}),
    "usage_pattern": frozenset({"days_active", "session_count_tier", "daypart", "season"}),
}

_AMBIENT_CONTEXT_TYPES = frozenset(
    {
        "quiet_period",
        "focus_detected",
        "system_resumed",
        "companion_startup",
        "usage_pattern",
        "settings_preview",
    }
)

_TRIGGER_TO_OBSERVATION: dict[str, str] = {
    "idle": "quiet_period",
    "ingest_complete": "library_update_completed",
    "download_complete": "model_download_completed",
    "test": "settings_preview",
    "model_loaded": "model_ready",
    "startup": "companion_startup",
    "focus_session": "focus_detected",
    "wake_from_sleep": "system_resumed",
    "milestone": "usage_milestone",
    "long_term_usage_patterns": "usage_pattern",
}

_BASENAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,120}$")


def sanitize_basename(value: Any) -> str:
    """Filename only — strip paths and reject unsafe characters."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    base = os.path.basename(raw.replace("\\", "/"))
    if _BASENAME_RE.match(base):
        return base
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "", base)[:120]
    return cleaned


def _reject_payload_keys(payload: dict[str, Any], allowed: frozenset[str]) -> str | None:
    for key in payload:
        if key not in allowed:
            return f"disallowed_key:{key}"
    return None


def observe(
    event: CompanionTriggerEvent,
    ctx: CompanionVerbalGateContext,
) -> CompanionObservation | None:
    """Map a trigger event to a structured observation, or None if rejected."""
    source = str(event.source or "").strip().lower()
    obs_type = _TRIGGER_TO_OBSERVATION.get(source)
    if obs_type is None:
        logger.info("[CompanionCognition] rejected_observation reason=unknown_trigger source=%s", source)
        return None

    allowed = ALLOWED_FACT_KEYS.get(obs_type, frozenset())
    payload = dict(event.payload or {})
    reject = _reject_payload_keys(payload, allowed | frozenset({"trigger"}))
    if reject:
        logger.info(
            "[CompanionCognition] rejected_observation reason=%s source=%s",
            reject,
            source,
        )
        return None

    facts: dict[str, Any] = {}

    if obs_type == "quiet_period":
        idle_since = ctx.idle_since
        idle_sec = 0.0
        if idle_since is not None:
            idle_sec = max(0.0, (ctx.now if ctx.now else time.time()) - idle_since)
        facts = {
            "idle_sec": round(idle_sec, 1),
            "main_hidden": not ctx.main_window_visible or ctx.main_window_minimized,
        }
    elif obs_type == "library_update_completed":
        try:
            fc = max(1, int(payload.get("file_count", 1)))
        except (TypeError, ValueError):
            fc = 1
        facts = {"file_count": fc}
    elif obs_type in ("model_download_completed", "model_ready"):
        basename = sanitize_basename(payload.get("basename") or payload.get("model_basename"))
        if not basename:
            logger.info("[CompanionCognition] rejected_observation reason=invalid_basename source=%s", source)
            return None
        facts = {"basename": basename}
    elif obs_type == "companion_startup":
        try:
            idx = max(1, int(payload.get("session_index", 1)))
        except (TypeError, ValueError):
            idx = 1
        facts = {"session_index": idx}
    elif obs_type == "focus_detected":
        idle_since = ctx.idle_since
        idle_sec = 0.0
        if idle_since is not None:
            idle_sec = max(0.0, (ctx.now if ctx.now else time.time()) - idle_since)
        facts = {
            "idle_sec": round(idle_sec, 1),
            "main_hidden": not ctx.main_window_visible or ctx.main_window_minimized,
        }
    elif obs_type == "system_resumed":
        try:
            gap = max(0.0, float(payload.get("gap_sec", 0)))
        except (TypeError, ValueError):
            gap = 0.0
        facts = {"gap_sec": round(gap, 1)}
    elif obs_type == "usage_milestone":
        mid = str(payload.get("milestone_id") or "").strip()[:64]
        if not mid or not re.match(r"^[a-z0-9_]+$", mid):
            logger.info("[CompanionCognition] rejected_observation reason=invalid_milestone source=%s", source)
            return None
        facts = {"milestone_id": mid}
    elif obs_type == "usage_pattern":
        try:
            days = max(0, int(payload.get("days_active", 0)))
        except (TypeError, ValueError):
            days = 0
        tier = str(payload.get("session_count_tier") or "unknown").strip()[:32]
        if tier not in ("new", "regular", "veteran", "unknown"):
            tier = "unknown"
        facts = {"days_active": days, "session_count_tier": tier}
    # settings_preview: empty facts

    if obs_type in _AMBIENT_CONTEXT_TYPES:
        now_ts = ctx.now if ctx.now else time.time()
        facts["daypart"] = resolve_daypart(now_ts)
        if obs_type != "settings_preview" and app_settings.get_companion_seasonal_enabled():
            season = resolve_season(now_ts, hemisphere=app_settings.get_companion_seasonal_hemisphere())
            if season:
                facts["season"] = season

    # Final sanity: facts keys must match allowlist
    if frozenset(facts.keys()) - allowed:
        logger.info("[CompanionCognition] rejected_observation reason=facts_allowlist source=%s", source)
        return None

    return CompanionObservation(
        type=obs_type,
        facts=facts,
        confidence=1.0,
        trigger_source=source,
    )


def trigger_event_from_legacy(
    trigger: str,
    *,
    ts: float | None = None,
    **payload: Any,
) -> CompanionTriggerEvent:
    """Build a CompanionTriggerEvent from legacy scheduler kwargs."""
    return CompanionTriggerEvent(
        source=str(trigger or "idle").strip().lower(),
        ts=ts if ts is not None else time.time(),
        payload=dict(payload),
    )
