"""Coarse Qube usage counters for companion milestones (no content profiling)."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

logger = logging.getLogger("Qube.CompanionVerbal")

_SCHEMA_VERSION = 2
_COUNTERS_PATH = Path.home() / ".qube" / "companion" / "usage_counters.json"

# Priority order: days > sessions > library > companion captions > years
_MILESTONE_CHECKS: tuple[tuple[str, str, int], ...] = (
    ("days_active", "days_7", 7),
    ("days_active", "days_30", 30),
    ("days_active", "days_100", 100),
    ("days_active", "days_365", 365),
    ("days_active", "years_2", 730),
    ("session_count", "sessions_10", 10),
    ("session_count", "sessions_50", 50),
    ("session_count", "sessions_100", 100),
    ("session_count", "sessions_365", 365),
    ("ingest_events", "library_25", 25),
    ("ingest_events", "library_100", 100),
    ("captions_emitted", "companion_50", 50),
    ("captions_emitted", "companion_200", 200),
)

_USAGE_PATTERN_MIN_DAYS = 7
_USAGE_PATTERN_COOLDOWN_DAYS = 7


def _today_iso() -> str:
    return date.today().isoformat()


def _default_counters() -> dict:
    return {
        "schema_version": _SCHEMA_VERSION,
        "session_count": 0,
        "days_active": 0,
        "last_active_date": "",
        "ingest_events": 0,
        "captions_emitted": 0,
        "milestones_emitted": [],
        "last_milestone_ts": 0.0,
        "last_usage_pattern_date": "",
    }


def load_counters() -> dict:
    if not _COUNTERS_PATH.is_file():
        return _default_counters()
    try:
        data = json.loads(_COUNTERS_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            out = _default_counters()
            out.update(data)
            out["schema_version"] = _SCHEMA_VERSION
            return out
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("[CompanionCognition] usage_counters load failed: %s", e)
    return _default_counters()


def save_counters(data: dict) -> None:
    _COUNTERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    data["schema_version"] = _SCHEMA_VERSION
    _COUNTERS_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _pick_milestone(data: dict) -> str | None:
    emitted = set(data.get("milestones_emitted") or [])
    for field, mid, threshold in _MILESTONE_CHECKS:
        val = int(data.get(field) or 0)
        if val >= threshold and mid not in emitted:
            return mid
    return None


def record_session_start() -> tuple[str | None, dict]:
    """Increment session counter; return (milestone_id, updated counters) if new milestone."""
    data = load_counters()
    data["session_count"] = int(data.get("session_count") or 0) + 1
    today = _today_iso()
    if data.get("last_active_date") != today:
        data["days_active"] = int(data.get("days_active") or 0) + 1
        data["last_active_date"] = today
    milestone_id = _pick_milestone(data)
    if milestone_id:
        emitted = set(data.get("milestones_emitted") or [])
        emitted.add(milestone_id)
        data["milestones_emitted"] = sorted(emitted)
        import time

        data["last_milestone_ts"] = time.time()
    save_counters(data)
    return milestone_id, data


def record_ingest_event() -> None:
    data = load_counters()
    data["ingest_events"] = int(data.get("ingest_events") or 0) + 1
    save_counters(data)


def record_caption_emission() -> None:
    data = load_counters()
    data["captions_emitted"] = int(data.get("captions_emitted") or 0) + 1
    save_counters(data)


def should_emit_usage_pattern(data: dict | None = None) -> bool:
    """At most once per 7 days when days_active >= 7."""
    counters = data if data is not None else load_counters()
    days = int(counters.get("days_active") or 0)
    if days < _USAGE_PATTERN_MIN_DAYS:
        return False
    today = _today_iso()
    last = str(counters.get("last_usage_pattern_date") or "")
    if last:
        try:
            last_d = date.fromisoformat(last)
            delta = (date.today() - last_d).days
            if delta < _USAGE_PATTERN_COOLDOWN_DAYS:
                return False
        except ValueError:
            pass
    return True


def mark_usage_pattern_emitted() -> None:
    data = load_counters()
    data["last_usage_pattern_date"] = _today_iso()
    save_counters(data)


def session_count_tier(session_count: int) -> str:
    if session_count < 5:
        return "new"
    if session_count < 50:
        return "regular"
    return "veteran"
