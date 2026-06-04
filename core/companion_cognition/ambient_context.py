"""Ambient context — daypart, season, and context bundle (no user profiling)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from core.companion_cognition.mood_drift import AmbientMoodState, load_mood_state, tick_mood_drift
from core.companion_cognition.motifs import load_motif_state, resolve_active_motif
from core.companion_cognition.personality import CompanionPersonalityVector

DAYPARTS = ("morning", "afternoon", "evening", "late_night")
SEASONS = ("spring", "summer", "autumn", "winter")


@dataclass(frozen=True)
class AmbientContext:
    """Computed ambient layer passed through the cognition pipeline."""

    ambient_mood: AmbientMoodState
    daypart: str
    season: str | None
    active_motif: str | None
    now_ts: float


def _local_datetime(ts: float) -> datetime:
    return datetime.fromtimestamp(ts).astimezone()


def resolve_daypart(ts: float) -> str:
    """Local wall-clock daypart — morning 05–11, afternoon 12–16, evening 17–21, late_night else."""
    hour = _local_datetime(ts).hour
    if 5 <= hour <= 11:
        return "morning"
    if 12 <= hour <= 16:
        return "afternoon"
    if 17 <= hour <= 21:
        return "evening"
    return "late_night"


def resolve_season(ts: float, *, hemisphere: str = "north") -> str | None:
    """Deterministic calendar season from local date."""
    month = _local_datetime(ts).month
    if hemisphere == "south":
        month = ((month + 5) % 12) + 1
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def build_ambient_context(
    *,
    now_ts: float,
    personality: CompanionPersonalityVector,
    seasonal_enabled: bool = True,
    hemisphere: str = "north",
    motifs_enabled: bool = True,
    mood_drift_enabled: bool = True,
) -> AmbientContext:
    mood = load_mood_state()
    if mood_drift_enabled:
        mood = tick_mood_drift(mood, personality, now_ts=now_ts, on_session_start=False)
    daypart = resolve_daypart(now_ts)
    season = resolve_season(now_ts, hemisphere=hemisphere) if seasonal_enabled else None
    motif_state = load_motif_state()
    active_motif = resolve_active_motif(motif_state, now_ts) if motifs_enabled else None
    return AmbientContext(
        ambient_mood=mood,
        daypart=daypart,
        season=season,
        active_motif=active_motif,
        now_ts=now_ts,
    )


def session_start_ambient_context(
    *,
    now_ts: float,
    personality: CompanionPersonalityVector,
    seasonal_enabled: bool = True,
    hemisphere: str = "north",
    motifs_enabled: bool = True,
    mood_drift_enabled: bool = True,
) -> AmbientContext:
    """Session-start mood tick (major drift eligible)."""
    mood = load_mood_state()
    if mood_drift_enabled:
        mood = tick_mood_drift(mood, personality, now_ts=now_ts, on_session_start=True)
    daypart = resolve_daypart(now_ts)
    season = resolve_season(now_ts, hemisphere=hemisphere) if seasonal_enabled else None
    motif_state = load_motif_state()
    active_motif = resolve_active_motif(motif_state, now_ts) if motifs_enabled else None
    return AmbientContext(
        ambient_mood=mood,
        daypart=daypart,
        season=season,
        active_motif=active_motif,
        now_ts=now_ts,
    )
