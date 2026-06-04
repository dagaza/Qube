"""Ambient mood drift — weather-like state, not emotion simulation."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from core.companion_cognition.personality import CompanionPersonalityVector

logger = logging.getLogger("Qube.CompanionVerbal")

_SCHEMA_VERSION = 1
_MOOD_PATH = Path.home() / ".qube" / "companion" / "mood_state.json"
_MAJOR_DRIFT_HOURS = 12.0
_STRENGTH_MIN = 0.35
_STRENGTH_MAX = 0.85
_NUDGE_CAP = 0.03
_NUDGE_MIN_INTERVAL_H = 1.0

AMBIENT_MOOD_STATES = ("reflective", "cozy", "curious", "playful", "observant", "quiet")

_MOOD_NEIGHBORS: dict[str, tuple[str, ...]] = {
    "reflective": ("quiet", "observant", "cozy"),
    "cozy": ("reflective", "quiet", "playful"),
    "curious": ("observant", "playful", "reflective"),
    "playful": ("curious", "cozy", "observant"),
    "observant": ("curious", "quiet", "reflective"),
    "quiet": ("reflective", "cozy", "observant"),
}

_PERSONALITY_BIAS: dict[str, tuple[str, ...]] = {
    "warmth": ("cozy", "quiet"),
    "humor": ("playful",),
    "curiosity": ("curious", "observant"),
    "playfulness": ("playful", "curious"),
}


@dataclass
class AmbientMoodState:
    state: str = "observant"
    strength: float = 0.5
    last_drift_ts: float = 0.0
    drift_generation: int = 0
    last_nudge_ts: float = 0.0

    def clamped(self) -> AmbientMoodState:
        st = self.state if self.state in AMBIENT_MOOD_STATES else "observant"
        strength = max(_STRENGTH_MIN, min(_STRENGTH_MAX, float(self.strength)))
        return AmbientMoodState(
            state=st,
            strength=strength,
            last_drift_ts=float(self.last_drift_ts),
            drift_generation=int(self.drift_generation),
            last_nudge_ts=float(self.last_nudge_ts),
        )


def _default_state(now_ts: float) -> AmbientMoodState:
    return AmbientMoodState(state="observant", strength=0.5, last_drift_ts=now_ts)


def load_mood_state() -> AmbientMoodState:
    if not _MOOD_PATH.is_file():
        return _default_state(0.0)
    try:
        data = json.loads(_MOOD_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return _default_state(0.0)
        return AmbientMoodState(
            state=str(data.get("state") or "observant"),
            strength=float(data.get("strength") or 0.5),
            last_drift_ts=float(data.get("last_drift_ts") or 0.0),
            drift_generation=int(data.get("drift_generation") or 0),
            last_nudge_ts=float(data.get("last_nudge_ts") or 0.0),
        ).clamped()
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning("[CompanionCognition] mood_state load failed: %s", e)
        return _default_state(0.0)


def save_mood_state(state: AmbientMoodState) -> None:
    s = state.clamped()
    try:
        _MOOD_PATH.parent.mkdir(parents=True, exist_ok=True)
        _MOOD_PATH.write_text(
            json.dumps(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "state": s.state,
                    "strength": round(s.strength, 4),
                    "last_drift_ts": s.last_drift_ts,
                    "drift_generation": s.drift_generation,
                    "last_nudge_ts": s.last_nudge_ts,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError as e:
        logger.warning("[CompanionCognition] mood_state save failed: %s", e)


def _stable_hash(*parts: str) -> float:
    h = hashlib.md5(":".join(parts).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _personality_bias(personality: CompanionPersonalityVector, candidate: str) -> float:
    p = personality.clamped()
    score = 0.0
    for attr, moods in _PERSONALITY_BIAS.items():
        val = getattr(p, attr, 0.5)
        if candidate in moods:
            score += val * 0.1
    return min(score, 0.25)


def _pick_neighbor(
    current: str,
    personality: CompanionPersonalityVector,
    *,
    date_iso: str,
    strength: float,
) -> str:
    neighbors = _MOOD_NEIGHBORS.get(current, AMBIENT_MOOD_STATES)
    scored: list[tuple[float, str]] = []
    for n in neighbors:
        h = _stable_hash(date_iso, current, n)
        score = _personality_bias(personality, n) * 0.25 + strength * 0.15 + h * 0.60
        scored.append((score, n))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def tick_mood_drift(
    state: AmbientMoodState,
    personality: CompanionPersonalityVector,
    *,
    now_ts: float,
    on_session_start: bool = False,
) -> AmbientMoodState:
    """Apply major drift when due; persist if changed."""
    s = state.clamped()
    if s.last_drift_ts <= 0:
        return AmbientMoodState(
            state=s.state,
            strength=s.strength,
            last_drift_ts=now_ts,
            drift_generation=s.drift_generation,
            last_nudge_ts=s.last_nudge_ts,
        )

    elapsed_h = max(0.0, (now_ts - s.last_drift_ts) / 3600.0)
    last_date = datetime.fromtimestamp(s.last_drift_ts).date().isoformat()
    today = datetime.fromtimestamp(now_ts).date().isoformat()
    major_due = elapsed_h >= _MAJOR_DRIFT_HOURS or last_date != today or on_session_start

    if not major_due:
        return s

    next_state = _pick_neighbor(s.state, personality, date_iso=today, strength=s.strength)
    h = _stable_hash(today, next_state, str(s.drift_generation))
    next_strength = _STRENGTH_MIN + h * (_STRENGTH_MAX - _STRENGTH_MIN)
    updated = AmbientMoodState(
        state=next_state,
        strength=next_strength,
        last_drift_ts=now_ts,
        drift_generation=s.drift_generation + 1,
        last_nudge_ts=s.last_nudge_ts,
    )
    save_mood_state(updated)
    return updated


def nudge_mood_after_emission(
    state: AmbientMoodState,
    *,
    intent: str,
    now_ts: float,
) -> AmbientMoodState:
    """Minor strength nudge toward intent-aligned ambient mood — max once per hour."""
    s = state.clamped()
    if (now_ts - s.last_nudge_ts) < _NUDGE_MIN_INTERVAL_H * 3600:
        return s
    aligned = _intent_aligned_mood(intent)
    if aligned is None:
        return s
    delta = _NUDGE_CAP if aligned == s.state else _NUDGE_CAP * 0.5
    strength = min(_STRENGTH_MAX, s.strength + delta)
    updated = AmbientMoodState(
        state=s.state,
        strength=strength,
        last_drift_ts=s.last_drift_ts,
        drift_generation=s.drift_generation,
        last_nudge_ts=now_ts,
    )
    save_mood_state(updated)
    return updated


def _intent_aligned_mood(intent: str) -> str | None:
    mapping = {
        "reflection": "reflective",
        "fact": "reflective",
        "humor": "playful",
        "self_expression": "playful",
        "curiosity": "curious",
        "atmosphere": "quiet",
        "wellbeing": "cozy",
    }
    return mapping.get(intent)


def ambient_mood_intent_bias(ambient_state: str, intent: str) -> float:
    """Score multiplier for intent candidates from ambient mood."""
    biases: dict[str, dict[str, float]] = {
        "reflective": {"reflection": 1.25, "fact": 1.15, "wellbeing": 1.05},
        "cozy": {"wellbeing": 1.2, "self_expression": 1.1, "atmosphere": 1.1},
        "curious": {"curiosity": 1.25, "fact": 1.1, "reflection": 1.05},
        "playful": {"humor": 1.25, "self_expression": 1.15, "curiosity": 1.05},
        "observant": {"atmosphere": 1.2, "fact": 1.1, "curiosity": 1.1},
        "quiet": {"wellbeing": 1.2, "atmosphere": 1.2, "reflection": 1.05},
    }
    return biases.get(ambient_state, {}).get(intent, 1.0)


def ambient_mood_voice_nudge(ambient_state: str, voice: str) -> str:
    """Nudge voice toward ambient-mood-aligned expression voice."""
    mapping: dict[str, str] = {
        "reflective": "reflective",
        "cozy": "cozy",
        "curious": "curious",
        "playful": "playful",
        "observant": "observational",
        "quiet": "observational",
    }
    target = mapping.get(ambient_state)
    if not target or target == voice:
        return voice
    return target
