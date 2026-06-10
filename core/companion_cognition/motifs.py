"""Companion motif rotation — recurring identity themes (not memory)."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("Qube.CompanionVerbal")

_SCHEMA_VERSION = 1
_MOTIF_PATH = Path.home() / ".qube" / "companion" / "motif_state.json"

MOTIF_CATALOG = ("pixels", "routines", "observing", "weather", "tea", "quiet")

_MOTIF_BOOST = 1.12
_MOTIF_PENALTY = 0.75


@dataclass
class MotifState:
    active_motif: str = "observing"
    motif_since_ts: float = 0.0
    recent_motifs: list[str] = field(default_factory=list)


def load_motif_state() -> MotifState:
    if not _MOTIF_PATH.is_file():
        return MotifState(active_motif="observing", motif_since_ts=0.0)
    try:
        data = json.loads(_MOTIF_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return MotifState()
        recent = [str(m) for m in (data.get("recent_motifs") or []) if str(m) in MOTIF_CATALOG]
        active = str(data.get("active_motif") or "observing")
        if active not in MOTIF_CATALOG:
            active = "observing"
        return MotifState(
            active_motif=active,
            motif_since_ts=float(data.get("motif_since_ts") or 0.0),
            recent_motifs=recent[-16:],
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
        logger.warning("[CompanionCognition] motif_state load failed: %s", e)
        return MotifState()


def save_motif_state(state: MotifState) -> None:
    try:
        _MOTIF_PATH.parent.mkdir(parents=True, exist_ok=True)
        _MOTIF_PATH.write_text(
            json.dumps(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "active_motif": state.active_motif,
                    "motif_since_ts": state.motif_since_ts,
                    "recent_motifs": list(state.recent_motifs)[-16:],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError as e:
        logger.warning("[CompanionCognition] motif_state save failed: %s", e)


def _iso_week_key(ts: float) -> str:
    dt = datetime.fromtimestamp(ts).astimezone()
    iso = dt.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _pick_motif_for_week(week_key: str) -> str:
    h = hashlib.md5(week_key.encode()).hexdigest()
    idx = int(h[:8], 16) % len(MOTIF_CATALOG)
    return MOTIF_CATALOG[idx]


def resolve_active_motif(state: MotifState, now_ts: float) -> str | None:
    week_key = _iso_week_key(now_ts)
    expected = _pick_motif_for_week(week_key)
    if state.active_motif == expected and state.motif_since_ts > 0:
        return state.active_motif
    updated = MotifState(
        active_motif=expected,
        motif_since_ts=now_ts,
        recent_motifs=state.recent_motifs,
    )
    save_motif_state(updated)
    return expected


def motif_selection_boost(active_motif: str | None, message_motifs: tuple[str, ...]) -> float:
    if not active_motif or not message_motifs:
        return 1.0
    if active_motif in message_motifs:
        return _MOTIF_BOOST
    return 1.0


def motif_recent_penalty(recent_motifs: tuple[str, ...], message_motifs: tuple[str, ...]) -> float:
    if not message_motifs or not recent_motifs:
        return 1.0
    recent = list(recent_motifs)[-10:]
    for m in message_motifs:
        if recent.count(m) >= 3:
            return _MOTIF_PENALTY
    return 1.0


def record_motif_emission(active_motif: str | None, message_motifs: tuple[str, ...], now_ts: float) -> None:
    if not message_motifs:
        return
    state = load_motif_state()
    recent = list(state.recent_motifs)
    for m in message_motifs:
        if m in MOTIF_CATALOG:
            recent.append(m)
    save_motif_state(
        MotifState(
            active_motif=state.active_motif or active_motif or "observing",
            motif_since_ts=state.motif_since_ts or now_ts,
            recent_motifs=recent[-16:],
        )
    )
