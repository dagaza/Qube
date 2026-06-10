"""Curated companion message library — load, validate, select."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.companion_cognition.ambient_context import AmbientContext
from core.companion_cognition.motifs import motif_recent_penalty, motif_selection_boost
from core.companion_cognition.personality import CompanionPersonalityVector
from core.companion_cognition.types import CompanionThought
from core.companion_cognition.variety import VarietySnapshot
from core.companion_line_quality import is_acceptable_companion_line
from core.companion_verbal_prompts import COMPANION_LINE_MAX_CHARS, truncate_companion_caption

logger = logging.getLogger("Qube.CompanionVerbal")

VALID_INTENTS = frozenset(
    {
        "acknowledge_effort",
        "observation",
        "reflection",
        "wellbeing",
        "self_expression",
        "atmosphere",
        "humor",
        "fact",
        "curiosity",
        "celebration",
    }
)

VALID_VOICES = frozenset(
    {
        "cozy",
        "dry",
        "curious",
        "playful",
        "reflective",
        "wry",
        "observational",
    }
)

VOICE_ADJACENCY: dict[str, tuple[str, ...]] = {
    "cozy": ("observational", "reflective"),
    "dry": ("wry", "observational"),
    "curious": ("observational", "reflective"),
    "playful": ("wry", "dry"),
    "reflective": ("observational", "cozy"),
    "wry": ("dry", "playful"),
    "observational": ("cozy", "reflective"),
}

RARITY_WEIGHT = {"common": 1.0, "uncommon": 1.4, "rare": 2.0}

VALID_AMBIENT_MOODS = frozenset(
    {"reflective", "cozy", "curious", "playful", "observant", "quiet"}
)
VALID_DAYPARTS = frozenset({"morning", "afternoon", "evening", "late_night"})
VALID_SEASONS = frozenset({"spring", "summer", "autumn", "winter"})
VALID_MOTIFS = frozenset({"pixels", "routines", "observing", "weather", "tea", "quiet"})


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def bundled_messages_path() -> Path:
    return _project_root() / "assets" / "companion" / "messages.v1.json"


def default_user_messages_path() -> Path:
    return Path.home() / ".qube" / "companion" / "messages.json"


@dataclass(frozen=True)
class CuratedMessage:
    id: str
    intent: str
    text: str
    mood: tuple[str, ...]
    energy: str
    rarity: str
    contexts: tuple[str, ...]
    cooldown_hours: float
    voice: str = "observational"
    tags: tuple[str, ...] = ()
    ambient_moods: tuple[str, ...] = ()
    dayparts: tuple[str, ...] = ()
    seasons: tuple[str, ...] = ()
    motifs: tuple[str, ...] = ()
    milestone_ids: tuple[str, ...] = ()
    pack: str = ""
    min_warmth: float = 0.0
    max_warmth: float = 1.0


@dataclass(frozen=True)
class MessageTemplate:
    id: str
    intent: str
    pattern: str
    placeholders: tuple[str, ...]
    contexts: tuple[str, ...]
    cooldown_hours: float
    voice: str = "observational"
    mood: tuple[str, ...] = ()
    energy: str = "low"
    rarity: str = "common"
    min_warmth: float = 0.0
    max_warmth: float = 1.0


@dataclass
class MessageLibrary:
    schema_version: int = 1
    messages: list[CuratedMessage] = field(default_factory=list)
    templates: list[MessageTemplate] = field(default_factory=list)
    _by_intent: dict[str, list[CuratedMessage]] = field(default_factory=dict, repr=False)
    _templates_by_intent: dict[str, list[MessageTemplate]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._by_intent = {}
        for msg in self.messages:
            self._by_intent.setdefault(msg.intent, []).append(msg)
        self._templates_by_intent = {}
        for tpl in self.templates:
            self._templates_by_intent.setdefault(tpl.intent, []).append(tpl)

    def select_message(
        self,
        thought: CompanionThought,
        variety: VarietySnapshot,
        personality: CompanionPersonalityVector,
        ambient: AmbientContext | None = None,
        *,
        for_preview: bool = False,
    ) -> CuratedMessage | None:
        candidates = self._filter_messages(
            thought, variety, personality, ambient, for_preview=for_preview
        )
        return self._pick_best(candidates, variety, thought, ambient)

    def select_template(
        self,
        thought: CompanionThought,
        variety: VarietySnapshot,
        personality: CompanionPersonalityVector,
    ) -> MessageTemplate | None:
        candidates: list[MessageTemplate] = []
        for tpl in self._templates_by_intent.get(thought.intent, []):
            if thought.observation_type and thought.observation_type not in tpl.contexts:
                continue
            if tpl.mood and thought.mood not in tpl.mood:
                continue
            if tpl.energy and thought.energy != tpl.energy:
                continue
            if not (tpl.min_warmth <= personality.warmth <= tpl.max_warmth):
                continue
            if tpl.id in variety.recent_message_ids:
                continue
            store = variety.message_last_used
            last = store.get(tpl.id)
            if last is not None and (variety.now - last) < tpl.cooldown_hours * 3600:
                continue
            candidates.append(tpl)
        if not candidates:
            return None
        return self._pick_best_templates(candidates, variety, thought)

    def _filter_messages(
        self,
        thought: CompanionThought,
        variety: VarietySnapshot,
        personality: CompanionPersonalityVector,
        ambient: AmbientContext | None = None,
        *,
        for_preview: bool = False,
    ) -> list[CuratedMessage]:
        out: list[CuratedMessage] = []
        milestone_id = str((thought.slots or {}).get("milestone_id") or "")
        for msg in self._by_intent.get(thought.intent, []):
            if thought.observation_type and thought.observation_type not in msg.contexts:
                continue
            if thought.observation_type == "usage_milestone" and msg.milestone_ids:
                if milestone_id not in msg.milestone_ids:
                    continue
            elif msg.milestone_ids and thought.observation_type != "usage_milestone":
                continue
            if msg.mood and thought.mood not in msg.mood:
                continue
            if msg.energy and thought.energy != msg.energy:
                continue
            if not (msg.min_warmth <= personality.warmth <= msg.max_warmth):
                continue
            if not for_preview:
                if msg.id in variety.recent_message_ids:
                    continue
                last = variety.message_last_used.get(msg.id)
                if last is not None and (variety.now - last) < msg.cooldown_hours * 3600:
                    continue
            if not is_acceptable_companion_line(msg.text):
                continue
            out.append(msg)
        return out

    def _pick_best(
        self,
        candidates: list[CuratedMessage],
        variety: VarietySnapshot,
        thought: CompanionThought,
        ambient: AmbientContext | None = None,
    ) -> CuratedMessage | None:
        if not candidates:
            return None
        day = datetime.fromtimestamp(variety.now, tz=timezone.utc).timetuple().tm_yday
        scored: list[tuple[float, CuratedMessage]] = []
        intent_pen = _intent_penalty(variety, thought.intent)
        voice_pen = _voice_penalty(variety, thought.voice)
        ambient_state = ambient.ambient_mood.state if ambient else ""
        daypart = ambient.daypart if ambient else ""
        season = ambient.season if ambient else None
        active_motif = ambient.active_motif if ambient else None
        for msg in candidates:
            rarity = RARITY_WEIGHT.get(msg.rarity, 1.0)
            tie = _stable_tie(msg.id, day)
            voice_match = _voice_match_bonus(thought.voice, msg.voice)
            amb_boost = _ambient_mood_boost(ambient_state, msg.ambient_moods)
            day_boost = _daypart_boost(daypart, msg.dayparts)
            season_boost = _season_boost(season, msg.seasons)
            motif_boost = motif_selection_boost(active_motif, msg.motifs)
            motif_pen = motif_recent_penalty(variety.recent_motifs, msg.motifs)
            score = (
                rarity
                * intent_pen
                * voice_pen
                * voice_match
                * amb_boost
                * day_boost
                * season_boost
                * motif_boost
                * motif_pen
                * tie
            )
            scored.append((score, msg))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def _pick_best_templates(
        self,
        candidates: list[MessageTemplate],
        variety: VarietySnapshot,
        thought: CompanionThought,
    ) -> MessageTemplate | None:
        day = datetime.fromtimestamp(variety.now, tz=timezone.utc).timetuple().tm_yday
        scored: list[tuple[float, MessageTemplate]] = []
        intent_pen = _intent_penalty(variety, thought.intent)
        voice_pen = _voice_penalty(variety, thought.voice)
        for tpl in candidates:
            rarity = RARITY_WEIGHT.get(tpl.rarity, 1.0)
            tie = _stable_tie(tpl.id, day)
            voice_match = _voice_match_bonus(thought.voice, tpl.voice)
            scored.append((rarity * intent_pen * voice_pen * voice_match * tie, tpl))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1] if scored else None


def _intent_penalty(variety: VarietySnapshot, intent: str) -> float:
    recent = list(variety.recent_intents)[-3:]
    if not recent:
        return 1.0
    count = recent.count(intent)
    if count >= 2:
        return 0.2
    if count >= 1:
        return 0.6
    return 1.0


def _voice_penalty(variety: VarietySnapshot, voice: str) -> float:
    if not voice:
        return 1.0
    recent = list(variety.recent_voices)[-3:]
    if not recent:
        return 1.0
    count = recent.count(voice)
    if count >= 2:
        return 0.25
    if count >= 1:
        return 0.7
    return 1.0


def _voice_match_bonus(thought_voice: str, message_voice: str) -> float:
    if not thought_voice or not message_voice:
        return 1.0
    if thought_voice == message_voice:
        return 1.35
    if message_voice in VOICE_ADJACENCY.get(thought_voice, ()):
        return 1.1
    return 0.85


def _ambient_mood_boost(ambient_state: str, msg_moods: tuple[str, ...]) -> float:
    if not ambient_state or not msg_moods:
        return 1.0
    if ambient_state in msg_moods:
        return 1.2
    return 1.0


def _daypart_boost(daypart: str, msg_dayparts: tuple[str, ...]) -> float:
    if not daypart or not msg_dayparts:
        return 1.0
    if daypart in msg_dayparts:
        return 1.15
    return 1.0


def _season_boost(season: str | None, msg_seasons: tuple[str, ...]) -> float:
    if not season or not msg_seasons:
        return 1.0
    if season in msg_seasons:
        return 1.2
    return 1.0


def _stable_tie(item_id: str, day_of_year: int) -> float:
    h = hashlib.md5(f"{item_id}:{day_of_year}".encode()).hexdigest()
    return 0.8 + (int(h[:8], 16) % 1000) / 5000.0


def ensure_user_messages_seeded(user_path: Path | None = None) -> Path:
    path = Path(user_path) if user_path else default_user_messages_path()
    bundled = bundled_messages_path()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        if bundled.is_file():
            shutil.copyfile(bundled, path)
        else:
            path.write_text(json.dumps({"schema_version": 1, "messages": [], "templates": []}))
        return path
    if bundled.is_file() and bundled.stat().st_mtime > path.stat().st_mtime:
        try:
            bundled_data = json.loads(bundled.read_text(encoding="utf-8"))
            user_data = json.loads(path.read_text(encoding="utf-8"))
            b_ver = int(bundled_data.get("schema_version", 0) or 0)
            u_ver = int(user_data.get("schema_version", 0) or 0)
            if b_ver > u_ver:
                shutil.copyfile(bundled, path)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return path


def load_message_library(path: Path | None = None) -> MessageLibrary:
    user_path = ensure_user_messages_seeded(path)
    try:
        data = json.loads(user_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("[CompanionCognition] message library load failed: %s — using bundled", e)
        bundled = bundled_messages_path()
        if bundled.is_file():
            data = json.loads(bundled.read_text(encoding="utf-8"))
        else:
            return MessageLibrary()

    ok, err = validate_library_dict(data)
    if not ok:
        logger.warning("[CompanionCognition] message library invalid: %s — bundled fallback", err)
        bundled = bundled_messages_path()
        if bundled.is_file():
            data = json.loads(bundled.read_text(encoding="utf-8"))
        else:
            return MessageLibrary()

    return _parse_library(data)


def _parse_library(data: dict[str, Any]) -> MessageLibrary:
    messages: list[CuratedMessage] = []
    for raw in data.get("messages") or []:
        if not isinstance(raw, dict):
            continue
        messages.append(
            CuratedMessage(
                id=str(raw.get("id") or ""),
                intent=str(raw.get("intent") or ""),
                text=str(raw.get("text") or ""),
                mood=tuple(str(m) for m in (raw.get("mood") or [])),
                energy=str(raw.get("energy") or "low"),
                rarity=str(raw.get("rarity") or "common"),
                contexts=tuple(str(c) for c in (raw.get("contexts") or [])),
                cooldown_hours=float(raw.get("cooldown_hours") or 168),
                voice=str(raw.get("voice") or "observational"),
                tags=tuple(str(t) for t in (raw.get("tags") or [])),
                ambient_moods=tuple(str(m) for m in (raw.get("ambient_moods") or [])),
                dayparts=tuple(str(d) for d in (raw.get("dayparts") or [])),
                seasons=tuple(str(s) for s in (raw.get("seasons") or [])),
                motifs=tuple(str(m) for m in (raw.get("motifs") or [])),
                milestone_ids=tuple(str(m) for m in (raw.get("milestone_ids") or [])),
                pack=str(raw.get("pack") or ""),
                min_warmth=float(raw.get("min_warmth", 0.0)),
                max_warmth=float(raw.get("max_warmth", 1.0)),
            )
        )
    templates: list[MessageTemplate] = []
    for raw in data.get("templates") or []:
        if not isinstance(raw, dict):
            continue
        templates.append(
            MessageTemplate(
                id=str(raw.get("id") or ""),
                intent=str(raw.get("intent") or ""),
                pattern=str(raw.get("pattern") or ""),
                placeholders=tuple(str(p) for p in (raw.get("placeholders") or [])),
                contexts=tuple(str(c) for c in (raw.get("contexts") or [])),
                cooldown_hours=float(raw.get("cooldown_hours") or 72),
                voice=str(raw.get("voice") or "observational"),
                mood=tuple(str(m) for m in (raw.get("mood") or [])),
                energy=str(raw.get("energy") or "low"),
                rarity=str(raw.get("rarity") or "common"),
                min_warmth=float(raw.get("min_warmth", 0.0)),
                max_warmth=float(raw.get("max_warmth", 1.0)),
            )
        )
    return MessageLibrary(
        schema_version=int(data.get("schema_version") or 1),
        messages=messages,
        templates=templates,
    )


def validate_library_dict(data: dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(data, dict):
        return False, "root_not_object"
    ids: set[str] = set()
    for section in ("messages", "templates"):
        for i, raw in enumerate(data.get(section) or []):
            if not isinstance(raw, dict):
                return False, f"{section}[{i}]_not_object"
            mid = str(raw.get("id") or "")
            if not mid:
                return False, f"{section}[{i}]_missing_id"
            if mid in ids:
                return False, f"duplicate_id:{mid}"
            ids.add(mid)
            intent = str(raw.get("intent") or "")
            if intent not in VALID_INTENTS:
                return False, f"invalid_intent:{intent}"
            if section == "messages":
                text = str(raw.get("text") or "")
                if len(text) > COMPANION_LINE_MAX_CHARS:
                    return False, f"text_too_long:{mid}"
                if not is_acceptable_companion_line(text):
                    return False, f"low_quality:{mid}"
                voice = str(raw.get("voice") or "observational")
                if voice not in VALID_VOICES:
                    return False, f"invalid_voice:{voice}"
            elif section == "templates":
                voice = str(raw.get("voice") or "observational")
                if voice not in VALID_VOICES:
                    return False, f"invalid_voice:{voice}"
    return True, ""


_library_cache: MessageLibrary | None = None


def get_message_library() -> MessageLibrary:
    global _library_cache
    if _library_cache is None:
        _library_cache = load_message_library()
    return _library_cache


def reload_message_library() -> MessageLibrary:
    global _library_cache
    _library_cache = load_message_library()
    return _library_cache
