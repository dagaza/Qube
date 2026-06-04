"""Variety store — cooldowns, intent/mood balance, semantic dedup."""

from __future__ import annotations

import json
import logging
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("Qube.CompanionVerbal")

_SCHEMA_VERSION = 2
_MAX_RECENT = 64
_MAX_FINGERPRINTS = 128
_JACCARD_DUP_THRESHOLD = 0.85


def _default_state_path() -> Path:
    return Path.home() / ".qube" / "companion" / "variety_state.json"


def normalize_line_fingerprint(line: str) -> str:
    cleaned = re.sub(r"\s+", " ", (line or "").strip().lower())
    cleaned = re.sub(r"[^\w\s]", "", cleaned)
    return cleaned


def _token_set(text: str) -> set[str]:
    return {t for t in text.split() if t}


def jaccard_similarity(a: str, b: str) -> float:
    sa, sb = _token_set(a), _token_set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


@dataclass
class VarietySnapshot:
    """Read-only view for thought/message selection."""

    recent_message_ids: tuple[str, ...] = ()
    recent_intents: tuple[str, ...] = ()
    recent_moods: tuple[str, ...] = ()
    recent_voices: tuple[str, ...] = ()
    recent_motifs: tuple[str, ...] = ()
    recent_fingerprints: tuple[str, ...] = ()
    message_last_used: dict[str, float] = field(default_factory=dict)
    now: float = 0.0


@dataclass
class VarietyStore:
    """Bounded persisted anti-repetition state."""

    message_last_used: dict[str, float] = field(default_factory=dict)
    recent_message_ids: deque[str] = field(default_factory=lambda: deque(maxlen=_MAX_RECENT))
    recent_intents: deque[str] = field(default_factory=lambda: deque(maxlen=_MAX_RECENT))
    recent_moods: deque[str] = field(default_factory=lambda: deque(maxlen=_MAX_RECENT))
    recent_voices: deque[str] = field(default_factory=lambda: deque(maxlen=_MAX_RECENT))
    recent_motifs: deque[str] = field(default_factory=lambda: deque(maxlen=_MAX_RECENT))
    recent_line_fingerprints: deque[str] = field(
        default_factory=lambda: deque(maxlen=_MAX_FINGERPRINTS)
    )

    def snapshot(self, *, now: float) -> VarietySnapshot:
        return VarietySnapshot(
            recent_message_ids=tuple(self.recent_message_ids),
            recent_intents=tuple(self.recent_intents),
            recent_moods=tuple(self.recent_moods),
            recent_voices=tuple(self.recent_voices),
            recent_motifs=tuple(self.recent_motifs),
            recent_fingerprints=tuple(self.recent_line_fingerprints),
            message_last_used=dict(self.message_last_used),
            now=now,
        )

    def is_message_on_cooldown(self, message_id: str, cooldown_hours: float, *, now: float) -> bool:
        last = self.message_last_used.get(message_id)
        if last is None:
            return False
        return (now - last) < (cooldown_hours * 3600.0)

    def intent_balance_penalty(self, intent: str, *, window: int = 3) -> float:
        recent = list(self.recent_intents)[-window:]
        if not recent:
            return 1.0
        count = recent.count(intent)
        if count >= 2:
            return 0.2
        if count >= 1:
            return 0.6
        return 1.0

    def mood_balance_penalty(self, mood: str, *, window: int = 3) -> float:
        recent = list(self.recent_moods)[-window:]
        if len(recent) >= window and all(m == mood for m in recent[-window:]):
            return 0.3
        return 1.0

    def is_semantic_duplicate(self, line: str) -> bool:
        fp = normalize_line_fingerprint(line)
        if not fp:
            return True
        for existing in self.recent_line_fingerprints:
            if jaccard_similarity(fp, existing) >= _JACCARD_DUP_THRESHOLD:
                return True
        return False

    def should_veto_intent(self, intent: str, *, window: int = 3) -> bool:
        recent = list(self.recent_intents)[-window:]
        return len(recent) >= window and all(i == intent for i in recent[-window:])

    def record_emission(
        self,
        *,
        message_id: str,
        intent: str,
        mood: str,
        line: str,
        now: float,
        voice: str = "",
        motifs: tuple[str, ...] = (),
    ) -> None:
        if message_id:
            self.message_last_used[message_id] = now
            self.recent_message_ids.append(message_id)
        if intent:
            self.recent_intents.append(intent)
        if mood:
            self.recent_moods.append(mood)
        if voice:
            self.recent_voices.append(voice)
        for m in motifs:
            if m:
                self.recent_motifs.append(m)
        fp = normalize_line_fingerprint(line)
        if fp:
            self.recent_line_fingerprints.append(fp)
        self._prune_old_cooldowns(now)

    def _prune_old_cooldowns(self, now: float, max_age_hours: float = 8760) -> None:
        cutoff = now - (max_age_hours * 3600.0)
        stale = [k for k, v in self.message_last_used.items() if v < cutoff]
        for k in stale:
            del self.message_last_used[k]

    def to_dict(self) -> dict:
        return {
            "schema_version": _SCHEMA_VERSION,
            "message_last_used": self.message_last_used,
            "recent_message_ids": list(self.recent_message_ids),
            "recent_intents": list(self.recent_intents),
            "recent_moods": list(self.recent_moods),
            "recent_voices": list(self.recent_voices),
            "recent_motifs": list(self.recent_motifs),
            "recent_line_fingerprints": list(self.recent_line_fingerprints),
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> VarietyStore:
        if not data:
            return cls()
        store = cls()
        raw_used = data.get("message_last_used") or {}
        if isinstance(raw_used, dict):
            for k, v in raw_used.items():
                try:
                    store.message_last_used[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue
        for key, attr in (
            ("recent_message_ids", store.recent_message_ids),
            ("recent_intents", store.recent_intents),
            ("recent_moods", store.recent_moods),
            ("recent_voices", store.recent_voices),
            ("recent_motifs", store.recent_motifs),
            ("recent_line_fingerprints", store.recent_line_fingerprints),
        ):
            items = data.get(key) or []
            if isinstance(items, list):
                for item in items[-attr.maxlen :]:
                    attr.append(str(item))
        return store

    @classmethod
    def load(cls, path: Path | None = None) -> VarietyStore:
        p = path or _default_state_path()
        if not p.is_file():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return cls.from_dict(data)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("[CompanionCognition] variety_state load failed: %s", e)
        return cls()

    def save(self, path: Path | None = None) -> None:
        p = path or _default_state_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


_global_store: VarietyStore | None = None


def get_variety_store() -> VarietyStore:
    global _global_store
    if _global_store is None:
        _global_store = VarietyStore.load()
    return _global_store


def persist_variety_store() -> None:
    get_variety_store().save()
