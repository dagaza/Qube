"""
Conversation-inferred user preferences at ``~/.qube/user_profile.json``.

Explicit user-controlled defaults live in ``settings.json`` (``qube.profile.*``).
This file holds tentative assistant-inferred presentation preferences with
provenance and confidence.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Optional

logger = logging.getLogger("Qube.UserProfile")

MAX_INFERRED_KEYS = 64


def default_user_profile_path() -> str:
    return os.path.join(os.path.expanduser("~"), ".qube", "user_profile.json")


class UserProfileStore:
    """Load/save inferred presentation preferences."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path or default_user_profile_path()
        self._lock = threading.RLock()
        self._data: dict[str, Any] = {"inferred_preferences": {}}
        self._load()

    def _load(self) -> None:
        with self._lock:
            if not os.path.isfile(self.path):
                return
            try:
                with open(self.path, encoding="utf-8") as fh:
                    raw = json.load(fh)
                if isinstance(raw, dict):
                    inferred = raw.get("inferred_preferences")
                    if isinstance(inferred, dict):
                        self._data["inferred_preferences"] = inferred
            except Exception as e:
                logger.warning("[UserProfile] load failed: %s", e)

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2, sort_keys=True)

    def get_inferred_preferences(self) -> dict[str, dict]:
        with self._lock:
            raw = self._data.get("inferred_preferences") or {}
            return dict(raw) if isinstance(raw, dict) else {}

    def set_inferred(
        self,
        key: str,
        value: str,
        *,
        confidence: float = 0.85,
        source: str = "conversation",
    ) -> None:
        """Upsert one inferred preference entry."""
        k = str(key or "").strip()
        v = str(value or "").strip()
        if not k or not v:
            return
        entry = {
            "value": v,
            "confidence": max(0.0, min(1.0, float(confidence))),
            "source": str(source or "conversation"),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with self._lock:
            prefs = self._data.setdefault("inferred_preferences", {})
            if not isinstance(prefs, dict):
                prefs = {}
                self._data["inferred_preferences"] = prefs
            prefs[k] = entry
            if len(prefs) > MAX_INFERRED_KEYS:
                ordered = sorted(
                    prefs.items(),
                    key=lambda item: str((item[1] or {}).get("updated_at") or ""),
                )
                for drop_key, _ in ordered[: len(prefs) - MAX_INFERRED_KEYS]:
                    prefs.pop(drop_key, None)
            self._save()

    def remove_inferred(self, key: str) -> None:
        k = str(key or "").strip()
        if not k:
            return
        with self._lock:
            prefs = self._data.get("inferred_preferences") or {}
            if isinstance(prefs, dict) and k in prefs:
                prefs.pop(k, None)
                self._save()

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._data))


_store: UserProfileStore | None = None
_store_lock = threading.Lock()


def get_user_profile_store() -> UserProfileStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = UserProfileStore()
        return _store


__all__ = [
    "UserProfileStore",
    "default_user_profile_path",
    "get_user_profile_store",
    "MAX_INFERRED_KEYS",
]
