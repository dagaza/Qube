"""Ring buffer of recent notifications persisted under ~/.qube."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from core import app_settings

logger = logging.getLogger("Qube.Notifications")

_MAX_ITEMS = 100
_HISTORY_PATH = Path.home() / ".qube" / "notification_history.json"


@dataclass
class HistoryEntry:
    event_id: str
    title: str
    body: str
    severity: str
    category: str
    timestamp: float
    dismissed: bool = False


class NotificationHistoryStore:
    def __init__(self) -> None:
        self._entries: list[HistoryEntry] = []
        self._dirty = False
        self._load()

    def append(
        self,
        *,
        event_id: str,
        title: str,
        body: str,
        severity: str,
        category: str,
        timestamp: float | None = None,
    ) -> None:
        if not app_settings.get_notifications_keep_history():
            return
        entry = HistoryEntry(
            event_id=event_id,
            title=title,
            body=body,
            severity=severity,
            category=category,
            timestamp=timestamp or time.time(),
        )
        self._entries.insert(0, entry)
        if len(self._entries) > _MAX_ITEMS:
            self._entries = self._entries[:_MAX_ITEMS]
        self._dirty = True

    def recent(self, limit: int = 5) -> list[HistoryEntry]:
        return list(self._entries[:limit])

    def clear(self) -> None:
        self._entries.clear()
        self._dirty = True
        self.flush()

    def flush(self) -> None:
        if not self._dirty:
            return
        if not app_settings.get_notifications_keep_history():
            if _HISTORY_PATH.is_file():
                try:
                    _HISTORY_PATH.unlink()
                except OSError:
                    pass
            self._dirty = False
            return
        try:
            _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = {"entries": [asdict(e) for e in self._entries]}
            _HISTORY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self._dirty = False
        except OSError as exc:
            logger.warning("Failed to persist notification history: %s", exc)

    def _load(self) -> None:
        if not _HISTORY_PATH.is_file():
            return
        try:
            raw = json.loads(_HISTORY_PATH.read_text(encoding="utf-8"))
            items = raw.get("entries") if isinstance(raw, dict) else []
            if not isinstance(items, list):
                return
            for item in items[:_MAX_ITEMS]:
                if not isinstance(item, dict):
                    continue
                self._entries.append(
                    HistoryEntry(
                        event_id=str(item.get("event_id", "")),
                        title=str(item.get("title", "")),
                        body=str(item.get("body", "")),
                        severity=str(item.get("severity", "info")),
                        category=str(item.get("category", "system")),
                        timestamp=float(item.get("timestamp", 0)),
                        dismissed=bool(item.get("dismissed", False)),
                    )
                )
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("Failed to load notification history: %s", exc)
