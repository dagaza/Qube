"""Shared notification payload for in-app toasts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AppNotificationRequest:
    title: str
    body: str
    action_label: str | None = None
    action_id: str | None = None
    auto_dismiss_ms: int = 0
    severity: str = "info"
    category: str = "system"
    event_id: str = ""
    dedupe_key: str | None = None
