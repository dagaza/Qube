"""Central notification dispatcher — main-thread only."""

from __future__ import annotations

import logging
import time
from typing import Callable

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from core.notification_history import NotificationHistoryStore
from core.notification_policy import plan_delivery
from core.notification_types import NotificationEvent

logger = logging.getLogger("Qube.Notifications")


class NotificationService(QObject):
    """Single entry point for tray, in-app, and OS notifications."""

    action_triggered = pyqtSignal(str, str)  # action_id, event_id
    notification_shown = pyqtSignal(object)  # NotificationEvent

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._history = NotificationHistoryStore()
        self._rate_limit_until: dict[str, float] = {}
        self._coalesce_latest: dict[str, NotificationEvent] = {}
        self._coalesce_timers: dict[str, QTimer] = {}
        self._pending_turn_complete: dict[str, NotificationEvent] = {}
        self._window_visible_fn: Callable[[], bool] = lambda: True
        self._window_focused_fn: Callable[[], bool] = lambda: True
        self._tts_playing_fn: Callable[[], bool] = lambda: False
        self._show_in_app_fn: Callable[[NotificationEvent], None] | None = None
        self._show_os_fn: Callable[[NotificationEvent], None] | None = None
        self._flush_timer = QTimer(self)
        self._flush_timer.setInterval(30_000)
        self._flush_timer.timeout.connect(self._history.flush)
        self._flush_timer.start()

    def set_window_state_providers(
        self,
        *,
        visible: Callable[[], bool],
        focused: Callable[[], bool],
        tts_playing: Callable[[], bool] | None = None,
    ) -> None:
        self._window_visible_fn = visible
        self._window_focused_fn = focused
        if tts_playing is not None:
            self._tts_playing_fn = tts_playing

    def set_show_handlers(
        self,
        *,
        in_app: Callable[[NotificationEvent], None],
        os_notify: Callable[[NotificationEvent], None] | None = None,
    ) -> None:
        self._show_in_app_fn = in_app
        self._show_os_fn = os_notify

    @property
    def history(self) -> NotificationHistoryStore:
        return self._history

    def cancel_turn_complete(self, session_id: str) -> None:
        self._pending_turn_complete.pop(session_id, None)

    def queue_turn_complete(self, event: NotificationEvent, *, wait_for_tts: bool) -> None:
        session_id = event.dedupe_key or event.event_id
        if wait_for_tts:
            key = session_id.replace("turn_complete:", "") if session_id.startswith("turn_complete:") else session_id
            self._pending_turn_complete[key] = event
            return
        self.emit(event)

    def flush_turn_complete(self, session_id: str) -> None:
        event = self._pending_turn_complete.pop(session_id, None)
        if event is not None:
            self.emit(event)

    def emit(self, event: NotificationEvent) -> None:
        if event.rate_limit_key and event.rate_limit_sec > 0:
            until = self._rate_limit_until.get(event.rate_limit_key, 0.0)
            if time.time() < until:
                logger.debug("Rate-limited notification: %s", event.rate_limit_key)
                return
            self._rate_limit_until[event.rate_limit_key] = time.time() + event.rate_limit_sec

        if event.coalesce_group:
            self._schedule_coalesced(event)
            return

        self._deliver(event)

    def _schedule_coalesced(self, event: NotificationEvent) -> None:
        group = event.coalesce_group or event.event_id
        self._coalesce_latest[group] = event
        if group in self._coalesce_timers:
            return
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.setInterval(800)

        def _fire() -> None:
            latest = self._coalesce_latest.pop(group, None)
            self._coalesce_timers.pop(group, None)
            if latest is not None:
                self._deliver(latest)

        timer.timeout.connect(_fire)
        self._coalesce_timers[group] = timer
        timer.start()

    def _deliver(self, event: NotificationEvent) -> None:
        plan = plan_delivery(
            event,
            window_visible=self._window_visible_fn(),
            window_focused=self._window_focused_fn(),
            tts_playing=self._tts_playing_fn(),
        )
        if plan.reason in ("disabled", "category_off", "dnd", "focused_suppressed", "speaking_suppressed", "info_tray_only", "no_channel"):
            logger.debug("Notification suppressed (%s): %s", plan.reason, event.title)
            return

        self._history.append(
            event_id=event.event_id,
            title=event.title,
            body=event.body,
            severity=event.severity.value,
            category=event.category,
            timestamp=event.timestamp,
        )

        if plan.show_in_app and self._show_in_app_fn is not None:
            self._show_in_app_fn(event)

        if plan.show_os and self._show_os_fn is not None:
            self._show_os_fn(event)

        self.notification_shown.emit(event)

    def shutdown(self) -> None:
        for timer in self._coalesce_timers.values():
            timer.stop()
        self._coalesce_timers.clear()
        self._history.flush()
