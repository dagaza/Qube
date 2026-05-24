"""Pure policy rules for notification delivery channels."""

from __future__ import annotations

from dataclasses import dataclass

from core import app_settings
from core.notification_types import NotificationEvent, NotificationSeverity


@dataclass(frozen=True)
class NotificationDeliveryPlan:
    show_in_app: bool
    show_os: bool
    play_sound: bool
    reason: str = ""


def _category_enabled(category: str) -> bool:
    getters = {
        "voice": app_settings.get_notifications_category_voice,
        "turn": app_settings.get_notifications_category_turn_complete,
        "tool": app_settings.get_notifications_category_tools,
        "background": app_settings.get_notifications_category_background,
        "memory": app_settings.get_notifications_category_memory,
        "update": app_settings.get_notifications_category_updates,
        "system": lambda: True,
    }
    fn = getters.get(category)
    return fn() if fn else True


def plan_delivery(
    event: NotificationEvent,
    *,
    window_visible: bool,
    window_focused: bool,
    tts_playing: bool = False,
) -> NotificationDeliveryPlan:
    if not app_settings.get_notifications_enabled():
        return NotificationDeliveryPlan(False, False, False, "disabled")

    if not _category_enabled(event.category):
        return NotificationDeliveryPlan(False, False, False, "category_off")

    dnd = app_settings.get_notifications_dnd()
    is_critical = event.severity == NotificationSeverity.CRITICAL

    if dnd and not is_critical:
        return NotificationDeliveryPlan(False, False, False, "dnd")

    suppress_focused = app_settings.get_notifications_suppress_when_focused()
    is_low_priority = event.severity in (NotificationSeverity.INFO, NotificationSeverity.SUCCESS)

    if window_focused and suppress_focused and is_low_priority:
        return NotificationDeliveryPlan(False, False, False, "focused_suppressed")

    if tts_playing and is_low_priority:
        return NotificationDeliveryPlan(False, False, False, "speaking_suppressed")

    show_in_app = event.severity in (
        NotificationSeverity.WARNING,
        NotificationSeverity.ERROR,
        NotificationSeverity.CRITICAL,
        NotificationSeverity.SUCCESS,
    ) or (not window_focused)

    os_when_hidden = app_settings.get_notifications_os_when_hidden()
    hidden = not window_visible or not window_focused
    show_os = hidden and os_when_hidden and event.severity != NotificationSeverity.INFO

    if hidden and event.severity in (
        NotificationSeverity.SUCCESS,
        NotificationSeverity.WARNING,
        NotificationSeverity.ERROR,
        NotificationSeverity.CRITICAL,
    ):
        show_os = os_when_hidden

    if not show_in_app and not show_os:
        if event.severity == NotificationSeverity.INFO:
            return NotificationDeliveryPlan(False, False, False, "info_tray_only")
        return NotificationDeliveryPlan(False, False, False, "no_channel")

    play_sound = (
        app_settings.get_notifications_sound_enabled()
        and event.severity in (NotificationSeverity.ERROR, NotificationSeverity.CRITICAL)
        and not dnd
    )

    return NotificationDeliveryPlan(show_in_app, show_os, play_sound, "deliver")
