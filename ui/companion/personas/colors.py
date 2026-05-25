"""Shared activity color pairs for companion personas."""

from __future__ import annotations

from core.assistant_activity import AssistantActivity
from core.companion_idle_color import CompanionIdleColor, idle_color_pair

ACTIVITY_COLORS: dict[AssistantActivity, tuple[str, str]] = {
    AssistantActivity.ASSISTANT_OFF: ("#64748b", "#475569"),
    AssistantActivity.CAPTURING: ("#f38ba8", "#fab387"),
    AssistantActivity.WORKING: ("#74c7ec", "#89dceb"),
    AssistantActivity.SPEAKING: ("#a6e3a1", "#94e2d5"),
    AssistantActivity.NEEDS_ATTENTION: ("#f9e2af", "#f5c842"),
    AssistantActivity.ERROR: ("#f38ba8", "#eba0ac"),
    AssistantActivity.BACKGROUND_BUSY: ("#cba6f7", "#b4a0fa"),
}


def activity_color_pair(
    activity: AssistantActivity,
    idle_color: CompanionIdleColor | str | None = None,
) -> tuple[str, str]:
    """Resolve primary/secondary hex for a companion activity."""
    if activity == AssistantActivity.IDLE_LISTEN:
        return idle_color_pair(idle_color)
    return ACTIVITY_COLORS.get(activity, idle_color_pair(idle_color))
