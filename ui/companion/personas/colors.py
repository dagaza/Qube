"""Shared activity color pairs for companion personas."""

from __future__ import annotations

from core.assistant_activity import AssistantActivity

ACTIVITY_COLORS: dict[AssistantActivity, tuple[str, str]] = {
    AssistantActivity.ASSISTANT_OFF: ("#64748b", "#475569"),
    AssistantActivity.IDLE_LISTEN: ("#89b4fa", "#b4befe"),
    AssistantActivity.CAPTURING: ("#f38ba8", "#fab387"),
    AssistantActivity.WORKING: ("#74c7ec", "#89dceb"),
    AssistantActivity.SPEAKING: ("#a6e3a1", "#94e2d5"),
    AssistantActivity.NEEDS_ATTENTION: ("#f9e2af", "#f5c842"),
    AssistantActivity.ERROR: ("#f38ba8", "#eba0ac"),
    AssistantActivity.BACKGROUND_BUSY: ("#cba6f7", "#b4a0fa"),
}
