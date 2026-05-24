"""Assistant presence state — shared by status bubble, tray icon, and notifications."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AssistantActivity(str, Enum):
    """Canonical assistant activity for tray + status UI."""

    ASSISTANT_OFF = "assistant_off"
    IDLE_LISTEN = "idle_listen"
    CAPTURING = "capturing"
    WORKING = "working"  # STT / thinking / generating / transcribing
    SPEAKING = "speaking"
    NEEDS_ATTENTION = "needs_attention"
    ERROR = "error"
    BACKGROUND_BUSY = "background_busy"


# Maps legacy status-bubble property names to AssistantActivity.
_BUBBLE_STATE_TO_ACTIVITY: dict[str, AssistantActivity] = {
    "idle": AssistantActivity.IDLE_LISTEN,
    "recording": AssistantActivity.CAPTURING,
    "speaking": AssistantActivity.SPEAKING,
    "thinking": AssistantActivity.WORKING,
    "needs_model": AssistantActivity.NEEDS_ATTENTION,
}


_ACTIVITY_TO_BUBBLE_STATE: dict[AssistantActivity, str] = {
    AssistantActivity.IDLE_LISTEN: "idle",
    AssistantActivity.CAPTURING: "recording",
    AssistantActivity.SPEAKING: "speaking",
    AssistantActivity.WORKING: "thinking",
    AssistantActivity.NEEDS_ATTENTION: "needs_model",
    AssistantActivity.ASSISTANT_OFF: "idle",
    AssistantActivity.ERROR: "needs_model",
    AssistantActivity.BACKGROUND_BUSY: "thinking",
}


@dataclass(frozen=True)
class ActivityTransition:
    """Result of reducing a worker status string into UI + tray state."""

    activity: AssistantActivity
    bubble_state: str
    display_text: str
    blocked: bool = False


def activity_from_bubble_state(state: str | None) -> AssistantActivity:
    return _BUBBLE_STATE_TO_ACTIVITY.get(str(state or "idle"), AssistantActivity.IDLE_LISTEN)


def bubble_state_for_activity(activity: AssistantActivity) -> str:
    return _ACTIVITY_TO_BUBBLE_STATE.get(activity, "idle")


def tray_tooltip_for_activity(activity: AssistantActivity, *, voice_paused: bool = False) -> str:
    if voice_paused or activity == AssistantActivity.ASSISTANT_OFF:
        return "Qube — Assistant paused"
    tips = {
        AssistantActivity.IDLE_LISTEN: "Qube — Listening",
        AssistantActivity.CAPTURING: "Qube — Listening to you…",
        AssistantActivity.WORKING: "Qube — Working on your request…",
        AssistantActivity.SPEAKING: "Qube — Speaking",
        AssistantActivity.NEEDS_ATTENTION: "Qube — Needs your attention",
        AssistantActivity.ERROR: "Qube — Something went wrong",
        AssistantActivity.BACKGROUND_BUSY: "Qube — Working in background",
    }
    return tips.get(activity, "Qube")


def menu_status_line(activity: AssistantActivity, *, voice_paused: bool = False) -> str:
    if voice_paused or activity == AssistantActivity.ASSISTANT_OFF:
        return "Assistant paused"
    lines = {
        AssistantActivity.IDLE_LISTEN: "Listening",
        AssistantActivity.CAPTURING: "Listening to you…",
        AssistantActivity.WORKING: "Working on your request…",
        AssistantActivity.SPEAKING: "Speaking",
        AssistantActivity.NEEDS_ATTENTION: "Needs attention",
        AssistantActivity.ERROR: "Something went wrong",
        AssistantActivity.BACKGROUND_BUSY: "Working in background",
    }
    return lines.get(activity, "Idle")


class AssistantActivityReducer:
    """Priority gate for worker status strings (extracted from MainWindow.update_status)."""

    def __init__(self) -> None:
        self._bubble_state = "idle"
        self._forced_activity: AssistantActivity | None = None

    @property
    def bubble_state(self) -> str:
        return self._bubble_state

    @property
    def activity(self) -> AssistantActivity:
        if self._forced_activity is not None:
            return self._forced_activity
        return activity_from_bubble_state(self._bubble_state)

    def set_forced_activity(self, activity: AssistantActivity | None) -> None:
        self._forced_activity = activity

    def set_voice_paused(self, paused: bool) -> None:
        if paused:
            self._forced_activity = AssistantActivity.ASSISTANT_OFF
        elif self._forced_activity == AssistantActivity.ASSISTANT_OFF:
            self._forced_activity = None

    def set_background_busy(self, busy: bool) -> None:
        if busy:
            self._forced_activity = AssistantActivity.BACKGROUND_BUSY
        elif self._forced_activity == AssistantActivity.BACKGROUND_BUSY:
            self._forced_activity = None

    def reduce(self, message: str, *, force: bool = False) -> ActivityTransition:
        msg_upper = message.upper().strip()

        if "MIC ERROR" in msg_upper or "VOICE INPUT DEACTIVATED" in msg_upper:
            new_bubble = "idle"
            if "MIC ERROR" in msg_upper:
                activity = AssistantActivity.NEEDS_ATTENTION
            elif self._forced_activity == AssistantActivity.ASSISTANT_OFF:
                activity = AssistantActivity.ASSISTANT_OFF
            else:
                activity = AssistantActivity.ASSISTANT_OFF if "DEACTIVATED" in msg_upper else AssistantActivity.IDLE_LISTEN
        elif any(k in msg_upper for k in ("RECORDING", "LISTENING")):
            new_bubble = "recording"
            activity = AssistantActivity.CAPTURING
        elif "SPEAKING" in msg_upper:
            new_bubble = "speaking"
            activity = AssistantActivity.SPEAKING
        elif msg_upper == "LOAD A MODEL":
            new_bubble = "needs_model"
            activity = AssistantActivity.NEEDS_ATTENTION
        elif any(k in msg_upper for k in ("THINKING", "GENERATING", "SYNTHESIZING", "TRANSCRIBING", "SEARCHING")):
            new_bubble = "thinking"
            activity = AssistantActivity.WORKING
        elif "INGESTING" in msg_upper:
            new_bubble = "thinking"
            activity = AssistantActivity.BACKGROUND_BUSY
            self._forced_activity = AssistantActivity.BACKGROUND_BUSY
        else:
            new_bubble = "idle"
            activity = AssistantActivity.IDLE_LISTEN

        current_state = self._bubble_state

        # Block stray Idle from the always-on mic listener while the assistant
        # is capturing, thinking, or speaking (audio_worker emits Idle every loop).
        if new_bubble == "idle" and current_state in ("recording", "thinking", "speaking"):
            if not force and msg_upper != "VOICE CAPTURE IDLE":
                return ActivityTransition(
                    activity=self.activity,
                    bubble_state=current_state,
                    display_text=self._format_display(msg_upper, new_bubble),
                    blocked=True,
                )

        self._bubble_state = new_bubble
        if self._forced_activity not in (
            AssistantActivity.ASSISTANT_OFF,
            AssistantActivity.BACKGROUND_BUSY,
        ):
            self._forced_activity = None

        if self._forced_activity == AssistantActivity.ASSISTANT_OFF:
            activity = AssistantActivity.ASSISTANT_OFF
        elif self._forced_activity == AssistantActivity.BACKGROUND_BUSY and new_bubble == "idle":
            activity = AssistantActivity.BACKGROUND_BUSY

        display = self._format_display(msg_upper, new_bubble)
        return ActivityTransition(
            activity=activity,
            bubble_state=new_bubble,
            display_text=display,
            blocked=False,
        )

    @staticmethod
    def _format_display(msg_upper: str, bubble_state: str) -> str:
        if msg_upper == "VOICE CAPTURE IDLE":
            return " IDLE"
        if bubble_state == "needs_model":
            return "Load a Model"
        return msg_upper
