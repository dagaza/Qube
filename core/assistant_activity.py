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
    "listening": AssistantActivity.CAPTURING,
    "recording": AssistantActivity.CAPTURING,  # legacy QSS alias
    "speaking": AssistantActivity.SPEAKING,
    "thinking": AssistantActivity.WORKING,
    "writing": AssistantActivity.WORKING,
    "needs_model": AssistantActivity.NEEDS_ATTENTION,
}


_ACTIVITY_TO_BUBBLE_STATE: dict[AssistantActivity, str] = {
    AssistantActivity.IDLE_LISTEN: "idle",
    AssistantActivity.CAPTURING: "listening",
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


def bubble_state_for_activity(
    activity: AssistantActivity,
    *,
    voice_output_muted: bool = False,
) -> str:
    if activity == AssistantActivity.WORKING:
        return "writing" if voice_output_muted else "thinking"
    return _ACTIVITY_TO_BUBBLE_STATE.get(activity, "idle")


def user_presence_label(
    activity: AssistantActivity,
    *,
    voice_output_muted: bool = False,
) -> str:
    """User-facing presence line (status bubble, tray, companion)."""
    if activity == AssistantActivity.NEEDS_ATTENTION:
        return "Needs attention"
    if activity == AssistantActivity.CAPTURING:
        return "Listening"
    if activity == AssistantActivity.SPEAKING:
        return "Speaking"
    if activity in (AssistantActivity.WORKING, AssistantActivity.BACKGROUND_BUSY):
        return "Writing" if voice_output_muted else "Thinking"
    return "Idle"


def tray_tooltip_for_activity(
    activity: AssistantActivity,
    *,
    voice_output_muted: bool = False,
) -> str:
    label = user_presence_label(activity, voice_output_muted=voice_output_muted)
    return f"Qube — {label}"


def menu_status_line(
    activity: AssistantActivity,
    *,
    voice_output_muted: bool = False,
) -> str:
    return user_presence_label(activity, voice_output_muted=voice_output_muted)


def _is_voice_capture_message(msg_upper: str) -> bool:
    return msg_upper == "LISTENING" or "RECORDING" in msg_upper


def _is_native_engine_status(msg_upper: str) -> bool:
    """Model load/unload and engine routing — not user-visible assistant activity."""
    return (
        msg_upper.startswith("NATIVE ENGINE")
        or msg_upper.startswith("NATIVE MODEL")
        or msg_upper.startswith("LOADING NATIVE")
        or msg_upper.startswith("UNLOADING NATIVE")
        or msg_upper.startswith("ENGINE:")
    )


def _is_assistant_working_message(msg_upper: str) -> bool:
    """Canonical in-flight turn statuses only — not arbitrary substrings in filenames."""
    if msg_upper.startswith(("THINKING", "TRANSCRIBING", "GENERATING", "SYNTHESIZING")):
        return True
    return "SEARCHING" in msg_upper and "WEB" in msg_upper


class AssistantActivityReducer:
    """Priority gate for worker status strings (extracted from MainWindow.update_status)."""

    def __init__(self) -> None:
        self._bubble_state = "idle"
        self._forced_activity: AssistantActivity | None = None
        self._voice_output_muted = False

    @property
    def bubble_state(self) -> str:
        return self._bubble_state

    @property
    def activity(self) -> AssistantActivity:
        if self._forced_activity is not None:
            return self._forced_activity
        return activity_from_bubble_state(self._bubble_state)

    def set_voice_output_muted(self, muted: bool) -> None:
        self._voice_output_muted = bool(muted)

    def set_forced_activity(self, activity: AssistantActivity | None) -> None:
        self._forced_activity = activity

    def set_background_busy(self, busy: bool) -> None:
        if busy:
            self._forced_activity = AssistantActivity.BACKGROUND_BUSY
        elif self._forced_activity == AssistantActivity.BACKGROUND_BUSY:
            self._forced_activity = None

    def reduce(self, message: str, *, force: bool = False) -> ActivityTransition:
        msg_upper = message.upper().strip()

        if "MIC ERROR" in msg_upper:
            new_bubble = "idle"
            activity = AssistantActivity.NEEDS_ATTENTION
        elif "VOICE INPUT DEACTIVATED" in msg_upper:
            new_bubble = "idle"
            activity = AssistantActivity.IDLE_LISTEN
        elif _is_voice_capture_message(msg_upper):
            new_bubble = "listening"
            activity = AssistantActivity.CAPTURING
        elif "SPEAKING" in msg_upper:
            new_bubble = "speaking"
            activity = AssistantActivity.SPEAKING
        elif msg_upper == "LOAD A MODEL":
            new_bubble = "needs_model"
            activity = AssistantActivity.NEEDS_ATTENTION
        elif _is_native_engine_status(msg_upper):
            new_bubble = "idle"
            activity = AssistantActivity.IDLE_LISTEN
        elif _is_assistant_working_message(msg_upper):
            activity = AssistantActivity.WORKING
            new_bubble = (
                "writing" if self._voice_output_muted else "thinking"
            )
        elif "INGESTING" in msg_upper or "REPROCESSING" in msg_upper:
            new_bubble = "writing" if self._voice_output_muted else "thinking"
            activity = AssistantActivity.BACKGROUND_BUSY
            self._forced_activity = AssistantActivity.BACKGROUND_BUSY
        else:
            new_bubble = "idle"
            activity = AssistantActivity.IDLE_LISTEN

        current_state = self._bubble_state

        # Block stray Idle from the always-on mic listener while the assistant
        # is capturing, thinking, writing, or speaking.
        if new_bubble == "idle" and current_state in (
            "listening",
            "recording",
            "thinking",
            "writing",
            "speaking",
        ):
            if not force and msg_upper != "VOICE CAPTURE IDLE":
                return ActivityTransition(
                    activity=self.activity,
                    bubble_state=current_state,
                    display_text=self._format_display(
                        message, current_state, self.activity
                    ),
                    blocked=True,
                )

        self._bubble_state = new_bubble
        if self._forced_activity != AssistantActivity.BACKGROUND_BUSY:
            self._forced_activity = None

        if (
            self._forced_activity == AssistantActivity.BACKGROUND_BUSY
            and new_bubble == "idle"
        ):
            activity = AssistantActivity.BACKGROUND_BUSY

        display = self._format_display(message, new_bubble, activity)
        return ActivityTransition(
            activity=activity,
            bubble_state=new_bubble,
            display_text=display,
            blocked=False,
        )

    def _format_display(
        self,
        message: str,
        bubble_state: str,
        activity: AssistantActivity,
    ) -> str:
        msg_upper = message.upper().strip()
        if msg_upper == "VOICE CAPTURE IDLE":
            return " Idle"
        if bubble_state == "needs_model" or msg_upper == "LOAD A MODEL":
            return "Load a Model"
        if "MIC ERROR" in msg_upper:
            return " Voice input unavailable"
        if activity == AssistantActivity.BACKGROUND_BUSY:
            detail = message.strip()
            if detail and msg_upper not in ("IDLE",):
                return f" {detail}"
        label = user_presence_label(
            activity, voice_output_muted=self._voice_output_muted
        )
        return f" {label}"
