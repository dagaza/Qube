"""Canonical assistant presence — shared by tray, status bubble, and companion."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from PyQt6.QtCore import QObject, pyqtSignal

from core.assistant_activity import (
    ActivityTransition,
    AssistantActivity,
    AssistantActivityReducer,
    user_presence_label,
)
from core.platform.companion_capabilities import CompanionPlatformTier, detect_companion_platform_tier


class AssistantPhase(str, Enum):
    """Visual sub-phase within a canonical AssistantActivity."""

    VAD_ACTIVE = "vad_active"
    STT = "stt"
    LLM = "llm"
    ROUTING = "routing"
    MODEL_LOAD = "model_load"
    TTS_STREAM = "tts_stream"
    NO_MODEL = "no_model"
    PERMISSION = "permission"
    INGEST = "ingest"


@dataclass(frozen=True)
class AssistantPresenceSnapshot:
    """Immutable presence snapshot for companion and policy consumers."""

    activity: AssistantActivity
    phase: AssistantPhase | None
    display_text: str
    bubble_state: str
    voice_input_paused: bool
    voice_output_muted: bool
    dnd: bool
    background_busy: bool
    caption_text: str | None
    attention_required: bool
    platform_tier: CompanionPlatformTier
    audio_level: float = 0.0
    speech_level: float = 0.0


def phase_from_message(message: str, activity: AssistantActivity, bubble_state: str) -> AssistantPhase | None:
    """Derive a visual sub-phase from a worker status string."""
    msg_upper = message.upper().strip()

    if activity == AssistantActivity.CAPTURING and (
        _is_capture_phase_message(msg_upper)
    ):
        return AssistantPhase.VAD_ACTIVE
    if activity == AssistantActivity.SPEAKING:
        return AssistantPhase.TTS_STREAM
    if activity == AssistantActivity.NEEDS_ATTENTION:
        if bubble_state == "needs_model" or "LOAD A MODEL" in msg_upper:
            return AssistantPhase.NO_MODEL
        if "MIC ERROR" in msg_upper:
            return AssistantPhase.PERMISSION
        return AssistantPhase.PERMISSION
    if activity == AssistantActivity.BACKGROUND_BUSY:
        if "INGESTING" in msg_upper:
            return AssistantPhase.INGEST
        return None
    if activity != AssistantActivity.WORKING:
        return None

    if "TRANSCRIBING" in msg_upper:
        return AssistantPhase.STT
    if any(k in msg_upper for k in ("LOADING NATIVE", "LOADING MODEL", "UNLOADING")):
        return AssistantPhase.MODEL_LOAD
    if "SEARCHING" in msg_upper:
        return AssistantPhase.ROUTING
    if any(k in msg_upper for k in ("THINKING", "GENERATING", "SYNTHESIZING")):
        return AssistantPhase.LLM
    return AssistantPhase.LLM


def _is_capture_phase_message(msg_upper: str) -> bool:
    return msg_upper == "LISTENING" or "RECORDING" in msg_upper


def companion_status_caption(
    activity: AssistantActivity,
    phase: AssistantPhase | None,
    *,
    voice_output_muted: bool = False,
) -> str | None:
    """Short companion chip text — mirrors user_presence_label (never Idle; see companion UI)."""
    if activity in (AssistantActivity.NEEDS_ATTENTION, AssistantActivity.IDLE_LISTEN):
        return None
    return user_presence_label(activity, voice_output_muted=voice_output_muted)


class AssistantPresenceService(QObject):
    """Priority-gated presence reducer with Qt signal for downstream UI."""

    presence_changed = pyqtSignal(object)  # AssistantPresenceSnapshot

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._reducer = AssistantActivityReducer()
        self._voice_output_muted = False
        self._dnd = False
        self._caption_text: str | None = None
        self._audio_level = 0.0
        self._speech_level = 0.0
        self._last_snapshot: AssistantPresenceSnapshot | None = None
        self._platform_tier = detect_companion_platform_tier()

    @property
    def bubble_state(self) -> str:
        return self._reducer.bubble_state

    @property
    def activity(self) -> AssistantActivity:
        return self._reducer.activity

    def set_forced_activity(self, activity: AssistantActivity | None) -> None:
        self._reducer.set_forced_activity(activity)
        self._publish_from_current("")

    def set_voice_paused(self, paused: bool) -> None:
        self._reducer.set_voice_paused(paused)
        self._publish_from_current("")

    def set_background_busy(self, busy: bool) -> None:
        self._reducer.set_background_busy(busy)
        self._publish_from_current("")

    def set_voice_output_muted(self, muted: bool) -> None:
        self._voice_output_muted = muted
        self._reducer.set_voice_output_muted(muted)
        self._publish_from_current("")

    def set_dnd(self, enabled: bool) -> None:
        self._dnd = enabled
        self._publish_from_current("")

    def set_caption_text(self, text: str | None) -> None:
        self._caption_text = (text or "").strip() or None
        self._publish_from_current("")

    def set_audio_level(self, level: float) -> None:
        self._audio_level = max(0.0, min(1.0, float(level)))
        if self.activity == AssistantActivity.CAPTURING:
            self._publish_from_current("")

    def set_speech_level(self, level: float) -> None:
        self._speech_level = max(0.0, min(1.0, float(level)))

    def refresh_platform_tier(self) -> None:
        self._platform_tier = detect_companion_platform_tier()
        self._publish_from_current("")

    def reduce(self, message: str, *, force: bool = False) -> ActivityTransition:
        transition = self._reducer.reduce(message, force=force)
        if not transition.blocked:
            self._publish(transition, message)
        return transition

    def snapshot(self) -> AssistantPresenceSnapshot:
        if self._last_snapshot is not None:
            return self._last_snapshot
        activity = self.activity
        bubble = self.bubble_state
        return self._build_snapshot(
            ActivityTransition(
                activity=activity,
                bubble_state=bubble,
                display_text=self._reducer._format_display("", bubble, activity),
            ),
            "",
        )

    def _publish_from_current(self, message: str) -> None:
        activity = self.activity
        bubble = self.bubble_state
        transition = ActivityTransition(
            activity=activity,
            bubble_state=bubble,
            display_text=self._reducer._format_display("", bubble, activity),
        )
        self._publish(transition, message)

    def _publish(self, transition: ActivityTransition, message: str) -> None:
        snap = self._build_snapshot(transition, message)
        changed = snap != self._last_snapshot
        self._last_snapshot = snap
        if changed:
            self.presence_changed.emit(snap)

    def _build_snapshot(self, transition: ActivityTransition, message: str) -> AssistantPresenceSnapshot:
        activity = transition.activity
        phase = phase_from_message(message, activity, transition.bubble_state)
        voice_paused = activity == AssistantActivity.ASSISTANT_OFF
        background_busy = activity == AssistantActivity.BACKGROUND_BUSY
        attention = activity in (
            AssistantActivity.NEEDS_ATTENTION,
            AssistantActivity.ERROR,
        )
        caption = self._caption_text
        if caption is None:
            from core import app_settings

            if app_settings.get_companion_show_caption():
                caption = companion_status_caption(
                    activity,
                    phase,
                    voice_output_muted=self._voice_output_muted,
                )

        return AssistantPresenceSnapshot(
            activity=activity,
            phase=phase,
            display_text=transition.display_text,
            bubble_state=transition.bubble_state,
            voice_input_paused=voice_paused,
            voice_output_muted=self._voice_output_muted,
            dnd=self._dnd,
            background_busy=background_busy,
            caption_text=caption,
            attention_required=attention,
            platform_tier=self._platform_tier,
            audio_level=self._audio_level,
            speech_level=self._speech_level,
        )