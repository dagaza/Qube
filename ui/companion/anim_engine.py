"""Shared animation engine for companion personas (audio-reactive hooks)."""

from __future__ import annotations

import math

from core.assistant_activity import AssistantActivity
from core.assistant_presence import AssistantPhase, AssistantPresenceSnapshot

WAVE_BAR_COUNT = 36
FRAME_DT = 1.0 / 30.0


class CompanionAnimEngine:
    """Centralized tick/update logic shared by all companion personas."""

    def __init__(self) -> None:
        self.anim_time = 0.0
        self.rotation = 0.0
        self.input_level = 0.0
        self.speech_level = 0.0
        self.speech_level_smooth = 0.0
        self.notify_pulse = 0.0
        self.wave_bars = [0.08] * WAVE_BAR_COUNT
        self.ripple_rings: list[tuple[float, float]] = []
        self.last_ripple_emit = 0.0
        self.reduced_motion = False
        self._snapshot: AssistantPresenceSnapshot | None = None

    def set_snapshot(self, snapshot: AssistantPresenceSnapshot | None) -> None:
        self._snapshot = snapshot
        if snapshot is not None:
            self.input_level = snapshot.audio_level
            self.speech_level = snapshot.speech_level

    def set_speech_level(self, level: float) -> None:
        self.speech_level = max(0.0, min(1.0, float(level)))

    def pulse_notification(self) -> None:
        self.notify_pulse = 1.0

    def reset_motion(self) -> None:
        self.anim_time = 0.0
        self.rotation = 0.0

    def activity(self) -> AssistantActivity:
        if self._snapshot is None:
            return AssistantActivity.IDLE_LISTEN
        return self._snapshot.activity

    def phase(self) -> AssistantPhase | None:
        if self._snapshot is None:
            return None
        return self._snapshot.phase

    def tick(self, dt: float | None = None) -> bool:
        """Advance animation state. Returns True if a repaint is recommended."""
        step = FRAME_DT if dt is None else dt
        if self.reduced_motion:
            step = 0.5

        activity = self.activity()
        phase = self.phase()

        if not self.reduced_motion:
            self.anim_time += step
            if activity == AssistantActivity.WORKING:
                spin = 0.035 if phase == AssistantPhase.STT else 0.022
                if phase == AssistantPhase.MODEL_LOAD:
                    spin = 0.015
                self.rotation = (self.rotation + spin) % (2 * math.pi)
            elif activity == AssistantActivity.SPEAKING:
                self.rotation = (self.rotation + 0.004) % (2 * math.pi)
            elif activity == AssistantActivity.IDLE_LISTEN:
                self.rotation = (self.rotation + 0.004) % (2 * math.pi)

        target_speech = self.speech_level
        if activity == AssistantActivity.SPEAKING and target_speech < 0.05:
            target_speech = self._synthetic_speech_level()
        self.speech_level_smooth += (target_speech - self.speech_level_smooth) * 0.35

        if activity == AssistantActivity.SPEAKING:
            self._push_wave_bar(self.speech_level_smooth)
        elif activity == AssistantActivity.CAPTURING and self.input_level > 0.04:
            self._push_wave_bar(self.input_level * 0.85)
            if not self.reduced_motion and (self.anim_time - self.last_ripple_emit) > 0.15:
                self.ripple_rings.append((0.0, self.input_level))
                self.last_ripple_emit = self.anim_time
        else:
            for i, h in enumerate(self.wave_bars):
                self.wave_bars[i] = max(0.06, h * 0.92)

        if not self.reduced_motion:
            updated: list[tuple[float, float]] = []
            for age, strength in self.ripple_rings:
                age += step * 1.6
                if age < 1.2:
                    updated.append((age, strength))
            self.ripple_rings = updated

        if self.notify_pulse > 0:
            self.notify_pulse = max(0.0, self.notify_pulse - step * 2.5)

        return activity in (
            AssistantActivity.CAPTURING,
            AssistantActivity.WORKING,
            AssistantActivity.SPEAKING,
            AssistantActivity.NEEDS_ATTENTION,
            AssistantActivity.BACKGROUND_BUSY,
            AssistantActivity.IDLE_LISTEN,
        ) or self.notify_pulse > 0

    def breathe_scale(self) -> float:
        activity = self.activity()
        if self.reduced_motion:
            return 1.0
        if activity == AssistantActivity.IDLE_LISTEN:
            return 1.0 + 0.065 * math.sin(self.anim_time * 1.8)
        if activity == AssistantActivity.SPEAKING:
            return (
                1.0
                + 0.04 * self.speech_level_smooth
                + 0.025 * math.sin(self.anim_time * 6.0)
            )
        if activity == AssistantActivity.WORKING:
            return 1.0 + 0.03 * math.sin(self.anim_time * 3.5)
        if activity == AssistantActivity.CAPTURING:
            return 1.0 + 0.05 * self.input_level
        return 1.0

    def float_offset_y(self) -> float:
        if self.reduced_motion:
            return 0.0
        return math.sin(self.anim_time * 1.4) * 3.5

    def _synthetic_speech_level(self) -> float:
        t = self.anim_time
        return min(
            1.0,
            0.22
            + 0.35 * abs(math.sin(t * 9.7)) * abs(math.sin(t * 3.4))
            + 0.12 * abs(math.sin(t * 14.2)),
        )

    def _push_wave_bar(self, level: float) -> None:
        jitter = 0.78 + 0.22 * math.sin(self.anim_time * 13.7 + len(self.wave_bars))
        self.wave_bars.pop(0)
        self.wave_bars.append(min(1.0, max(0.05, level * jitter)))
