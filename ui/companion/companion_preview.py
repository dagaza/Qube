"""Live companion preview widget for Settings."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget

from core import app_settings
from core.assistant_activity import AssistantActivity, user_presence_label
from core.assistant_presence import AssistantPhase, AssistantPresenceSnapshot
from core.companion_personas import CompanionPersonaId, DEFAULT_COMPANION_PERSONA, normalize_companion_persona
from core.companion_verbal_prompts import truncate_companion_caption
from core.platform.companion_capabilities import CompanionPlatformTier
from ui.companion.anim_engine import CompanionAnimEngine, FRAME_DT
from ui.companion.persona_context import CompanionPaintContext
from ui.companion.personas.base import get_persona_renderer
from ui.companion.personas.colors import activity_color_pair

_PREVIEW_DIMENSION = 280
# Keep draw size stable so extra canvas space becomes visible margin (avoids scaling the cube up).
_PREVIEW_BODY_RADIUS = 38.0


def _demo_snapshot(
    activity: AssistantActivity,
    *,
    phase: AssistantPhase | None = None,
    audio_level: float = 0.0,
    speech_level: float = 0.0,
) -> AssistantPresenceSnapshot:
    return AssistantPresenceSnapshot(
        activity=activity,
        phase=phase,
        display_text="",
        presence_label=user_presence_label(activity),
        bubble_state="idle",
        voice_input_paused=False,
        voice_output_muted=False,
        dnd=False,
        background_busy=False,
        caption_text=None,
        attention_required=False,
        platform_tier=CompanionPlatformTier.FULL,
        audio_level=audio_level,
        speech_level=speech_level,
    )


class CompanionPreviewWidget(QFrame):
    """Animated preview of the selected companion persona."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("CompanionPreviewFrame")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setFixedSize(_PREVIEW_DIMENSION, _PREVIEW_DIMENSION)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)

        self._is_dark = True
        self._persona_id = DEFAULT_COMPANION_PERSONA
        self._renderer = get_persona_renderer(self._persona_id)
        self._demo_activity = AssistantActivity.IDLE_LISTEN
        self._demo_phase: AssistantPhase | None = None

        self._anim = CompanionAnimEngine()
        self._anim.set_snapshot(_demo_snapshot(AssistantActivity.IDLE_LISTEN))

        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        self._sample_caption = QLabel("")
        self._sample_caption.setObjectName("CompanionPreviewCaption")
        self._sample_caption.setWordWrap(True)
        self._sample_caption.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        self._sample_caption.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Minimum,
        )
        self._sample_caption.hide()
        layout.addWidget(self._sample_caption)

        self._sample_caption_timer = QTimer(self)
        self._sample_caption_timer.setSingleShot(True)
        self._sample_caption_timer.timeout.connect(self._clear_sample_caption)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._timer.isActive():
            self._timer.start()

    def hideEvent(self, event) -> None:
        self._timer.stop()
        super().hideEvent(event)

    def apply_theme(self, is_dark: bool) -> None:
        self._is_dark = is_dark
        self.update()

    def set_persona(self, persona_id: CompanionPersonaId | str) -> None:
        persona_id = normalize_companion_persona(persona_id)
        self._persona_id = persona_id
        self._renderer = get_persona_renderer(persona_id)
        if not self._timer.isActive():
            self._timer.start()
        self.repaint()

    def show_sample_caption(self, text: str, ttl_sec: float = 12.0) -> None:
        line = (text or "").strip()
        if not line:
            return
        line = truncate_companion_caption(line, 72)
        self._sample_caption.setText(line)
        inner_w = max(120, min(240, self.width() - 48))
        self._sample_caption.setFixedWidth(inner_w)
        label_h = self._sample_caption.heightForWidth(inner_w)
        if label_h > 0:
            self._sample_caption.setFixedHeight(label_h + 4)
        self._sample_caption.show()
        self._sample_caption_timer.stop()
        self._sample_caption_timer.start(int(max(1000, ttl_sec * 1000)))
        self.update()

    def _clear_sample_caption(self) -> None:
        self._sample_caption.clear()
        self._sample_caption.setMinimumHeight(0)
        self._sample_caption.setMaximumHeight(16777215)
        self._sample_caption.hide()
        self.update()

    def set_demo_activity(self, activity: AssistantActivity) -> None:
        self._demo_activity = activity
        phase = AssistantPhase.STT if activity == AssistantActivity.WORKING else None
        self._demo_phase = phase
        audio = 0.35 if activity == AssistantActivity.CAPTURING else 0.0
        speech = 0.55 if activity == AssistantActivity.SPEAKING else 0.0
        self._anim.set_snapshot(
            _demo_snapshot(activity, phase=phase, audio_level=audio, speech_level=speech)
        )
        if activity == AssistantActivity.SPEAKING:
            self._anim.set_speech_level(0.55)
        self.update()

    def _on_tick(self) -> None:
        self._anim.tick(FRAME_DT)
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        cx = self.width() / 2
        cy = self.height() / 2
        body_radius = _PREVIEW_BODY_RADIUS

        primary_hex, secondary_hex = activity_color_pair(
            self._demo_activity,
            app_settings.get_companion_idle_color(),
            is_dark=self._is_dark,
        )
        ctx = CompanionPaintContext(
            activity=self._demo_activity,
            phase=self._demo_phase,
            primary=QColor(primary_hex),
            secondary=QColor(secondary_hex),
            center_x=cx,
            center_y=cy,
            body_radius=body_radius,
            breathe=self._anim.breathe_scale(),
            float_offset_y=self._anim.float_offset_y(),
            opacity=1.0,
            anim_time=self._anim.anim_time,
            rotation=self._anim.rotation,
            reduced_motion=self._anim.reduced_motion,
            is_dark=self._is_dark,
            input_level=self._anim.input_level,
            speech_level_smooth=self._anim.speech_level_smooth,
            wave_bars=tuple(self._anim.wave_bars),
            ripple_rings=tuple(self._anim.ripple_rings),
            notify_pulse=self._anim.notify_pulse,
            persona_blend=1.0,
        )

        self._renderer.paint(painter, ctx)
        painter.end()
