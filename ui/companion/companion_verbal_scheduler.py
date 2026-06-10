"""Companion verbal commentary scheduler — idle timer + event hooks."""

from __future__ import annotations

import logging
import time

from PyQt6.QtCore import QObject, QTimer

from core import app_settings
from core.assistant_presence import AssistantPresenceService
from core.companion_cognition.ambient_context import (
    build_ambient_context,
    session_start_ambient_context,
)
from core.companion_cognition.orchestrator import (
    CompanionCognitionOrchestrator,
    cognition_v2_enabled,
)
from core.companion_cognition.personality import load_personality_vector
from core.companion_cognition.types import CognitionProcessResult
from core.companion_cognition.usage_counters import (
    load_counters,
    mark_usage_pattern_emitted,
    record_ingest_event,
    session_count_tier,
    should_emit_usage_pattern,
)
from core.companion_verbal_policy import (
    CompanionVerbalGateContext,
    CompanionVerbalRateLimiter,
    CompanionVerbalTrigger,
    record_emitted,
    should_emit_event,
    should_emit_idle,
    should_show_companion_line,
)

logger = logging.getLogger("Qube.CompanionVerbal")

_IDLE_CHECK_MS = 30_000


class CompanionVerbalScheduler(QObject):
    """Evaluates policy and runs companion cognition or legacy sidecar lines."""

    def __init__(
        self,
        controller,
        presence: AssistantPresenceService,
        sidecar_client,
        sidecar_worker=None,
        *,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._presence = presence
        self._sidecar = sidecar_client
        self._sidecar_worker = sidecar_worker
        self._limiter = CompanionVerbalRateLimiter()
        self._orchestrator = CompanionCognitionOrchestrator()
        self._pending_sidecar: dict[str, CognitionProcessResult] = {}
        self._startup_caption_shown = False
        self._idle_timer = QTimer(self)
        self._idle_timer.setInterval(_IDLE_CHECK_MS)
        self._idle_timer.timeout.connect(self._on_idle_tick)
        if sidecar_worker is not None and hasattr(sidecar_worker, "companion_line_ready"):
            sidecar_worker.companion_line_ready.connect(self.on_companion_line_ready)

    def start(self) -> None:
        if not self._idle_timer.isActive():
            self._idle_timer.start()

    def stop(self) -> None:
        self._idle_timer.stop()

    def refresh_settings(self) -> None:
        """Settings are read at gate evaluation time; limiter state is preserved."""

    def _gate_context(self) -> CompanionVerbalGateContext:
        ctrl = self._controller
        snap = self._presence.snapshot()
        main_visible = False
        main_minimized = False
        if hasattr(ctrl, "_main_visible"):
            main_visible = bool(ctrl._main_visible())
        if hasattr(ctrl, "_main_minimized"):
            main_minimized = bool(ctrl._main_minimized())
        return CompanionVerbalGateContext(
            snapshot=snap,
            companion_visible=bool(getattr(ctrl, "is_visible_for_policy", False)),
            idle_since=getattr(ctrl, "_idle_since", None),
            snooze_until=float(getattr(ctrl, "_snooze_until", 0.0) or 0.0),
            main_window_visible=main_visible,
            main_window_minimized=main_minimized,
            companion_user_visible=bool(getattr(ctrl, "_user_visible", True)),
            fullscreen_detected=bool(getattr(ctrl, "_fullscreen_detected", False)),
        )

    def _sidecar_available(self) -> bool:
        if self._sidecar is None:
            return False
        return bool(getattr(self._sidecar, "available", False))

    def _request_line(self, trigger: str, **extra) -> None:
        if cognition_v2_enabled():
            self._process_cognition(trigger, **extra)
            return
        if self._sidecar is None:
            return
        payload = {"trigger": trigger, **extra}
        request = getattr(self._sidecar, "request_companion_line", None)
        if not callable(request):
            return
        if not request(payload):
            logger.info(
                "[CompanionVerbal] enqueue skipped trigger=%s (sidecar unavailable or reloading)",
                trigger,
            )

    def _ambient_for_gate(self, ctx: CompanionVerbalGateContext, *, session_start: bool = False):
        now_ts = ctx.now if ctx.now else time.time()
        personality = load_personality_vector()
        kwargs = dict(
            now_ts=now_ts,
            personality=personality,
            seasonal_enabled=app_settings.get_companion_seasonal_enabled(),
            hemisphere=app_settings.get_companion_seasonal_hemisphere(),
            motifs_enabled=app_settings.get_companion_motifs_enabled(),
            mood_drift_enabled=app_settings.get_companion_mood_drift_enabled(),
        )
        if session_start:
            return session_start_ambient_context(**kwargs)
        return build_ambient_context(**kwargs)

    def _process_cognition(
        self,
        trigger: str,
        *,
        ambient=None,
        session_start: bool = False,
        **extra,
    ) -> None:
        ctx = self._gate_context()
        if ambient is None:
            ambient = self._ambient_for_gate(ctx, session_start=session_start)
        result = self._orchestrator.process_legacy(
            trigger,
            ctx,
            sidecar_available=self._sidecar_available(),
            ambient=ambient,
            **extra,
        )
        if result.skip_reason:
            logger.debug(
                "[CompanionCognition] skipped trigger=%s reason=%s",
                trigger,
                result.skip_reason,
            )
            return
        if result.local is not None:
            self._emit_local_line(result, trigger)
            if str(trigger or "").strip().lower() == "startup":
                self._startup_caption_shown = True
            return
        if result.sidecar is not None:
            request = getattr(self._sidecar, "request_companion_line", None)
            if not callable(request):
                return
            trig_key = str(trigger or "idle").strip().lower()
            self._pending_sidecar[trig_key] = result
            if not request(result.sidecar.to_payload()):
                self._pending_sidecar.pop(trig_key, None)
                logger.info(
                    "[CompanionVerbal] enqueue skipped trigger=%s (sidecar unavailable)",
                    trigger,
                )

    def _emit_local_line(self, result, trigger: str) -> None:
        local = result.local
        if local is None:
            return
        trig = str(trigger or local.trigger or "idle")
        if trig != "test" and not should_show_companion_line(trig, self._presence.snapshot()):
            return
        window = getattr(self._controller, "window", None)
        if window is None:
            return
        window.show_banter_caption(local.line)
        record_emitted(trig, self._limiter)
        CompanionCognitionOrchestrator.record_successful_emission(result)

    def _on_idle_tick(self) -> None:
        ctx = self._gate_context()
        if not should_emit_idle(ctx, self._limiter):
            return
        self._request_line(CompanionVerbalTrigger.IDLE.value)

    def on_ingestion_complete(self, file_count: int) -> None:
        record_ingest_event()
        ctx = self._gate_context()
        if not should_emit_event(CompanionVerbalTrigger.INGEST_COMPLETE, ctx, self._limiter):
            return
        self._request_line(
            CompanionVerbalTrigger.INGEST_COMPLETE.value,
            file_count=max(1, int(file_count)),
        )

    def on_model_download_complete(self, basename: str) -> None:
        ctx = self._gate_context()
        if not should_emit_event(CompanionVerbalTrigger.DOWNLOAD_COMPLETE, ctx, self._limiter):
            return
        self._request_line(
            CompanionVerbalTrigger.DOWNLOAD_COMPLETE.value,
            basename=str(basename or ""),
        )

    def on_model_loaded(self, basename: str) -> None:
        if not cognition_v2_enabled():
            return
        ctx = self._gate_context()
        if not should_emit_event(CompanionVerbalTrigger.MODEL_LOADED, ctx, self._limiter):
            return
        self._process_cognition(
            CompanionVerbalTrigger.MODEL_LOADED.value,
            basename=str(basename or ""),
        )

    def on_startup(self, session_index: int = 1) -> None:
        if not cognition_v2_enabled():
            return
        ctx = self._gate_context()
        if not should_emit_event(CompanionVerbalTrigger.STARTUP, ctx, self._limiter):
            self._maybe_emit_usage_pattern(ctx)
            return
        self._process_cognition(
            CompanionVerbalTrigger.STARTUP.value,
            session_index=max(1, int(session_index)),
            session_start=True,
        )
        self._maybe_emit_usage_pattern(ctx)

    def _maybe_emit_usage_pattern(self, ctx: CompanionVerbalGateContext) -> None:
        counters = load_counters()
        if not should_emit_usage_pattern(counters):
            return
        if not should_emit_event(CompanionVerbalTrigger.USAGE_PATTERN, ctx, self._limiter):
            return
        self._process_cognition(
            CompanionVerbalTrigger.USAGE_PATTERN.value,
            days_active=int(counters.get("days_active") or 0),
            session_count_tier=session_count_tier(int(counters.get("session_count") or 0)),
        )
        mark_usage_pattern_emitted()

    def on_milestone(self, milestone_id: str) -> None:
        if not cognition_v2_enabled():
            return
        if self._startup_caption_shown:
            return
        ctx = self._gate_context()
        if not should_emit_event(CompanionVerbalTrigger.MILESTONE, ctx, self._limiter):
            return
        self._process_cognition(
            CompanionVerbalTrigger.MILESTONE.value,
            milestone_id=str(milestone_id),
        )

    def process_test_preview(self) -> None:
        """Settings test — runs cognition pipeline for settings_preview."""
        ctx = self._gate_context()
        result = self._orchestrator.process_legacy(
            "test",
            ctx,
            sidecar_available=self._sidecar_available(),
        )
        if result.local is not None:
            return result.local.line, result.local.kind
        if result.sidecar is not None:
            preview = getattr(self._sidecar, "preview_companion_line", None)
            if callable(preview):
                from core import app_settings

                sr = preview(**result.sidecar.to_payload())
                if sr.ok and sr.text:
                    return sr.text, str((sr.parsed or {}).get("kind") or "idle_quip")
        return None, None

    def on_companion_line_ready(self, line: str, kind: str, trigger: str) -> None:
        if str(trigger or "").strip().lower() == "test":
            return
        snap = self._presence.snapshot()
        if not should_show_companion_line(trigger, snap):
            return
        window = getattr(self._controller, "window", None)
        if window is None:
            return
        window.show_banter_caption(line)
        record_emitted(trigger, self._limiter)
        trig_key = str(trigger or "idle").strip().lower()
        pending = self._pending_sidecar.pop(trig_key, None)
        if pending is not None:
            CompanionCognitionOrchestrator.record_successful_emission(pending, line=line)
        elif cognition_v2_enabled():
            CompanionCognitionOrchestrator.record_successful_emission(
                CognitionProcessResult(
                    emission_intent="",
                    emission_mood="",
                    emission_voice="",
                ),
                line=line,
            )
