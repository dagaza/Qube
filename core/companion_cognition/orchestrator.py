"""Companion Cognition orchestrator — full pipeline entry point."""

from __future__ import annotations

import logging
import time

from core import app_settings
from core.companion_cognition.ambient_context import build_ambient_context
from core.companion_cognition.capability import (
    allows_full_generate,
    allows_sidecar_rewrite,
    max_expression_level,
    resolve_expression_capability,
)
from core.companion_cognition.expression import (
    build_expression_plan,
    try_local_expression,
)
from core.companion_cognition.message_library import get_message_library
from core.companion_cognition.mood_drift import load_mood_state, nudge_mood_after_emission
from core.companion_cognition.motifs import record_motif_emission
from core.companion_cognition.observations import observe, trigger_event_from_legacy
from core.companion_cognition.personality import load_personality_vector
from core.companion_cognition.thoughts import think
from core.companion_cognition.types import (
    CognitionProcessResult,
    CompanionTriggerEvent,
    ExpressionLevel,
    SidecarExpressionRequest,
)
from core.companion_cognition.usage_counters import record_caption_emission
from core.companion_cognition.variety import get_variety_store, persist_variety_store
from core.companion_line_quality import is_acceptable_companion_line
from core.companion_verbal_policy import CompanionVerbalGateContext

logger = logging.getLogger("Qube.CompanionVerbal")


class CompanionCognitionOrchestrator:
    """Observation → thought → expression pipeline."""

    def process(
        self,
        event: CompanionTriggerEvent,
        ctx: CompanionVerbalGateContext,
        *,
        sidecar_available: bool = True,
        ambient=None,
    ) -> CognitionProcessResult:
        obs = observe(event, ctx)
        if obs is None:
            return CognitionProcessResult(skip_reason="observation_rejected")

        now_ts = ctx.now if ctx.now else time.time()
        personality = load_personality_vector()
        if ambient is None:
            ambient = build_ambient_context(
                now_ts=now_ts,
                personality=personality,
                seasonal_enabled=app_settings.get_companion_seasonal_enabled(),
                hemisphere=app_settings.get_companion_seasonal_hemisphere(),
                motifs_enabled=app_settings.get_companion_motifs_enabled(),
                mood_drift_enabled=app_settings.get_companion_mood_drift_enabled(),
            )

        store = get_variety_store()
        variety = store.snapshot(now=now_ts)

        thought = think(obs, personality, variety, ambient)
        if thought is None:
            logger.info(
                "[CompanionCognition] skip reason=no_thought obs=%s",
                obs.type,
            )
            return CognitionProcessResult(skip_reason="no_thought")

        trigger = obs.trigger_source or event.source
        tier = resolve_expression_capability()
        max_level = max_expression_level(tier)

        local = try_local_expression(
            thought,
            obs,
            trigger,
            variety,
            personality,
            ambient=ambient,
            library=get_message_library(),
        )
        if local is not None and local.level.value <= max_level.value:
            logger.info(
                "[CompanionCognition] level=%s intent=%s msg_id=%s trigger=%s",
                local.level.value,
                local.intent,
                local.message_id,
                trigger,
            )
            return CognitionProcessResult(
                local=local,
                emission_message_id=local.message_id,
                emission_intent=local.intent,
                emission_mood=local.mood,
                emission_voice=local.voice,
                emission_motifs=local.motifs,
                emission_ambient_mood=thought.ambient_mood,
            )

        if not sidecar_available:
            if local is not None:
                return CognitionProcessResult(
                    local=local,
                    emission_message_id=local.message_id,
                    emission_intent=local.intent,
                    emission_mood=local.mood,
                    emission_voice=local.voice,
                    emission_motifs=local.motifs,
                    emission_ambient_mood=thought.ambient_mood,
                )
            return CognitionProcessResult(skip_reason="sidecar_unavailable")

        plan = None
        if allows_sidecar_rewrite(tier) and max_level >= ExpressionLevel.SIDECAR_REWRITE:
            plan = build_expression_plan(
                thought,
                obs,
                trigger,
                variety,
                personality,
                ExpressionLevel.SIDECAR_REWRITE,
                ambient=ambient,
            )
            if plan and plan.seed_line:
                level = ExpressionLevel.SIDECAR_REWRITE
            else:
                plan = None

        if plan is None and allows_full_generate(tier, trigger=trigger):
            plan = build_expression_plan(
                thought,
                obs,
                trigger,
                variety,
                personality,
                ExpressionLevel.FULL_GENERATE,
                ambient=ambient,
            )
            level = ExpressionLevel.FULL_GENERATE
        elif plan is not None:
            level = ExpressionLevel.SIDECAR_REWRITE
        else:
            if local is not None:
                return CognitionProcessResult(
                    local=local,
                    emission_message_id=local.message_id,
                    emission_intent=local.intent,
                    emission_mood=local.mood,
                    emission_voice=local.voice,
                    emission_motifs=local.motifs,
                    emission_ambient_mood=thought.ambient_mood,
                )
            return CognitionProcessResult(skip_reason="no_expression_plan")

        thought_dict = {
            "intent": thought.intent,
            "mood": thought.mood,
            "energy": thought.energy,
            "voice": thought.voice,
            "ambient_mood": thought.ambient_mood,
            "slots": dict(thought.slots),
            "kind": plan.kind if plan else "idle_quip",
        }
        obs_dict = {"type": obs.type, "facts": dict(obs.facts)}

        req = SidecarExpressionRequest(
            trigger=trigger,
            expression_level=int(level.value),
            thought=thought_dict,
            observation=obs_dict,
            seed_line=plan.seed_line if plan else "",
            file_count=obs.facts.get("file_count") if obs.type == "library_update_completed" else None,
            basename=obs.facts.get("basename"),
        )

        logger.info(
            "[CompanionCognition] sidecar_enqueue level=%s intent=%s trigger=%s",
            level.value,
            thought.intent,
            trigger,
        )
        return CognitionProcessResult(
            sidecar=req,
            emission_message_id=plan.message_id if plan else "",
            emission_intent=thought.intent,
            emission_mood=thought.mood,
            emission_voice=thought.voice,
            emission_ambient_mood=thought.ambient_mood,
        )

    def process_legacy(
        self,
        trigger: str,
        ctx: CompanionVerbalGateContext,
        *,
        sidecar_available: bool = True,
        ambient=None,
        **payload,
    ) -> CognitionProcessResult:
        event = trigger_event_from_legacy(trigger, ts=ctx.now, **payload)
        return self.process(event, ctx, sidecar_available=sidecar_available, ambient=ambient)

    @staticmethod
    def record_successful_emission(
        result: CognitionProcessResult,
        *,
        line: str = "",
        now: float | None = None,
    ) -> None:
        ts = now if now is not None else time.time()
        local = result.local
        message_id = result.emission_message_id
        intent = result.emission_intent
        mood = result.emission_mood
        voice = result.emission_voice
        motifs = result.emission_motifs
        caption_line = line or (local.line if local else "")

        if not caption_line and local is None:
            return

        if local is not None and not line:
            caption_line = local.line
            message_id = local.message_id or message_id
            intent = local.intent or intent
            mood = local.mood or mood
            voice = local.voice or voice
            motifs = local.motifs or motifs

        store = get_variety_store()
        store.record_emission(
            message_id=message_id,
            intent=intent,
            mood=mood,
            line=caption_line,
            now=ts,
            voice=voice,
            motifs=motifs,
        )
        persist_variety_store()

        record_caption_emission()

        active_motif = None
        if app_settings.get_companion_motifs_enabled():
            from core.companion_cognition.motifs import load_motif_state

            active_motif = load_motif_state().active_motif
        record_motif_emission(active_motif, motifs, ts)

        if app_settings.get_companion_mood_drift_enabled() and intent:
            mood_state = load_mood_state()
            nudge_mood_after_emission(mood_state, intent=intent, now_ts=ts)


def cognition_v2_enabled() -> bool:
    return app_settings.get_companion_cognition_v2_enabled()
