"""ThoughtEngine — deterministic observation → internal intention."""

from __future__ import annotations

from dataclasses import dataclass

from core.companion_cognition.ambient_context import AmbientContext
from core.companion_cognition.mood_drift import (
    ambient_mood_intent_bias,
    ambient_mood_voice_nudge,
)
from core.companion_cognition.personality import CompanionPersonalityVector
from core.companion_cognition.types import CompanionObservation, CompanionThought
from core.companion_cognition.variety import VarietySnapshot


@dataclass(frozen=True)
class _IntentCandidate:
    intent: str
    mood: str
    energy: str
    weight_fn: str  # personality attribute name
    tone_constraints: tuple[str, ...] = ()


_OBSERVATION_RULES: dict[str, tuple[_IntentCandidate, ...]] = {
    "quiet_period": (
        _IntentCandidate("atmosphere", "calm", "low", "warmth"),
        _IntentCandidate("self_expression", "playful", "low", "playfulness"),
        _IntentCandidate("reflection", "neutral", "low", "curiosity"),
        _IntentCandidate("humor", "playful", "low", "humor"),
        _IntentCandidate("curiosity", "curious", "low", "curiosity"),
        _IntentCandidate("wellbeing", "calm", "low", "warmth"),
        _IntentCandidate("fact", "neutral", "low", "curiosity"),
    ),
    "library_update_completed": (
        _IntentCandidate("acknowledge_effort", "warm", "low", "warmth"),
    ),
    "model_download_completed": (
        _IntentCandidate("celebration", "warm", "low", "warmth"),
        _IntentCandidate("acknowledge_effort", "neutral", "low", "warmth"),
    ),
    "model_ready": (
        _IntentCandidate("celebration", "warm", "low", "warmth"),
        _IntentCandidate("acknowledge_effort", "neutral", "low", "warmth"),
    ),
    "settings_preview": (
        _IntentCandidate("wellbeing", "warm", "low", "warmth"),
        _IntentCandidate("self_expression", "calm", "low", "warmth"),
    ),
    "companion_startup": (
        _IntentCandidate("wellbeing", "warm", "low", "warmth"),
        _IntentCandidate("atmosphere", "calm", "low", "warmth"),
    ),
    "focus_detected": (
        _IntentCandidate("wellbeing", "calm", "low", "warmth"),
        _IntentCandidate("atmosphere", "calm", "low", "warmth"),
    ),
    "system_resumed": (
        _IntentCandidate("wellbeing", "calm", "low", "warmth"),
        _IntentCandidate("atmosphere", "calm", "low", "warmth"),
    ),
    "usage_milestone": (
        _IntentCandidate("celebration", "warm", "medium", "playfulness"),
        _IntentCandidate("reflection", "warm", "low", "warmth"),
    ),
    "usage_pattern": (
        _IntentCandidate("reflection", "neutral", "low", "curiosity"),
        _IntentCandidate("wellbeing", "calm", "low", "warmth"),
    ),
}

_AMBIENT_TURN_MOOD: dict[str, str] = {
    "reflective": "neutral",
    "cozy": "warm",
    "curious": "curious",
    "playful": "playful",
    "observant": "neutral",
    "quiet": "calm",
}

_KIND_FOR_INTENT: dict[str, str] = {
    "acknowledge_effort": "ingest_ack",
    "celebration": "download_ack",
    "wellbeing": "idle_quip",
    "observation": "idle_quip",
    "self_expression": "idle_quip",
    "curiosity": "idle_quip",
    "atmosphere": "idle_quip",
    "humor": "idle_quip",
    "reflection": "idle_quip",
    "fact": "idle_quip",
}


def kind_for_intent(intent: str, observation_type: str) -> str:
    if observation_type == "library_update_completed":
        return "ingest_ack"
    if observation_type in ("model_download_completed", "model_ready"):
        return "download_ack"
    return _KIND_FOR_INTENT.get(intent, "idle_quip")


def derive_voice(
    intent: str,
    personality: CompanionPersonalityVector,
    *,
    ambient: AmbientContext | None = None,
) -> str:
    """Map intent + personality + ambient mood to a preferred expression voice."""
    p = personality.clamped()
    if intent == "humor":
        voice = "dry" if p.humor >= 0.65 else "playful"
    elif intent == "reflection":
        voice = "reflective"
    elif intent == "self_expression":
        voice = "playful" if p.playfulness >= 0.65 else ("cozy" if p.warmth >= 0.55 else "wry")
    elif intent == "atmosphere":
        voice = "observational"
    elif intent == "curiosity":
        voice = "curious"
    elif intent == "wellbeing":
        voice = "cozy" if p.warmth >= 0.55 else "observational"
    elif intent == "celebration":
        voice = "cozy" if p.warmth >= 0.6 else "playful"
    elif intent == "acknowledge_effort":
        voice = "dry" if p.humor >= 0.55 else ("cozy" if p.warmth >= 0.55 else "observational")
    elif intent == "fact":
        voice = "observational"
    else:
        voice = "observational"
    if ambient is not None:
        voice = ambient_mood_voice_nudge(ambient.ambient_mood.state, voice)
    return voice


def _score_candidate(c: _IntentCandidate, personality: CompanionPersonalityVector) -> float:
    base = getattr(personality, c.weight_fn, 0.5)
    if c.intent == "humor":
        base = (personality.humor + personality.playfulness) / 2.0
    if c.intent in ("celebration", "acknowledge_effort"):
        base = (base + personality.warmth) / 2.0
    if c.intent == "self_expression":
        base = (personality.playfulness + personality.warmth) / 2.0
    if c.intent == "reflection":
        base = (personality.curiosity + 0.4) / 1.4
    if c.intent == "atmosphere":
        base = (personality.warmth + 0.35) / 1.35
    return base


def _build_slots(obs: CompanionObservation) -> dict:
    slots: dict = {}
    facts = obs.facts or {}
    if "basename" in facts:
        slots["basename"] = facts["basename"]
    if "file_count" in facts:
        fc = int(facts["file_count"])
        slots["file_count"] = fc
        if fc == 1:
            slots["file_count_word"] = "one"
        elif fc == 2:
            slots["file_count_word"] = "two"
        elif fc <= 3:
            slots["file_count_word"] = "a few"
        else:
            slots["file_count_word"] = "several"
    if "milestone_id" in facts:
        slots["milestone_id"] = facts["milestone_id"]
    return slots


def think(
    obs: CompanionObservation,
    personality: CompanionPersonalityVector,
    variety: VarietySnapshot,
    ambient: AmbientContext | None = None,
) -> CompanionThought | None:
    """Select internal intention from observation + personality + variety + ambient."""
    rules = _OBSERVATION_RULES.get(obs.type)
    if not rules:
        return None

    ambient_state = ambient.ambient_mood.state if ambient else ""

    scored: list[tuple[float, _IntentCandidate]] = []
    for cand in rules:
        if variety.recent_intents and variety.recent_intents[-3:].count(cand.intent) >= 2:
            continue
        score = _score_candidate(cand, personality)
        if ambient_state:
            score *= ambient_mood_intent_bias(ambient_state, cand.intent)
        intent_pen = 1.0
        recent = list(variety.recent_intents)[-3:]
        if cand.intent in recent:
            intent_pen = 0.5
        scored.append((score * intent_pen, cand))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][1]

    if best.intent == "humor" and personality.humor < 0.45:
        for _, alt in scored[1:]:
            if alt.intent != "humor":
                best = alt
                break

    mood = best.mood
    if ambient_state and ambient_state in _AMBIENT_TURN_MOOD and obs.type != "settings_preview":
        mood = _AMBIENT_TURN_MOOD[ambient_state]
    elif personality.humor >= 0.6 and best.intent in ("wellbeing", "atmosphere") and obs.type == "quiet_period":
        if personality.playfulness >= 0.55:
            mood = "playful"

    energy = best.energy
    if personality.verbosity >= 0.4 and obs.type in ("usage_milestone",):
        energy = "medium"

    tone = list(best.tone_constraints)
    if obs.type in ("model_download_completed", "model_ready"):
        tone.append("no_model_jargon")

    voice = derive_voice(best.intent, personality, ambient=ambient)

    return CompanionThought(
        intent=best.intent,
        mood=mood,
        energy=energy,
        voice=voice,
        ambient_mood=ambient_state,
        tone_constraints=tuple(tone),
        slots=_build_slots(obs),
        observation_type=obs.type,
    )
