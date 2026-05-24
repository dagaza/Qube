"""Companion persona identifiers and registry metadata."""

from __future__ import annotations

from enum import Enum


class CompanionPersonaId(str, Enum):
    SPHERE = "sphere"
    QUBE = "qube"


PERSONA_LABELS: dict[CompanionPersonaId, str] = {
    CompanionPersonaId.SPHERE: "Sphere",
    CompanionPersonaId.QUBE: "Qube",
}

PERSONA_DESCRIPTIONS: dict[CompanionPersonaId, str] = {
    CompanionPersonaId.SPHERE: "Organic, fluid presence with soft glow and warm energy.",
    CompanionPersonaId.QUBE: "Holographic living cube — structured, futuristic, premium.",
}

DEFAULT_COMPANION_PERSONA = CompanionPersonaId.QUBE


def normalize_companion_persona(value: str | CompanionPersonaId | None) -> CompanionPersonaId:
    if isinstance(value, CompanionPersonaId):
        return value
    raw = str(value or "").strip().lower()
    for persona in CompanionPersonaId:
        if persona.value == raw:
            return persona
    return DEFAULT_COMPANION_PERSONA
