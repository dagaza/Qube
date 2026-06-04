"""Companion personality vector — LLM-independent tone model."""

from __future__ import annotations

import json
from dataclasses import dataclass

from core.companion_verbal_traits import (
    CompanionVerbalTraitPreset,
    normalize_companion_verbal_trait,
)

PRESET_VECTORS: dict[CompanionVerbalTraitPreset, dict[str, float]] = {
    CompanionVerbalTraitPreset.NEUTRAL: {
        "warmth": 0.5,
        "humor": 0.2,
        "curiosity": 0.4,
        "playfulness": 0.3,
        "verbosity": 0.2,
    },
    CompanionVerbalTraitPreset.WARM: {
        "warmth": 0.85,
        "humor": 0.3,
        "curiosity": 0.5,
        "playfulness": 0.4,
        "verbosity": 0.25,
    },
    CompanionVerbalTraitPreset.WITTY: {
        "warmth": 0.6,
        "humor": 0.75,
        "curiosity": 0.6,
        "playfulness": 0.7,
        "verbosity": 0.3,
    },
    CompanionVerbalTraitPreset.DRY: {
        "warmth": 0.4,
        "humor": 0.65,
        "curiosity": 0.4,
        "playfulness": 0.35,
        "verbosity": 0.15,
    },
    CompanionVerbalTraitPreset.SARCASTIC: {
        "warmth": 0.45,
        "humor": 0.7,
        "curiosity": 0.45,
        "playfulness": 0.65,
        "verbosity": 0.2,
    },
}


@dataclass(frozen=True)
class CompanionPersonalityVector:
    warmth: float = 0.5
    humor: float = 0.2
    curiosity: float = 0.4
    playfulness: float = 0.3
    verbosity: float = 0.2

    def clamped(self) -> CompanionPersonalityVector:
        def _c(v: float) -> float:
            return max(0.0, min(1.0, float(v)))

        return CompanionPersonalityVector(
            warmth=_c(self.warmth),
            humor=_c(self.humor),
            curiosity=_c(self.curiosity),
            playfulness=_c(self.playfulness),
            verbosity=_c(self.verbosity),
        )

    def to_dict(self) -> dict[str, float]:
        c = self.clamped()
        return {
            "warmth": c.warmth,
            "humor": c.humor,
            "curiosity": c.curiosity,
            "playfulness": c.playfulness,
            "verbosity": c.verbosity,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> CompanionPersonalityVector:
        if not data:
            return cls()
        try:
            return cls(
                warmth=float(data.get("warmth", 0.5)),
                humor=float(data.get("humor", 0.2)),
                curiosity=float(data.get("curiosity", 0.4)),
                playfulness=float(data.get("playfulness", 0.3)),
                verbosity=float(data.get("verbosity", 0.2)),
            ).clamped()
        except (TypeError, ValueError):
            return cls()

    def summary_for_prompt(self) -> str:
        c = self.clamped()
        parts = []
        if c.warmth >= 0.65:
            parts.append("warm")
        if c.humor >= 0.55:
            parts.append("lightly humorous")
        if c.curiosity >= 0.55:
            parts.append("curious")
        if c.playfulness >= 0.55:
            parts.append("playful")
        if not parts:
            parts.append("calm and concise")
        return ", ".join(parts)


def vector_from_trait_preset(preset: CompanionVerbalTraitPreset | str) -> CompanionPersonalityVector:
    key = normalize_companion_verbal_trait(preset)
    return CompanionPersonalityVector.from_dict(PRESET_VECTORS.get(key, PRESET_VECTORS[CompanionVerbalTraitPreset.NEUTRAL]))


def load_personality_vector() -> CompanionPersonalityVector:
    """Load v2 vector from settings, falling back to trait preset migration."""
    from core import app_settings

    raw = app_settings.get_companion_personality_v2_json()
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and data:
                return CompanionPersonalityVector.from_dict(data)
        except json.JSONDecodeError:
            pass
    return vector_from_trait_preset(app_settings.get_companion_verbal_trait_preset())
