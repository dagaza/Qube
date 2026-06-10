"""Companion verbal personality presets (tone fragments for sidecar prompts)."""

from __future__ import annotations

from enum import Enum


class CompanionVerbalTraitPreset(str, Enum):
    NEUTRAL = "neutral"
    WARM = "warm"
    WITTY = "witty"
    DRY = "dry"
    SARCASTIC = "sarcastic"


DEFAULT_COMPANION_VERBAL_TRAIT = CompanionVerbalTraitPreset.NEUTRAL

TRAIT_LABELS: dict[CompanionVerbalTraitPreset, str] = {
    CompanionVerbalTraitPreset.NEUTRAL: "Neutral",
    CompanionVerbalTraitPreset.WARM: "Warm",
    CompanionVerbalTraitPreset.WITTY: "Witty",
    CompanionVerbalTraitPreset.DRY: "Dry humor",
    CompanionVerbalTraitPreset.SARCASTIC: "Light sarcastic",
}

_TRAIT_FRAGMENTS: dict[CompanionVerbalTraitPreset, str] = {
    CompanionVerbalTraitPreset.NEUTRAL: (
        "Keep a calm, helpful desktop-companion tone. Brief and friendly."
    ),
    CompanionVerbalTraitPreset.WARM: (
        "Be warm and encouraging, like a supportive desk buddy. Gentle positivity."
    ),
    CompanionVerbalTraitPreset.WITTY: (
        "Use light, clever humor. Playful but never mean or distracting."
    ),
    CompanionVerbalTraitPreset.DRY: (
        "Dry, understated humor. Deadpan one-liners; no slapstick."
    ),
    CompanionVerbalTraitPreset.SARCASTIC: (
        "Light teasing or wry sarcasm only — never insulting, cruel, or demeaning."
    ),
}


def normalize_companion_verbal_trait(
    value: str | CompanionVerbalTraitPreset | None,
) -> CompanionVerbalTraitPreset:
    if isinstance(value, CompanionVerbalTraitPreset):
        return value
    raw = str(value or "").strip().lower()
    for preset in CompanionVerbalTraitPreset:
        if preset.value == raw:
            return preset
    return DEFAULT_COMPANION_VERBAL_TRAIT


def trait_system_fragment(preset: CompanionVerbalTraitPreset | str) -> str:
    key = normalize_companion_verbal_trait(preset)
    return _TRAIT_FRAGMENTS[key]
