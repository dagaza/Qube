"""Sidecar prompts for L2/L3 companion expression."""

from __future__ import annotations

from typing import Any

from core.cognition_prompt_adapter import build_cognition_prompt
from core.companion_cognition.personality import CompanionPersonalityVector, load_personality_vector
from core.companion_verbal_prompts import build_companion_line_prompt

_L2_SYSTEM = (
    "You are the Qube desktop companion — not the main chat assistant. "
    "Rephrase the given seed line for the specified mood. "
    "Respond with STRICT JSON only: "
    '{"line":"your line here","kind":"idle_quip|ingest_ack|download_ack|skip"}. '
    "Rules: line must be under 72 characters; no markdown; "
    "never mention JSON, captions, triggers, or 'the companion'; "
    "do not add new facts beyond the seed; "
    'if unsure, return {"line":"","kind":"skip"}.'
)


def build_expression_rewrite_prompt(
    *,
    chat_format: str = "chatml",
    model_path: str = "",
    thought: dict[str, Any],
    observation: dict[str, Any],
    seed_line: str,
    kind: str = "idle_quip",
    user_system_prompt: str = "",
    personality: CompanionPersonalityVector | None = None,
) -> str:
    persona = personality or load_personality_vector()
    mood = str(thought.get("mood") or "neutral")
    intent = str(thought.get("intent") or "")
    obs_type = str(observation.get("type") or "")

    system_parts = [
        _L2_SYSTEM,
        f"Personality tone: {persona.summary_for_prompt()}.",
        f"Intent: {intent}. Mood: {mood}.",
    ]
    extra = (user_system_prompt or "").strip()
    if extra:
        system_parts.append(f"User style notes: {extra[:800]}")
    system = "\n\n".join(system_parts)

    user_lines = [
        f"observation_type: {obs_type}",
        f"kind: {kind}",
        f"seed_line: {seed_line}",
        "Rewrite seed_line only — same meaning, fresh wording.",
    ]
    user = "\n".join(user_lines)
    return build_cognition_prompt(system, user, chat_format, model_path=model_path)


def build_companion_line_prompt_v2(
    *,
    chat_format: str = "chatml",
    model_path: str = "",
    payload: dict[str, Any],
    trait_preset: str = "neutral",
    user_system_prompt: str = "",
) -> str:
    """Route to L2 rewrite or legacy L3 full generation based on payload."""
    level = int(payload.get("expression_level") or 3)
    if level <= 2 and payload.get("seed_line"):
        thought = payload.get("thought") or {}
        observation = payload.get("observation") or {}
        kind = str(thought.get("kind") or payload.get("kind") or "idle_quip")
        return build_expression_rewrite_prompt(
            chat_format=chat_format,
            model_path=model_path,
            thought=thought if isinstance(thought, dict) else {},
            observation=observation if isinstance(observation, dict) else {},
            seed_line=str(payload.get("seed_line") or ""),
            kind=kind,
            user_system_prompt=user_system_prompt,
        )

    trigger = str(payload.get("trigger") or "idle")
    return build_companion_line_prompt(
        chat_format=chat_format,
        model_path=model_path,
        trait_preset=trait_preset,
        user_system_prompt=user_system_prompt,
        trigger=trigger,
        file_count=payload.get("file_count"),
        filename=payload.get("filename"),
        basename=payload.get("basename"),
    )
