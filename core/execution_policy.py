"""
Central execution policy: how reasoning / thinking tokens are handled (native engine path).

Single resolver from ModelReasoningProfile + user Think preference + engine mode.
Does not modify prompts, sampling, or model weights — policy fields for UI + stream routing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from core.model_reasoning_profile import ModelReasoningProfile

ExecutionMode = Literal["direct", "thinking", "hybrid"]
EnforcementMode = Literal["soft", "hard"]


@dataclass
class ExecutionPolicy:
    execution_mode: ExecutionMode
    allow_thinking_tokens: bool
    strip_thinking_output: bool
    ui_display_thinking: bool
    tts_strip_thinking: bool
    enforcement_mode: EnforcementMode


def resolve_user_think_enabled(
    model_profile: Optional["ModelReasoningProfile"],
    user_override: Optional[bool],
) -> bool:
    """Merge QSettings override with model default when override is unset."""
    if user_override is not None:
        return bool(user_override)
    if model_profile is None or not model_profile.supports_thinking_tokens:
        return False
    return model_profile.default_mode in ("thinking", "hybrid")


def _policy_non_thinking() -> ExecutionPolicy:
    """Case A — non-thinking model, or native engine inactive / no profile."""
    return ExecutionPolicy(
        execution_mode="direct",
        allow_thinking_tokens=False,
        strip_thinking_output=True,
        ui_display_thinking=False,
        tts_strip_thinking=True,
        enforcement_mode="hard",
    )


def _policy_thinking_user_off() -> ExecutionPolicy:
    """Case B — thinking-capable model, user chose to hide reasoning."""
    return ExecutionPolicy(
        execution_mode="direct",
        allow_thinking_tokens=False,
        strip_thinking_output=True,
        ui_display_thinking=False,
        tts_strip_thinking=True,
        enforcement_mode="hard",
    )


def _policy_thinking_user_on() -> ExecutionPolicy:
    """Case C — thinking model, user wants full reasoning visible."""
    return ExecutionPolicy(
        execution_mode="thinking",
        allow_thinking_tokens=True,
        strip_thinking_output=False,
        ui_display_thinking=True,
        tts_strip_thinking=False,
        enforcement_mode="soft",
    )


def _policy_hybrid_user_on() -> ExecutionPolicy:
    """Case D — ambiguous / hybrid profile, user ON: soft enforcement, hybrid execution label."""
    return ExecutionPolicy(
        execution_mode="hybrid",
        allow_thinking_tokens=True,
        strip_thinking_output=False,
        ui_display_thinking=True,
        tts_strip_thinking=False,
        enforcement_mode="soft",
    )


def resolve_execution_policy(
    model_profile: Optional["ModelReasoningProfile"],
    user_think_enabled: Optional[bool],
    engine_mode: str,
) -> ExecutionPolicy:
    """
    Resolve the effective ExecutionPolicy (strict cases A–D).

    user_think_enabled: None means “use model default” via resolve_user_think_enabled.
    """
    em = str(engine_mode or "external").lower().strip()
    if em != "internal":
        return _policy_non_thinking()

    user_on = resolve_user_think_enabled(model_profile, user_think_enabled)

    if model_profile is None or not model_profile.supports_thinking_tokens:
        return _policy_non_thinking()

    if not user_on:
        return _policy_thinking_user_off()

    # User ON + thinking-capable model
    if model_profile.default_mode == "hybrid":
        return _policy_hybrid_user_on()
    return _policy_thinking_user_on()
