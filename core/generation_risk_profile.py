"""
Risk-tiered generation parameters for collapse-prone turns.

Adjusts temperature, repeat penalty, completion budget, and stream-guard
sensitivity from turn signals — model-family agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from core.chat_format_mode import ChatFormatMode
from core.reply_shape_policy import detect_enumeration_intent

RiskTier = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class GenerationRiskProfile:
    risk_tier: RiskTier
    risk_score: int
    temperature_multiplier: float
    repeat_penalty_adjust: float
    max_tokens_cap: int | None
    stream_guard_min_repeats: int
    stream_guard_tail_chars: int
    enable_list_loop_guard: bool
    signals: tuple[str, ...]

    def effective_temperature(self, base: float) -> float:
        scaled = float(base) * self.temperature_multiplier
        return max(0.05, min(2.0, scaled))

    def effective_repeat_penalty(self, base: float) -> float:
        return max(0.0, min(2.0, float(base) + self.repeat_penalty_adjust))

    def effective_max_tokens(self, base: int) -> int:
        cap = self.max_tokens_cap
        if cap is None:
            return int(base)
        return max(128, min(int(base), int(cap)))

    def trace_fields(self) -> dict[str, Any]:
        return {
            "generation_risk_tier": self.risk_tier,
            "generation_risk_score": self.risk_score,
            "generation_risk_signals": list(self.signals),
            "temperature_multiplier": self.temperature_multiplier,
            "repeat_penalty_adjust": self.repeat_penalty_adjust,
            "max_tokens_cap": self.max_tokens_cap,
            "stream_guard_min_repeats": self.stream_guard_min_repeats,
            "enable_list_loop_guard": self.enable_list_loop_guard,
        }


def resolve_generation_risk_profile(
    *,
    user_query: str,
    chat_format_mode: ChatFormatMode,
    follow_up_active: bool = False,
    prior_turn_unreliable: bool = False,
    history_turn_count: int = 0,
    require_list_format: bool = False,
) -> GenerationRiskProfile:
    """Score turn collapse risk and derive conservative generation knobs."""
    score = 0
    signals: list[str] = []

    if prior_turn_unreliable:
        score += 2
        signals.append("prior_turn_unreliable")

    if require_list_format or detect_enumeration_intent(user_query):
        score += 1
        signals.append("enumeration_intent")

    if chat_format_mode == "structured":
        score += 1
        signals.append("structured_reply_mode")

    if follow_up_active:
        score += 1
        signals.append("active_follow_up")

    if history_turn_count >= 6:
        score += 1
        signals.append("long_thread")

    if history_turn_count >= 10:
        score += 1
        signals.append("very_long_thread")

    if score <= 1:
        tier: RiskTier = "low"
    elif score <= 3:
        tier = "medium"
    else:
        tier = "high"

    if tier == "high":
        temp_mult = 0.82
        rp_adj = 0.06
        min_repeats = 6
        tail_chars = 800
        list_guard = True
        tokens_cap = 2048 if require_list_format else 1536
    elif tier == "medium":
        temp_mult = 0.92
        rp_adj = 0.03
        min_repeats = 8
        tail_chars = 700
        list_guard = require_list_format or chat_format_mode == "structured"
        tokens_cap = 2560 if require_list_format else None
    else:
        temp_mult = 1.0
        rp_adj = 0.0
        min_repeats = 10
        tail_chars = 600
        list_guard = require_list_format
        tokens_cap = None

    if prior_turn_unreliable and tier != "high":
        temp_mult = min(temp_mult, 0.88)
        rp_adj = max(rp_adj, 0.04)

    return GenerationRiskProfile(
        risk_tier=tier,
        risk_score=score,
        temperature_multiplier=temp_mult,
        repeat_penalty_adjust=rp_adj,
        max_tokens_cap=tokens_cap,
        stream_guard_min_repeats=min_repeats,
        stream_guard_tail_chars=tail_chars,
        enable_list_loop_guard=list_guard,
        signals=tuple(signals),
    )
