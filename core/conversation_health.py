"""
Session-level conversation health and turn-to-turn anomaly escalation.

Tracks ``conversation_health_score`` (1.0 = healthy) across turns. Poor outputs
reduce health; the next turn's prompt assembly and generation inherit stricter
policies automatically (normal → warning → recovery).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

from core.generation_risk_profile import GenerationRiskProfile

ConversationHealthMode = Literal["normal", "warning", "recovery"]

INITIAL_HEALTH_SCORE = 1.0
NORMAL_MIN = 0.75
WARNING_MIN = 0.45

HIGH_ANOMALY_PENALTY = 0.35
MEDIUM_ANOMALY_PENALTY = 0.18
STREAM_CANCEL_PENALTY = 0.10
CONSECUTIVE_ESCALATION_BONUS = 0.10
CLEAN_TURN_RECOVERY = 0.05


@dataclass(frozen=True)
class ConversationHealthState:
    health_score: float
    consecutive_anomalies: int = 0
    turn_count: int = 0

    @property
    def mode(self) -> ConversationHealthMode:
        if self.health_score >= NORMAL_MIN:
            return "normal"
        if self.health_score >= WARNING_MIN:
            return "warning"
        return "recovery"

    def trace_fields(self) -> dict[str, Any]:
        return {
            "conversation_health_score": round(self.health_score, 3),
            "conversation_health_mode": self.mode,
            "conversation_health_consecutive_anomalies": self.consecutive_anomalies,
            "conversation_health_turn_count": self.turn_count,
        }


@dataclass(frozen=True)
class ConversationHealthPolicy:
    mode: ConversationHealthMode
    health_score: float
    allow_discourse_rewrite: bool
    allow_query_rewrite: bool
    allow_salience_hints: bool
    temperature_multiplier: float
    max_tokens_cap: int | None
    stream_guard_min_repeats: int
    stream_guard_tail_chars: int
    degeneration_rescore_every: int
    enable_list_loop_guard: bool

    def trace_fields(self) -> dict[str, Any]:
        return {
            "conversation_health_policy_mode": self.mode,
            **ConversationHealthState(self.health_score).trace_fields(),
            "conversation_health_allow_discourse_rewrite": self.allow_discourse_rewrite,
            "conversation_health_allow_query_rewrite": self.allow_query_rewrite,
            "conversation_health_temperature_multiplier": self.temperature_multiplier,
            "conversation_health_max_tokens_cap": self.max_tokens_cap,
            "conversation_health_stream_guard_min_repeats": self.stream_guard_min_repeats,
            "conversation_health_degeneration_rescore_every": self.degeneration_rescore_every,
        }


@dataclass(frozen=True)
class TurnAnomalyOutcome:
    degeneration_risk: str = "LOW"
    history_suppressed: bool = False
    collapse_risk: str = "LOW"
    stream_degeneration_cancelled: bool = False

    @property
    def had_anomaly(self) -> bool:
        return self.anomaly_penalty() > 0.0

    @property
    def self_inflicted_stream_cancel(self) -> bool:
        """Worker cancelled the stream but post-turn review found no Class A pathology."""
        return bool(self.stream_degeneration_cancelled and not self.history_suppressed)

    def anomaly_penalty(self) -> float:
        if self.self_inflicted_stream_cancel:
            return 0.0
        penalty = 0.0
        if self.history_suppressed or self.degeneration_risk == "HIGH":
            penalty = max(penalty, HIGH_ANOMALY_PENALTY)
        elif self.degeneration_risk == "MEDIUM":
            penalty = max(penalty, MEDIUM_ANOMALY_PENALTY)
        if self.collapse_risk == "HIGH":
            penalty = max(penalty, 0.30)
        elif self.collapse_risk == "MEDIUM":
            penalty = max(penalty, 0.15)
        if self.stream_degeneration_cancelled:
            penalty += STREAM_CANCEL_PENALTY
        return min(0.85, penalty)


def initial_conversation_health() -> ConversationHealthState:
    return ConversationHealthState(
        health_score=INITIAL_HEALTH_SCORE,
        consecutive_anomalies=0,
        turn_count=0,
    )


def resolve_conversation_health_policy(
    state: ConversationHealthState | None,
) -> ConversationHealthPolicy:
    """Map current session health to execution policy for the upcoming turn."""
    health = state.health_score if state is not None else INITIAL_HEALTH_SCORE
    mode = (
        state.mode
        if state is not None
        else "normal"
    )

    if mode == "normal":
        return ConversationHealthPolicy(
            mode=mode,
            health_score=health,
            allow_discourse_rewrite=True,
            allow_query_rewrite=True,
            allow_salience_hints=True,
            temperature_multiplier=1.0,
            max_tokens_cap=None,
            stream_guard_min_repeats=10,
            stream_guard_tail_chars=600,
            degeneration_rescore_every=120,
            enable_list_loop_guard=False,
        )

    if mode == "warning":
        return ConversationHealthPolicy(
            mode=mode,
            health_score=health,
            allow_discourse_rewrite=False,
            allow_query_rewrite=False,
            allow_salience_hints=False,
            temperature_multiplier=0.88,
            max_tokens_cap=1536,
            stream_guard_min_repeats=6,
            stream_guard_tail_chars=750,
            degeneration_rescore_every=120,
            enable_list_loop_guard=True,
        )

    return ConversationHealthPolicy(
        mode="recovery",
        health_score=health,
        allow_discourse_rewrite=False,
        allow_query_rewrite=False,
        allow_salience_hints=False,
        temperature_multiplier=0.72,
        max_tokens_cap=1024,
        stream_guard_min_repeats=4,
        stream_guard_tail_chars=850,
        degeneration_rescore_every=100,
        enable_list_loop_guard=True,
    )


def update_conversation_health(
    state: ConversationHealthState | None,
    *,
    outcome: TurnAnomalyOutcome,
) -> ConversationHealthState:
    """Apply turn outcome and return updated session health."""
    current = state or initial_conversation_health()
    penalty = outcome.anomaly_penalty()
    if current.consecutive_anomalies >= 1 and penalty > 0:
        penalty = min(0.85, penalty + CONSECUTIVE_ESCALATION_BONUS)

    if penalty > 0:
        new_score = max(0.0, current.health_score - penalty)
        return ConversationHealthState(
            health_score=new_score,
            consecutive_anomalies=current.consecutive_anomalies + 1,
            turn_count=current.turn_count + 1,
        )

    recovered = min(INITIAL_HEALTH_SCORE, current.health_score + CLEAN_TURN_RECOVERY)
    return ConversationHealthState(
        health_score=recovered,
        consecutive_anomalies=0,
        turn_count=current.turn_count + 1,
    )


def merge_generation_risk_with_health(
    risk: GenerationRiskProfile,
    health: ConversationHealthPolicy | None,
) -> GenerationRiskProfile:
    """Combine per-turn risk profile with session health escalation policy."""
    if health is None or health.mode == "normal":
        return risk

    temp_mult = risk.temperature_multiplier * health.temperature_multiplier
    min_repeats = min(risk.stream_guard_min_repeats, health.stream_guard_min_repeats)
    tail_chars = max(risk.stream_guard_tail_chars, health.stream_guard_tail_chars)
    tokens_cap = risk.max_tokens_cap
    if health.max_tokens_cap is not None:
        tokens_cap = (
            min(tokens_cap, health.max_tokens_cap)
            if tokens_cap is not None
            else health.max_tokens_cap
        )

    tier = risk.risk_tier
    if health.mode == "recovery":
        tier = "high"
    elif health.mode == "warning" and tier == "low":
        tier = "medium"

    signals = list(risk.signals)
    signals.append(f"conversation_health_{health.mode}")

    enumeration_intent = "enumeration_intent" in risk.signals
    list_guard = (
        (risk.enable_list_loop_guard or health.enable_list_loop_guard)
        and not enumeration_intent
    )

    return replace(
        risk,
        risk_tier=tier,
        temperature_multiplier=temp_mult,
        max_tokens_cap=tokens_cap,
        stream_guard_min_repeats=min_repeats,
        stream_guard_tail_chars=tail_chars,
        enable_list_loop_guard=list_guard,
        signals=tuple(signals),
    )
