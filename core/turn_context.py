"""
Per-turn execution context: reply shape, generation risk, and history strategy.

Single resolution point consumed by ``LLMWorker`` before prompt render and inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from core.conversation_health import (
    ConversationHealthPolicy,
    ConversationHealthState,
    merge_generation_risk_with_health,
    resolve_conversation_health_policy,
)
from core.discourse_intent import FollowUpClassification
from core.generation_risk_profile import GenerationRiskProfile, resolve_generation_risk_profile
from core.reply_shape_policy import ReplyShapePolicy, resolve_reply_shape_policy

HistoryStrategy = Literal["native_roles", "harmony_compact"]


@dataclass(frozen=True)
class TurnContext:
    reply_shape: ReplyShapePolicy
    generation_risk: GenerationRiskProfile
    history_strategy: HistoryStrategy
    conversation_health: ConversationHealthPolicy | None = None

    @property
    def chat_format_mode(self):
        return self.reply_shape.chat_format_mode

    def trace_fields(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "history_strategy": self.history_strategy,
        }
        out.update(self.reply_shape.trace_fields())
        out.update(self.generation_risk.trace_fields())
        if self.conversation_health is not None:
            out.update(self.conversation_health.trace_fields())
        return out


def resolve_history_strategy(*, use_harmony_protocol: bool) -> HistoryStrategy:
    """Harmony models compact prior turns; other families keep native role messages."""
    return "harmony_compact" if use_harmony_protocol else "native_roles"


def resolve_turn_context(
    *,
    execution_route: str,
    user_query: str,
    follow_up: FollowUpClassification | None = None,
    prior_turn_unreliable: bool = False,
    has_retrieval_sources: bool = False,
    history_turn_count: int = 0,
    use_harmony_protocol: bool = False,
    conversation_health: ConversationHealthState | None = None,
) -> TurnContext:
    reply_shape = resolve_reply_shape_policy(
        execution_route=execution_route,
        user_query=user_query,
        follow_up=follow_up,
        prior_turn_unreliable=prior_turn_unreliable,
        has_retrieval_sources=has_retrieval_sources,
    )
    generation_risk = resolve_generation_risk_profile(
        user_query=user_query,
        chat_format_mode=reply_shape.chat_format_mode,
        follow_up_active=bool(follow_up and follow_up.active),
        prior_turn_unreliable=prior_turn_unreliable,
        history_turn_count=history_turn_count,
        require_list_format=reply_shape.require_list_format,
    )
    health_policy = resolve_conversation_health_policy(conversation_health)
    generation_risk = merge_generation_risk_with_health(generation_risk, health_policy)
    return TurnContext(
        reply_shape=reply_shape,
        generation_risk=generation_risk,
        history_strategy=resolve_history_strategy(
            use_harmony_protocol=use_harmony_protocol,
        ),
        conversation_health=health_policy,
    )
