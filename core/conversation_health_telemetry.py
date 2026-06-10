"""Structured conversation health telemetry."""
from __future__ import annotations

from core.conversation_health import ConversationHealthState, TurnAnomalyOutcome
from core.llm_structured_log import structured_llm_log


def log_conversation_health_update(
    *,
    session_id: str,
    before: ConversationHealthState,
    after: ConversationHealthState,
    outcome: TurnAnomalyOutcome,
) -> None:
    structured_llm_log(
        "conversation_health_update",
        {
            "session_id": session_id,
            "health_before": round(before.health_score, 3),
            "health_after": round(after.health_score, 3),
            "mode_before": before.mode,
            "mode_after": after.mode,
            "anomaly_penalty": round(outcome.anomaly_penalty(), 3),
            "degeneration_risk": outcome.degeneration_risk,
            "collapse_risk": outcome.collapse_risk,
            "history_suppressed": outcome.history_suppressed,
            "stream_degeneration_cancelled": outcome.stream_degeneration_cancelled,
            **after.trace_fields(),
        },
    )
