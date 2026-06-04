"""Companion Cognition v2 — observation → thought → expression pipeline."""

from core.companion_cognition.orchestrator import CompanionCognitionOrchestrator
from core.companion_cognition.types import (
    CompanionObservation,
    CompanionThought,
    CompanionTriggerEvent,
    ExpressionLevel,
    ExpressionResult,
    SidecarExpressionRequest,
)

__all__ = [
    "CompanionCognitionOrchestrator",
    "CompanionObservation",
    "CompanionThought",
    "CompanionTriggerEvent",
    "ExpressionLevel",
    "ExpressionResult",
    "SidecarExpressionRequest",
]
