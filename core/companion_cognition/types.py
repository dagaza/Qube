"""Frozen datatypes for Companion Cognition v2."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class ExpressionLevel(IntEnum):
    CURATED = 0
    TEMPLATE = 1
    SIDECAR_REWRITE = 2
    FULL_GENERATE = 3


@dataclass(frozen=True)
class CompanionTriggerEvent:
    """Allowlisted trigger input for the cognition pipeline."""

    source: str
    ts: float
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompanionObservation:
    """Deterministic, non-PII observation derived from a trigger."""

    type: str
    facts: dict[str, Any]
    confidence: float = 1.0
    trigger_source: str = ""


@dataclass(frozen=True)
class CompanionThought:
    """Internal companion intention — no natural language yet."""

    intent: str
    mood: str
    energy: str
    voice: str = "observational"
    ambient_mood: str = ""
    tone_constraints: tuple[str, ...] = ()
    slots: dict[str, Any] = field(default_factory=dict)
    observation_type: str = ""
    skip_reason: str = ""


@dataclass(frozen=True)
class ExpressionPlan:
    """Chosen expression strategy for a thought."""

    level: ExpressionLevel
    message_id: str = ""
    template_id: str = ""
    seed_line: str = ""
    kind: str = "idle_quip"


@dataclass(frozen=True)
class ExpressionResult:
    """Local (L0/L1) caption ready for display."""

    line: str
    kind: str
    trigger: str
    level: ExpressionLevel
    message_id: str = ""
    intent: str = ""
    mood: str = ""
    voice: str = ""
    motifs: tuple[str, ...] = ()


@dataclass(frozen=True)
class SidecarExpressionRequest:
    """Payload for L2/L3 sidecar companion_line queue."""

    trigger: str
    expression_level: int
    thought: dict[str, Any]
    observation: dict[str, Any]
    seed_line: str = ""
    file_count: int | None = None
    basename: str | None = None
    filename: str | None = None

    def to_payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "trigger": self.trigger,
            "expression_level": self.expression_level,
            "thought": dict(self.thought),
            "observation": dict(self.observation),
        }
        if self.seed_line:
            out["seed_line"] = self.seed_line
        if self.file_count is not None:
            out["file_count"] = self.file_count
        if self.basename is not None:
            out["basename"] = self.basename
        if self.filename is not None:
            out["filename"] = self.filename
        return out


@dataclass(frozen=True)
class CognitionProcessResult:
    """Orchestrator output — local caption or sidecar enqueue."""

    skip_reason: str = ""
    local: ExpressionResult | None = None
    sidecar: SidecarExpressionRequest | None = None
    emission_message_id: str = ""
    emission_intent: str = ""
    emission_mood: str = ""
    emission_voice: str = ""
    emission_motifs: tuple[str, ...] = ()
    emission_ambient_mood: str = ""
