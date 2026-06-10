"""
Detect degenerated assistant completions before they poison future session history.

Delegates scoring to ``core.output_degeneration``; preserves legacy trace field names.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from core.output_degeneration import (
    HIGH_THRESHOLD,
    HISTORY_SUPPRESSION_PLACEHOLDER,
    OutputDegenerationResult,
    detect_output_degeneration,
    should_mark_turn_unreliable,
)

HISTORY_DEGENERATION_THRESHOLD = 0.55

DegenerationFlag = Literal[
    "repeated_punctuation",
    "harmony_degeneration",
    "unfinished_markdown_list",
    "unfinished_numbering",
    "meta_commentary",
    "abrupt_cutoff",
]


@dataclass(frozen=True)
class HistoryDegenerationResult:
    score: float
    flags: tuple[str, ...]
    suspect: bool
    output_degeneration: OutputDegenerationResult | None = None

    @property
    def should_suppress(self) -> bool:
        return self.suspect

    def trace_fields(self) -> dict[str, object]:
        fields: dict[str, object] = {
            "history_degeneration_score": round(self.score, 3),
            "history_degeneration_flags": list(self.flags),
            "history_degeneration_suspect": self.suspect,
            "history_degeneration_suppressed": self.should_suppress,
        }
        if self.output_degeneration is not None:
            fields.update(self.output_degeneration.trace_fields())
        return fields


def _legacy_flags(result: OutputDegenerationResult) -> tuple[str, ...]:
    """Map unified detector flags to legacy history-degeneration names."""
    mapping = {
        "repetition": "repeated_punctuation",
        "malformed_list": "harmony_degeneration",
        "unfinished_bullet": "unfinished_numbering",
        "markdown_explosion": "unfinished_markdown_list",
        "meta_commentary": "meta_commentary",
        "self_correction": "meta_commentary",
        "truncation": "abrupt_cutoff",
        "punctuation_loop": "repeated_punctuation",
        "entropy_collapse": "repeated_punctuation",
    }
    out: list[str] = []
    for flag in result.flags:
        legacy = mapping.get(flag, flag)
        if legacy not in out:
            out.append(legacy)
    if not out and result.risk == "HIGH":
        out.append("harmony_degeneration")
    return tuple(out)


def score_history_degeneration(text: str) -> HistoryDegenerationResult:
    """Score assistant text for history-poisoning degeneration markers."""
    detected = detect_output_degeneration(text)
    suspect = should_mark_turn_unreliable(detected)
    score = detected.composite_score
    if suspect and score < HIGH_THRESHOLD:
        score = HIGH_THRESHOLD
    return HistoryDegenerationResult(
        score=score,
        flags=_legacy_flags(detected),
        suspect=suspect,
        output_degeneration=detected,
    )


def resolve_assistant_history_content(
    text: str,
) -> tuple[str, HistoryDegenerationResult]:
    """Return the assistant content to store in session history."""
    result = score_history_degeneration(text)
    if result.should_suppress:
        return HISTORY_SUPPRESSION_PLACEHOLDER, result
    return (text or "").strip(), result
