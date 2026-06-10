"""Structured output degeneration observability."""
from __future__ import annotations

from core.llm_structured_log import structured_llm_log
from core.output_degeneration import OutputDegenerationResult


def log_output_degeneration(
    *,
    session_id: str = "",
    result: OutputDegenerationResult,
    phase: str = "persist",
) -> None:
    structured_llm_log(
        "output_degeneration",
        {
            "session_id": session_id,
            "phase": phase,
            **result.trace_fields(),
        },
    )
