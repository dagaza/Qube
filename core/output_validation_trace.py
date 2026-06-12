"""Structured validation / retry telemetry for ~/.qube/logs/llm_debug.log."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from core.llm_structured_log import structured_llm_log
from core.output_validation import OutputValidationResult


@dataclass
class OutputValidationTrace:
    first_pass_raw_len: int = 0
    sanitized_len: int = 0
    validation_issues: list[str] = field(default_factory=list)
    validation_severity: str = "low"
    raw_validation_issues: list[str] = field(default_factory=list)
    raw_validation_severity: str = "low"
    retry_attempted: bool = False
    retry_used: bool = False
    retry_reason: str | None = None
    retry_len: int | None = None
    retry_max_tokens: int = 512
    original_max_tokens: int = 512
    original_format: str = ""
    final_format: str = ""
    replacement_suppressed: bool = False
    replacement_rejection_reason: str | None = None
    streamed_visible_len: int | None = None
    retry_replaced_stream: bool = False
    post_inference_ms: int | None = None
    stream_finish_reason: str = ""
    truncation_notice_reason: str | None = None
    effective_max_tokens: int | None = None
    degeneration_score: float | None = None
    degeneration_retry_eligible: bool | None = None
    degeneration_top_offender: str | None = None
    degeneration_clustered: bool | None = None
    markdown_heading_count: int | None = None
    bold_section_title_count: int | None = None
    heading_style_ratio: float | None = None

    def trace_fields(self) -> dict[str, Any]:
        return asdict(self)


def log_output_validation_trace(
    *,
    session_id: str = "",
    trace: OutputValidationTrace,
    phase: str = "post_stream",
) -> None:
    structured_llm_log(
        "output_validation",
        {
            "session_id": session_id,
            "phase": phase,
            **trace.trace_fields(),
        },
    )


def validation_result_fields(result: OutputValidationResult) -> dict[str, Any]:
    return {
        "issues": list(result.issues),
        "severity": result.severity,
        "is_valid": result.is_valid,
    }
