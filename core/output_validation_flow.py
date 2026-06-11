"""Shared post-stream validation + adaptive retry orchestration."""
from __future__ import annotations

import time
from typing import Any

from core.adaptive_retry import maybe_retry
from core.output_validation import OutputValidationResult, validate_output
from core.output_validation_sanitize import sanitize_output_for_validation
from core.output_validation_trace import (
    OutputValidationTrace,
    log_output_validation_trace,
)
from core.prompt_contract import PromptContract


def run_output_validation_and_retry(
    engine: Any,
    *,
    final_text: str,
    contract: PromptContract,
    messages: list[dict],
    max_tokens: int,
    session_id: str = "",
    phase: str = "post_stream",
    inference_finished_at: float | None = None,
    stream_finish_reason: str = "",
    truncation_notice_reason: str | None = None,
    effective_max_tokens: int | None = None,
) -> tuple[str, PromptContract, OutputValidationTrace, OutputValidationResult]:
    """Validate sanitized output; optionally adaptive-retry with matched token budget."""
    raw = final_text or ""
    sanitized = sanitize_output_for_validation(raw)
    validation = validate_output(sanitized, contract)
    raw_validation = validate_output(raw, contract)

    retry_budget = max(512, int(max_tokens))
    trace = OutputValidationTrace(
        first_pass_raw_len=len(raw),
        sanitized_len=len(sanitized),
        validation_issues=list(validation.issues),
        validation_severity=str(validation.severity),
        raw_validation_issues=list(raw_validation.issues),
        raw_validation_severity=str(raw_validation.severity),
        retry_max_tokens=retry_budget,
        original_max_tokens=int(max_tokens),
        original_format=str(contract.chat_format or contract.mode),
        stream_finish_reason=str(stream_finish_reason or ""),
        truncation_notice_reason=truncation_notice_reason,
        effective_max_tokens=effective_max_tokens,
    )

    setattr(engine, "_adaptive_retry_max_tokens", retry_budget)
    try:
        outcome = maybe_retry(
            engine,
            messages,
            contract,
            raw,
            validation,
            max_tokens=max_tokens,
        )
    finally:
        if hasattr(engine, "_adaptive_retry_max_tokens"):
            delattr(engine, "_adaptive_retry_max_tokens")

    retried_text = outcome.text
    final_contract = outcome.contract
    trace.retry_attempted = bool(outcome.retry_attempted)
    trace.retry_used = bool(outcome.retry_used)
    trace.retry_reason = outcome.retry_reason
    trace.final_format = str(final_contract.chat_format or final_contract.mode)
    trace.retry_replaced_stream = bool(
        outcome.retry_used and retried_text and retried_text != raw
    )
    if outcome.retry_used:
        trace.retry_len = len(retried_text or "")

    if inference_finished_at is not None:
        trace.post_inference_ms = max(
            0, int((time.monotonic() - float(inference_finished_at)) * 1000)
        )

    log_output_validation_trace(session_id=session_id, trace=trace, phase=phase)
    engine._last_output_validation_trace = trace

    if trace.retry_replaced_stream:
        return retried_text, final_contract, trace, validation
    return raw, contract if not outcome.retry_used else final_contract, trace, validation


def annotate_raw_vs_sanitized_validation(
    raw: str,
    contract: PromptContract,
) -> tuple[OutputValidationResult, OutputValidationResult, str]:
    sanitized = sanitize_output_for_validation(raw)
    return validate_output(sanitized, contract), validate_output(raw, contract), sanitized
