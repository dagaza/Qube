"""Sanitize model output before post-turn validation (matches worker complete-text path)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from core.harmony_degeneration import polish_harmony_visible_text
from core.harmony_protocol import harmony_model_active, is_harmony_contract
from core.native_meta_leading_strip import LeadingMetaInstructionStripper
from core.output_artifact_strip import strip_output_artifacts
from core.redacted_thinking_filter import RedactedThinkingStreamFilter

if TYPE_CHECKING:
    from core.execution_policy import ExecutionPolicy


def _strip_thinking_enabled(
    policy: "ExecutionPolicy | None",
    *,
    strip_thinking: bool | None = None,
) -> bool:
    if strip_thinking is not None:
        return bool(strip_thinking)
    if policy is None:
        return True
    return bool(policy.strip_thinking_output)


def sanitize_output_for_validation(
    text: str,
    *,
    harmony_active: bool | None = None,
    contract: Any | None = None,
    policy: "ExecutionPolicy | None" = None,
    strip_thinking: bool | None = None,
) -> str:
    """
    Apply the same complete-text cleanup the worker uses before presentation.

    Harmony-specific polish/strip layers run only when a Harmony model is active.
    Thinking blocks are removed when ``strip_thinking_output`` is true (default).
    """
    if not (text or "").strip():
        return ""

    active = (
        bool(harmony_active)
        if harmony_active is not None
        else harmony_model_active(contract=contract)
    )

    cleaned = text or ""
    if _strip_thinking_enabled(policy, strip_thinking=strip_thinking):
        cot = RedactedThinkingStreamFilter()
        cleaned = cot.feed(cleaned)
        cleaned += cot.flush()
    meta = LeadingMetaInstructionStripper()
    cleaned = meta.feed(cleaned) + meta.flush()
    if active:
        cleaned = polish_harmony_visible_text(cleaned)
    return strip_output_artifacts(cleaned, harmony_active=active).strip()
