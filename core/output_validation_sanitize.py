"""Sanitize model output before post-turn validation (matches worker complete-text path)."""
from __future__ import annotations

from typing import Any

from core.harmony_degeneration import polish_harmony_visible_text
from core.harmony_protocol import harmony_model_active, is_harmony_contract
from core.native_meta_leading_strip import LeadingMetaInstructionStripper
from core.output_artifact_strip import strip_output_artifacts
from core.redacted_thinking_filter import RedactedThinkingStreamFilter


def sanitize_output_for_validation(
    text: str,
    *,
    harmony_active: bool | None = None,
    contract: Any | None = None,
) -> str:
    """
    Apply the same complete-text cleanup the worker uses before presentation.

    Harmony-specific polish/strip layers run only when a Harmony model is active.
    """
    if not (text or "").strip():
        return ""

    active = (
        bool(harmony_active)
        if harmony_active is not None
        else harmony_model_active(contract=contract)
    )

    cot = RedactedThinkingStreamFilter()
    meta = LeadingMetaInstructionStripper()
    cleaned = cot.feed(text or "")
    cleaned += cot.flush()
    cleaned = meta.feed(cleaned) + meta.flush()
    if active:
        cleaned = polish_harmony_visible_text(cleaned)
    return strip_output_artifacts(cleaned, harmony_active=active).strip()
