"""Sanitize model output before post-turn validation (matches worker complete-text path)."""
from __future__ import annotations

from core.gemma_output_strip import strip_gemma_output_artifacts
from core.harmony_degeneration import polish_harmony_visible_text
from core.native_meta_leading_strip import LeadingMetaInstructionStripper
from core.output_artifact_strip import strip_harmony_oss_artifacts
from core.redacted_thinking_filter import RedactedThinkingStreamFilter


def sanitize_output_for_validation(text: str) -> str:
    """
    Apply the same complete-text cleanup the worker uses before presentation.

    Streaming filters may already hide Gemma thought-channel tokens from the UI;
    validation should judge this sanitized body, not raw engine deltas.
    """
    if not (text or "").strip():
        return ""

    cot = RedactedThinkingStreamFilter()
    meta = LeadingMetaInstructionStripper()
    cleaned = cot.feed(text or "")
    cleaned += cot.flush()
    cleaned = meta.feed(cleaned) + meta.flush()
    return strip_gemma_output_artifacts(
        strip_harmony_oss_artifacts(polish_harmony_visible_text(cleaned))
    ).strip()
