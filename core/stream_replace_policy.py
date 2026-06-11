"""Decide whether a format-retry may replace user-visible streamed text."""
from __future__ import annotations

from core.conversational_follow_up import preserve_streamed_follow_up
from core.output_artifact_strip import strip_harmony_oss_artifacts

_DEFAULT_MIN_RATIO = 0.8


def resolve_stream_replacement(
    replacement: str,
    streamed: str,
    *,
    min_ratio: float = _DEFAULT_MIN_RATIO,
) -> tuple[str, str | None]:
    """
    Return ``(text_to_use, rejection_reason)``.

    Reject replacements that would shrink a good streamed answer the user already saw.
    """
    rep = strip_harmony_oss_artifacts((replacement or "").strip())
    stream = strip_harmony_oss_artifacts((streamed or "").strip())
    if not stream:
        return rep, None
    if not rep:
        return stream, "empty_retry"

    if len(rep) >= len(stream):
        return preserve_streamed_follow_up(rep, stream), None

    if stream.startswith(rep):
        return stream, "retry_prefix_of_streamed"

    merged = preserve_streamed_follow_up(rep, stream)
    if len(merged) >= len(stream):
        return merged, None

    effective_len = max(len(rep), len(merged))
    if effective_len < len(stream) * min_ratio:
        return stream, "retry_shorter_than_streamed"

    return merged, None
