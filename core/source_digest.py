"""
Best-effort compression of retrieved sources for the primary model prompt.

Digest runs only when context exceeds ``get_sidecar_source_digest_min_chars()``
so clean, already-compact retrieval is passed through unchanged.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from core.app_settings import (
    get_sidecar_foreground_timeout_ms,
    get_sidecar_source_digest_enabled,
    get_sidecar_source_digest_min_chars,
)
from core.sidecar_types import SidecarTask

logger = logging.getLogger("Qube.SourceDigest")


@dataclass(frozen=True)
class DigestResult:
    text: str
    applied: bool
    chars_before: int = 0
    chars_after: int = 0
    source_count: int = 0
    skip_reason: str = ""


def _format_sources_for_sidecar(sources: list[dict]) -> str:
    lines: list[str] = []
    for src in sources or []:
        sid = src.get("id")
        fname = src.get("filename") or "Source"
        content = (src.get("content") or "").strip()[:1200]
        if content:
            lines.append(f"[{sid}] {fname}:\n{content}")
    return "\n\n".join(lines)


def _should_digest_context(context_text: str) -> tuple[bool, str]:
    if not get_sidecar_source_digest_enabled():
        return False, "disabled"
    chars_before = len((context_text or "").strip())
    min_chars = get_sidecar_source_digest_min_chars()
    if chars_before < min_chars:
        return False, "below_threshold"
    return True, ""


def digest_memory_context(
    memory_context: str,
    memory_sources: list[dict],
    sidecar_client: Any,
) -> DigestResult:
    """Return digest outcome; falls back to raw context on skip or failure."""
    chars_before = len((memory_context or "").strip())
    source_count = len(memory_sources or [])
    if not memory_context or not memory_sources or sidecar_client is None:
        return DigestResult(
            text=memory_context or "",
            applied=False,
            chars_before=chars_before,
            chars_after=chars_before,
            source_count=source_count,
            skip_reason="no_context",
        )

    should_run, skip_reason = _should_digest_context(memory_context)
    if not should_run:
        if skip_reason == "below_threshold":
            logger.debug(
                "[Sidecar] memory digest skipped (%s): %d chars < min %d",
                skip_reason,
                chars_before,
                get_sidecar_source_digest_min_chars(),
            )
        return DigestResult(
            text=memory_context,
            applied=False,
            chars_before=chars_before,
            chars_after=chars_before,
            source_count=source_count,
            skip_reason=skip_reason,
        )

    text = _format_sources_for_sidecar(memory_sources)
    if not text.strip():
        return DigestResult(
            text=memory_context,
            applied=False,
            chars_before=chars_before,
            chars_after=chars_before,
            source_count=source_count,
            skip_reason="empty_sources",
        )

    expected_ids = [s.get("id") for s in memory_sources if s.get("id") is not None]
    timeout = get_sidecar_foreground_timeout_ms() / 1000.0
    try:
        result = sidecar_client.complete(
            SidecarTask.source_digest,
            timeout_sec=timeout,
            sources_text=text,
            expected_ids=expected_ids,
        )
    except Exception as e:
        logger.debug("[Sidecar] source digest failed: %s", e)
        return DigestResult(
            text=memory_context,
            applied=False,
            chars_before=chars_before,
            chars_after=chars_before,
            source_count=source_count,
            skip_reason="error",
        )

    if not result.ok or not result.text:
        return DigestResult(
            text=memory_context,
            applied=False,
            chars_before=chars_before,
            chars_after=chars_before,
            source_count=source_count,
            skip_reason=result.error or "fail",
        )

    digest = result.text.strip()
    if not digest:
        return DigestResult(
            text=memory_context,
            applied=False,
            chars_before=chars_before,
            chars_after=chars_before,
            source_count=source_count,
            skip_reason="empty_output",
        )

    return DigestResult(
        text=digest,
        applied=True,
        chars_before=chars_before,
        chars_after=len(digest),
        source_count=source_count,
        skip_reason="",
    )


def digest_rag_context(
    llm_context: str,
    sources: list[dict],
    sidecar_client: Any,
) -> DigestResult:
    chars_before = len((llm_context or "").strip())
    source_count = len(sources or [])
    if not llm_context or not sources or sidecar_client is None:
        return DigestResult(
            text=llm_context or "",
            applied=False,
            chars_before=chars_before,
            chars_after=chars_before,
            source_count=source_count,
            skip_reason="no_context",
        )

    should_run, skip_reason = _should_digest_context(llm_context)
    if not should_run:
        if skip_reason == "below_threshold":
            logger.debug(
                "[Sidecar] RAG digest skipped (%s): %d chars < min %d",
                skip_reason,
                chars_before,
                get_sidecar_source_digest_min_chars(),
            )
        return DigestResult(
            text=llm_context,
            applied=False,
            chars_before=chars_before,
            chars_after=chars_before,
            source_count=source_count,
            skip_reason=skip_reason,
        )

    text = _format_sources_for_sidecar(sources)
    expected_ids = [s.get("id") for s in sources if s.get("id") is not None]
    timeout = get_sidecar_foreground_timeout_ms() / 1000.0
    try:
        result = sidecar_client.complete(
            SidecarTask.source_digest,
            timeout_sec=timeout,
            sources_text=text,
            expected_ids=expected_ids,
        )
    except Exception as e:
        logger.debug("[Sidecar] rag digest failed: %s", e)
        return DigestResult(
            text=llm_context,
            applied=False,
            chars_before=chars_before,
            chars_after=chars_before,
            source_count=source_count,
            skip_reason="error",
        )

    if not result.ok or not result.text:
        return DigestResult(
            text=llm_context,
            applied=False,
            chars_before=chars_before,
            chars_after=chars_before,
            source_count=source_count,
            skip_reason=result.error or "fail",
        )

    digest = result.text.strip()
    if not digest:
        return DigestResult(
            text=llm_context,
            applied=False,
            chars_before=chars_before,
            chars_after=chars_before,
            source_count=source_count,
            skip_reason="empty_output",
        )

    return DigestResult(
        text=digest,
        applied=True,
        chars_before=chars_before,
        chars_after=len(digest),
        source_count=source_count,
        skip_reason="",
    )
