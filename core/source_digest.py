"""
Best-effort compression of retrieved sources for the primary model prompt.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from core.app_settings import (
    get_sidecar_foreground_timeout_ms,
    get_sidecar_source_digest_enabled,
)
from core.sidecar_types import SidecarTask

logger = logging.getLogger("Qube.SourceDigest")


def _format_sources_for_sidecar(sources: list[dict]) -> str:
    lines: list[str] = []
    for src in sources or []:
        sid = src.get("id")
        fname = src.get("filename") or "Source"
        content = (src.get("content") or "").strip()[:1200]
        if content:
            lines.append(f"[{sid}] {fname}:\n{content}")
    return "\n\n".join(lines)


def digest_memory_context(
    memory_context: str,
    memory_sources: list[dict],
    sidecar_client: Any,
) -> tuple[str, bool]:
    """Return (context_text, applied)."""
    if not get_sidecar_source_digest_enabled():
        return memory_context, False
    if not memory_context or not memory_sources or sidecar_client is None:
        return memory_context, False

    text = _format_sources_for_sidecar(memory_sources)
    if not text.strip():
        return memory_context, False

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
        return memory_context, False

    if not result.ok or not result.text:
        return memory_context, False

    digest = result.text.strip()
    if not digest:
        return memory_context, False

    return digest, True


def digest_rag_context(
    llm_context: str,
    sources: list[dict],
    sidecar_client: Any,
) -> tuple[str, bool]:
    if not get_sidecar_source_digest_enabled():
        return llm_context, False
    if not llm_context or not sources or sidecar_client is None:
        return llm_context, False

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
        return llm_context, False

    if not result.ok or not result.text:
        return llm_context, False
    return result.text.strip(), True
