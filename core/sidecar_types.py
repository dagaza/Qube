"""Sidecar LLM shared types (no worker / prompt imports)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class SidecarTask(str, Enum):
    title = "title"
    contradiction_judge = "contradiction_judge"
    reflection_label = "reflection_label"
    episode_summary = "episode_summary"
    query_rewrite = "query_rewrite"
    source_digest = "source_digest"
    ingest_blurb = "ingest_blurb"


@dataclass
class SidecarResult:
    text: str = ""
    parsed: Optional[dict[str, Any]] = None
    confidence: float = 0.0
    ok: bool = False
    error: str = ""
    task: Optional[SidecarTask] = None


@dataclass
class QueryExpansion:
    original_query: str
    expanded_query: str
    confidence: float
    topic_source: str = "discourse_state"
