"""Fetch transport-layer types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class BlockerReason:
    """Structured fetch failure reasons (ADR 004 taxonomy)."""

    CLOUDFLARE = "cloudflare"
    JS_RENDERED = "js_rendered"
    COOKIE_WALL = "cookie_wall"
    PAYWALL = "paywall"
    ROBOTS_DISALLOWED = "robots_disallowed"
    TIMEOUT = "timeout"
    OVERSIZED = "oversized"
    EMPTY_EXTRACT = "empty_extract"
    EGRESS_BLOCKED = "egress_blocked"
    HTTP_ERROR = "http_error"


@dataclass(frozen=True)
class FetchResult:
    """HTTP fetch outcome — raw HTML for extractors; never passed to the LLM."""

    url: str
    final_url: str | None
    success: bool
    failure_reason: str | None
    status_code: int | None
    content_type_header: str | None
    html: str | None
    fetch_tier: str
    page_count: int = 1
    total_bytes: int = 0
    latency_ms: float = 0.0
    raw_metadata: dict[str, Any] = field(default_factory=dict)
