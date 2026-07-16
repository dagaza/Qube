"""HTTP fetch engine — uses knowledge_get() and blocker detection."""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from core.knowledge.egress_policy import EgressPolicy, EgressPolicyError
from core.knowledge.fetch.blockers import detect_blocker
from core.knowledge.fetch.types import BlockerReason, FetchResult
from core.knowledge.http_client import (
    BudgetExhaustedError,
    HostUnavailableError,
    knowledge_get,
)

logger = logging.getLogger("Qube.Knowledge.Fetch")


def fetch_url(
    url: str,
    *,
    egress_policy: EgressPolicy | None = None,
    max_fetch_bytes: int | None = None,
    timeout: float = 10.0,
    fetch_tier: str = "http",
    headers: dict[str, str] | None = None,
) -> FetchResult:
    """Fetch a single URL via the instrumented knowledge HTTP client."""
    started = time.perf_counter()
    request_headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; Qube/1.0; +https://github.com/qube-assistant/qube)"
        ),
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    }
    if headers:
        request_headers.update(headers)

    try:
        resp = knowledge_get(
            url,
            egress_policy=egress_policy,
            max_response_bytes=max_fetch_bytes,
            timeout=timeout,
            headers=request_headers,
        )
    except EgressPolicyError as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return FetchResult(
            url=url,
            final_url=None,
            success=False,
            failure_reason=BlockerReason.EGRESS_BLOCKED,
            status_code=None,
            content_type_header=None,
            html=None,
            fetch_tier=fetch_tier,
            latency_ms=latency_ms,
            raw_metadata={"error": str(exc)},
        )
    except requests.Timeout as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return FetchResult(
            url=url,
            final_url=None,
            success=False,
            failure_reason=BlockerReason.TIMEOUT,
            status_code=None,
            content_type_header=None,
            html=None,
            fetch_tier=fetch_tier,
            latency_ms=latency_ms,
            raw_metadata={"error": str(exc)},
        )
    except (BudgetExhaustedError, HostUnavailableError) as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return FetchResult(
            url=url,
            final_url=None,
            success=False,
            failure_reason=BlockerReason.TIMEOUT,
            status_code=None,
            content_type_header=None,
            html=None,
            fetch_tier=fetch_tier,
            latency_ms=latency_ms,
            raw_metadata={"error": str(exc), "host": getattr(exc, "host", "")},
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        logger.warning("Fetch failed for %s: %s", url, exc)
        return FetchResult(
            url=url,
            final_url=None,
            success=False,
            failure_reason=BlockerReason.HTTP_ERROR,
            status_code=None,
            content_type_header=None,
            html=None,
            fetch_tier=fetch_tier,
            latency_ms=latency_ms,
            raw_metadata={"error": str(exc)},
        )

    html = resp.text or ""
    content_type = resp.headers.get("Content-Type")
    final_url = str(resp.url or url)
    total_bytes = len(html.encode("utf-8", errors="replace"))
    latency_ms = (time.perf_counter() - started) * 1000.0

    failure_reason = detect_blocker(
        html,
        status_code=resp.status_code,
        content_type_header=content_type,
    )
    success = failure_reason is None

    return FetchResult(
        url=url,
        final_url=final_url,
        success=success,
        failure_reason=failure_reason,
        status_code=resp.status_code,
        content_type_header=content_type,
        html=html if success else None,
        fetch_tier=fetch_tier,
        total_bytes=total_bytes,
        latency_ms=latency_ms,
        raw_metadata=_response_metadata(resp),
    )


def fetch_html_string(
    html: str,
    *,
    url: str,
    status_code: int = 200,
    content_type_header: str | None = "text/html",
    fetch_tier: str = "http",
) -> FetchResult:
    """Build a FetchResult from pre-fetched HTML (tests and fixtures)."""
    total_bytes = len((html or "").encode("utf-8", errors="replace"))
    failure_reason = detect_blocker(
        html,
        status_code=status_code,
        content_type_header=content_type_header,
    )
    success = failure_reason is None
    return FetchResult(
        url=url,
        final_url=url,
        success=success,
        failure_reason=failure_reason,
        status_code=status_code,
        content_type_header=content_type_header,
        html=html if success else None,
        fetch_tier=fetch_tier,
        total_bytes=total_bytes,
        latency_ms=0.0,
    )


def _response_metadata(resp: requests.Response) -> dict[str, Any]:
    return {
        "reason": resp.reason,
        "encoding": resp.encoding,
    }
