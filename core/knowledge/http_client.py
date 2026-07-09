"""Instrumented knowledge HTTP client with scheduling and header-aware retries."""

from __future__ import annotations

import logging
import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Mapping

import requests

from core.knowledge.host_scheduler import (
    HostCircuitOpenError,
    get_host_scheduler,
    host_health_snapshot,
    metrics_host_for,
)
from core.knowledge.http_metrics import (
    hostname_from_url,
    record_http_request,
    record_http_retry_reason,
)
from core.knowledge.negative_cache import get_host_negative, mark_host_negative
from core.knowledge.provider_limit_events import notify_budget_exhausted

logger = logging.getLogger("Qube.Knowledge.HTTP")

DEFAULT_TIMEOUT_SEC = 10.0
MAX_RATE_LIMIT_RETRIES = 3
MAX_SERVER_ERROR_RETRIES = 3
SERVER_ERROR_CODES = frozenset({502, 503, 504})
DEFAULT_429_FALLBACK_SEC = 3.0
OPENALEX_HOST = "api.openalex.org"


class BudgetExhaustedError(Exception):
    """Daily or burst quota exhausted — callers should fail fast without retry loops."""

    def __init__(self, *, host: str, metrics_host: str) -> None:
        self.host = host
        self.metrics_host = metrics_host
        super().__init__(f"Budget exhausted for {host}")


class HostUnavailableError(Exception):
    """Host temporarily blocked (negative cache / future circuit breaker)."""

    def __init__(self, *, host: str, metrics_host: str, reason: str) -> None:
        self.host = host
        self.metrics_host = metrics_host
        self.reason = reason
        super().__init__(f"Host unavailable ({reason}): {host}")


def retry_after_seconds(
    headers: Mapping[str, str] | None,
    *,
    now: datetime | None = None,
) -> float | None:
    """Parse ``Retry-After`` header as seconds (integer or HTTP-date)."""
    if not headers:
        return None
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(text)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        current = now or datetime.now(timezone.utc)
        return max(0.0, (retry_at - current).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


def openalex_budget_exhausted(headers: Mapping[str, str] | None) -> bool:
    if not headers:
        return False
    remaining = headers.get("X-RateLimit-Remaining")
    if remaining is None:
        return False
    try:
        return float(remaining) <= 0.0
    except (TypeError, ValueError):
        return str(remaining).strip() == "0"


def openalex_anonymous_search_throttled(resp: requests.Response) -> bool:
    """True when OpenAlex blocks anonymous ``?search=`` under load (503 + JSON body)."""
    if resp.status_code != 503:
        return False
    try:
        body = resp.json()
    except (ValueError, TypeError):
        return False
    if not isinstance(body, dict):
        return False
    error = str(body.get("error") or "").strip().lower()
    message = str(body.get("message") or "").strip().lower()
    if error == "search temporarily unavailable":
        return True
    return "anonymous search" in message and "rate-limited" in message


def server_error_backoff_sec(attempt: int) -> float:
    """Exponential backoff with jitter (base 1s, cap ~16s)."""
    base = min(16.0, 1.0 * (2**attempt))
    jitter = random.uniform(0.0, base * 0.25)
    return base + jitter


def server_error_wait_sec(resp: requests.Response, attempt: int) -> float:
    """Prefer ``Retry-After`` on server errors; fall back to exponential backoff."""
    return retry_after_seconds(resp.headers) or server_error_backoff_sec(attempt)


def _sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def _record_circuit_outcome(
    hostname: str,
    status_code: int,
    *,
    budget_exhausted: bool = False,
) -> None:
    """Record one logical-request outcome for the host circuit breaker."""
    get_host_scheduler().record_outcome(
        hostname,
        status_code,
        budget_exhausted=budget_exhausted,
    )


def _fail_openalex_anonymous_search(
    resp: requests.Response,
    *,
    hostname: str,
    metrics_host: str,
) -> None:
    """Treat OpenAlex anonymous-search 503 as quota throttle (no retry storm / circuit trip)."""
    wait_s = retry_after_seconds(resp.headers)
    ttl = int(wait_s) if wait_s is not None else None
    record_http_retry_reason(f"{metrics_host}:budget_exhausted")
    _record_circuit_outcome(
        hostname,
        resp.status_code,
        budget_exhausted=True,
    )
    mark_host_negative(metrics_host, reason="budget_exhausted", ttl_seconds=ttl)
    logger.warning(
        "[HTTP] OpenAlex anonymous search throttled (%s)",
        metrics_host,
    )
    notify_budget_exhausted(metrics_host=metrics_host, kind="daily_quota")
    raise BudgetExhaustedError(host=hostname, metrics_host=metrics_host)


def _guard_negative_cache(*, hostname: str, metrics_host: str) -> None:
    entry = get_host_negative(metrics_host)
    if entry is None:
        return
    record_http_retry_reason(f"{metrics_host}:negative_cache_{entry.reason}")
    if entry.reason == "budget_exhausted":
        notify_budget_exhausted(metrics_host=metrics_host, kind="daily_quota")
        raise BudgetExhaustedError(host=hostname, metrics_host=metrics_host)
    raise HostUnavailableError(
        host=hostname,
        metrics_host=metrics_host,
        reason=entry.reason,
    )


def _execute_once(
    url: str,
    *,
    method: str = "GET",
    hostname: str,
    metrics_host: str,
    is_retry: bool,
    **kwargs: Any,
) -> requests.Response:
    scheduler = get_host_scheduler()
    try:
        scheduler.acquire(hostname)
    except HostCircuitOpenError as exc:
        record_http_retry_reason(f"{metrics_host}:circuit_open")
        raise HostUnavailableError(
            host=hostname,
            metrics_host=metrics_host,
            reason="circuit_open",
        ) from exc
    t0 = time.perf_counter()
    request_fn = requests.post if method.upper() == "POST" else requests.get
    try:
        resp = request_fn(url, **kwargs)
    except Exception:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        record_http_request(
            host=metrics_host,
            status_code=0,
            latency_ms=latency_ms,
            is_retry=is_retry,
            headers=None,
        )
        raise
    latency_ms = (time.perf_counter() - t0) * 1000.0
    record_http_request(
        host=metrics_host,
        status_code=resp.status_code,
        latency_ms=latency_ms,
        is_retry=is_retry,
        headers=resp.headers,
    )
    return resp


def knowledge_get(
    url: str,
    *,
    host: str | None = None,
    is_retry: bool = False,
    **kwargs: Any,
) -> requests.Response:
    """Rate-limited ``requests.get`` with metrics and header-aware retries."""
    _ = is_retry  # Retries are tracked internally; external flag is ignored.
    if kwargs.get("timeout") is None:
        kwargs["timeout"] = DEFAULT_TIMEOUT_SEC
    hostname = host or hostname_from_url(url)
    metrics_host = metrics_host_for(hostname)

    rate_limit_attempts = 0
    server_error_attempts = 0

    while True:
        try:
            get_host_scheduler().ensure_circuit_allows_request(hostname)
        except HostCircuitOpenError as exc:
            record_http_retry_reason(f"{metrics_host}:circuit_open")
            raise HostUnavailableError(
                host=hostname,
                metrics_host=metrics_host,
                reason="circuit_open",
            ) from exc
        _guard_negative_cache(hostname=hostname, metrics_host=metrics_host)
        attempt_is_retry = rate_limit_attempts > 0 or server_error_attempts > 0
        try:
            resp = _execute_once(
                url,
                method="GET",
                hostname=hostname,
                metrics_host=metrics_host,
                is_retry=attempt_is_retry,
                **kwargs,
            )
        except Exception:
            _record_circuit_outcome(hostname, 0)
            raise

        if (
            hostname == OPENALEX_HOST
            and openalex_anonymous_search_throttled(resp)
        ):
            _fail_openalex_anonymous_search(
                resp,
                hostname=hostname,
                metrics_host=metrics_host,
            )

        if resp.status_code == 429:
            if hostname == OPENALEX_HOST and openalex_budget_exhausted(resp.headers):
                record_http_retry_reason(f"{metrics_host}:budget_exhausted")
                _record_circuit_outcome(
                    hostname,
                    resp.status_code,
                    budget_exhausted=True,
                )
                mark_host_negative(metrics_host, reason="budget_exhausted")
                logger.warning("[HTTP] OpenAlex budget exhausted (%s)", metrics_host)
                notify_budget_exhausted(metrics_host=metrics_host, kind="daily_quota")
                raise BudgetExhaustedError(host=hostname, metrics_host=metrics_host)
            if rate_limit_attempts >= MAX_RATE_LIMIT_RETRIES:
                _record_circuit_outcome(hostname, resp.status_code)
                return resp
            wait_s = retry_after_seconds(resp.headers) or DEFAULT_429_FALLBACK_SEC
            record_http_retry_reason(
                f"{metrics_host}:429_retry_after_{wait_s:.1f}s"
            )
            _sleep(wait_s)
            rate_limit_attempts += 1
            continue

        if resp.status_code in SERVER_ERROR_CODES:
            if server_error_attempts >= MAX_SERVER_ERROR_RETRIES:
                _record_circuit_outcome(hostname, resp.status_code)
                return resp
            wait_s = server_error_wait_sec(resp, server_error_attempts)
            record_http_retry_reason(
                f"{metrics_host}:{resp.status_code}_backoff_{wait_s:.1f}s"
            )
            _sleep(wait_s)
            server_error_attempts += 1
            continue

        _record_circuit_outcome(hostname, resp.status_code)
        return resp


def knowledge_post(
    url: str,
    *,
    host: str | None = None,
    is_retry: bool = False,
    **kwargs: Any,
) -> requests.Response:
    """Rate-limited ``requests.post`` with metrics and header-aware retries."""
    _ = is_retry
    if kwargs.get("timeout") is None:
        kwargs["timeout"] = DEFAULT_TIMEOUT_SEC
    hostname = host or hostname_from_url(url)
    metrics_host = metrics_host_for(hostname)

    rate_limit_attempts = 0
    server_error_attempts = 0

    while True:
        try:
            get_host_scheduler().ensure_circuit_allows_request(hostname)
        except HostCircuitOpenError as exc:
            record_http_retry_reason(f"{metrics_host}:circuit_open")
            raise HostUnavailableError(
                host=hostname,
                metrics_host=metrics_host,
                reason="circuit_open",
            ) from exc
        _guard_negative_cache(hostname=hostname, metrics_host=metrics_host)
        attempt_is_retry = rate_limit_attempts > 0 or server_error_attempts > 0
        try:
            resp = _execute_once(
                url,
                method="POST",
                hostname=hostname,
                metrics_host=metrics_host,
                is_retry=attempt_is_retry,
                **kwargs,
            )
        except Exception:
            _record_circuit_outcome(hostname, 0)
            raise

        if resp.status_code == 429:
            if rate_limit_attempts >= MAX_RATE_LIMIT_RETRIES:
                _record_circuit_outcome(hostname, resp.status_code)
                return resp
            wait_s = retry_after_seconds(resp.headers) or DEFAULT_429_FALLBACK_SEC
            record_http_retry_reason(
                f"{metrics_host}:429_retry_after_{wait_s:.1f}s"
            )
            _sleep(wait_s)
            rate_limit_attempts += 1
            continue

        if resp.status_code in SERVER_ERROR_CODES:
            if server_error_attempts >= MAX_SERVER_ERROR_RETRIES:
                _record_circuit_outcome(hostname, resp.status_code)
                return resp
            wait_s = server_error_wait_sec(resp, server_error_attempts)
            record_http_retry_reason(
                f"{metrics_host}:{resp.status_code}_backoff_{wait_s:.1f}s"
            )
            _sleep(wait_s)
            server_error_attempts += 1
            continue

        _record_circuit_outcome(hostname, resp.status_code)
        return resp
