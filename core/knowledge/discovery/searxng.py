"""SearXNG discovery provider — bring-your-own meta-search (R8)."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urljoin

import requests

from core.knowledge.discovery.duckduckgo import format_site_bias_query
from core.knowledge.discovery.types import CandidateUrl, DiscoveryResult
from core.knowledge.search_outcome import SearchOutcome, SearchOutcomeKind

logger = logging.getLogger("Qube.Knowledge.Discovery.SearXNG")

SEARXNG_DISCOVERY_PROVIDER_ID = "searxng"
DEFAULT_SEARXNG_TIMEOUT_SEC = 10.0


def get_searxng_base_url() -> str:
    from core.app_settings import get_discovery_searxng_base_url

    return (get_discovery_searxng_base_url() or "").strip().rstrip("/")


def searxng_configured() -> bool:
    return bool(get_searxng_base_url())


def _searxng_headers() -> dict[str, str]:
    from core.knowledge.credentials import resolve_credential

    headers = {
        "User-Agent": "Qube/1.0 (local assistant)",
        "Accept": "application/json",
    }
    secret = (resolve_credential(SEARXNG_DISCOVERY_PROVIDER_ID).secret or "").strip()
    if secret:
        headers["Authorization"] = f"Bearer {secret}"
    return headers


def search_searxng(
    query: str,
    *,
    max_results: int = 5,
    target_site: str | None = None,
    timeout: float = DEFAULT_SEARXNG_TIMEOUT_SEC,
    base_url: str | None = None,
    api_key: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resolved_base = (base_url if base_url is not None else get_searxng_base_url()).strip().rstrip("/")
    if not resolved_base:
        return [], {
            "response_kind": "no_credentials",
            "http_status": None,
            "parsed_rows": 0,
        }

    scoped_query = query
    if target_site:
        scoped_query = f"site:{target_site} {query}"

    endpoint = urljoin(resolved_base + "/", "search")
    headers = _searxng_headers() if api_key is None else {
        "User-Agent": "Qube/1.0 (local assistant)",
        "Accept": "application/json",
        **(
            {"Authorization": f"Bearer {api_key.strip()}"}
            if (api_key or "").strip()
            else {}
        ),
    }
    try:
        response = requests.get(
            endpoint,
            params={
                "q": scoped_query,
                "format": "json",
                "language": "en",
            },
            headers=headers,
            timeout=timeout,
        )
        http_status = response.status_code
        if http_status in {401, 403}:
            return [], {
                "response_kind": "auth_error",
                "http_status": http_status,
                "parsed_rows": 0,
            }
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logger.warning("[SearXNG] search failed: %s", exc)
        return [], {
            "response_kind": "network_error",
            "http_status": None,
            "parsed_rows": 0,
        }

    raw_results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(raw_results, list):
        raw_results = []

    rows: list[dict[str, Any]] = []
    for item in raw_results[: max(1, max_results)]:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if not url.startswith(("http://", "https://")):
            continue
        rows.append(
            {
                "title": str(item.get("title") or "").strip(),
                "snippet": str(item.get("content") or item.get("snippet") or "").strip(),
                "url": url,
            }
        )

    response_kind = "serp" if rows else "no_results"
    return rows, {
        "response_kind": response_kind,
        "http_status": http_status,
        "parsed_rows": len(rows),
    }


def build_search_outcome_from_searxng(
    rows: list[dict[str, Any]] | None,
    inspection: dict[str, Any] | None,
    *,
    candidate_count: int = 0,
    provider: str = SEARXNG_DISCOVERY_PROVIDER_ID,
) -> SearchOutcome:
    inspect = dict(inspection or {})
    response_kind = str(inspect.get("response_kind") or "").strip().lower()
    http_status = inspect.get("http_status")

    if response_kind == "no_credentials":
        kind = SearchOutcomeKind.NO_RESULTS
        recovery = "Add your SearXNG base URL in Settings → Knowledge → Web search discovery."
    elif response_kind == "auth_error":
        kind = SearchOutcomeKind.NETWORK_ERROR
        recovery = "Check SearXNG API key or instance authentication."
    elif response_kind == "network_error":
        kind = SearchOutcomeKind.NETWORK_ERROR
        recovery = "Check SearXNG instance URL and network connectivity."
    elif candidate_count > 0:
        kind = SearchOutcomeKind.SERP_SUCCESS
        recovery = None
    elif rows:
        kind = SearchOutcomeKind.NO_CANDIDATES
        recovery = "SearXNG rows lacked parseable URLs."
    else:
        kind = SearchOutcomeKind.NO_RESULTS
        recovery = None

    return SearchOutcome(
        kind=kind,
        provider=provider,
        http_status=int(http_status) if http_status is not None else None,
        parsed_rows=int(inspect.get("parsed_rows") or len(rows or [])),
        candidate_count=int(candidate_count),
        recovery_hint=recovery,
    )


class SearXNGDiscovery:
    id = SEARXNG_DISCOVERY_PROVIDER_ID

    def discover_full(
        self,
        query: str,
        *,
        max_results: int,
        site_bias: tuple[str, ...] | None = None,
    ) -> DiscoveryResult:
        scoped_query, target_site = format_site_bias_query(query, site_bias)
        rows, inspection = search_searxng(
            scoped_query,
            max_results=max_results,
            target_site=target_site,
        )
        safe_rows = [dict(r) for r in rows if isinstance(r, dict)]
        candidates: list[CandidateUrl] = []
        for rank, row in enumerate(safe_rows):
            url = str((row or {}).get("url") or "").strip()
            if not url.startswith(("http://", "https://")):
                continue
            candidates.append(
                CandidateUrl(
                    url=url,
                    title=str((row or {}).get("title") or "").strip() or None,
                    snippet=str((row or {}).get("snippet") or "").strip() or None,
                    source=self.id,
                    rank=rank,
                )
            )
        outcome = build_search_outcome_from_searxng(
            safe_rows,
            inspection,
            candidate_count=len(candidates),
            provider=self.id,
        )
        return DiscoveryResult(
            candidates=tuple(candidates),
            raw_rows=tuple(safe_rows),
            search_outcome=outcome,
            provider_id=self.id,
        )

    def discover(
        self,
        query: str,
        *,
        max_results: int,
        site_bias: tuple[str, ...] | None = None,
    ) -> list[CandidateUrl]:
        return list(
            self.discover_full(
                query,
                max_results=max_results,
                site_bias=site_bias,
            ).candidates
        )
