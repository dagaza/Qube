"""DuckDuckGo discovery provider — wraps existing SERP adapter."""

from __future__ import annotations

from core.knowledge.adapters.duckduckgo import search_duckduckgo_detailed
from core.knowledge.discovery.types import CandidateUrl, DiscoveryResult
from core.knowledge.search_outcome import build_search_outcome_from_ddg


def format_site_bias_query(
    query: str,
    site_bias: tuple[str, ...] | None,
) -> tuple[str, str | None]:
    """Return (query, target_site) for DuckDuckGo search.

    Single-domain bias uses the adapter ``target_site`` param. Multiple domains
    are OR-joined into the query string per the web fetch plan.
    """
    sites = tuple(s.strip() for s in (site_bias or ()) if (s or "").strip())
    if not sites:
        return query, None
    if len(sites) == 1:
        return query, sites[0]
    clause = " OR ".join(f"site:{site}" for site in sites)
    return f"{clause} {query}", None


class DuckDuckGoDiscovery:
    id = "duckduckgo"

    def discover_full(
        self,
        query: str,
        *,
        max_results: int,
        site_bias: tuple[str, ...] | None = None,
    ) -> DiscoveryResult:
        scoped_query, target_site = format_site_bias_query(query, site_bias)
        rows, inspection = search_duckduckgo_detailed(
            scoped_query,
            max_results=max_results,
            target_site=target_site,
        )
        candidates: list[CandidateUrl] = []
        for rank, row in enumerate(rows):
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
        outcome = build_search_outcome_from_ddg(
            rows,
            inspection,
            candidate_count=len(candidates),
            provider=self.id,
        )
        pace_wait_ms = int((inspection or {}).get("pace_wait_ms") or 0)
        return DiscoveryResult(
            candidates=tuple(candidates),
            raw_rows=tuple(dict(r) for r in rows if isinstance(r, dict)),
            search_outcome=outcome,
            provider_id=self.id,
            discovery_pace_wait_ms=pace_wait_ms,
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
