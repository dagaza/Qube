"""Brave Search API discovery provider — optional fallback when DDG is blocked."""

from __future__ import annotations

from core.knowledge.adapters.brave_search import (
    brave_search_configured,
    search_brave,
)
from core.knowledge.discovery.duckduckgo import format_site_bias_query
from core.knowledge.discovery.types import CandidateUrl, DiscoveryResult
from core.knowledge.search_outcome import build_search_outcome_from_brave


class BraveSearchDiscovery:
    id = "brave_search"

    def discover_full(
        self,
        query: str,
        *,
        max_results: int,
        site_bias: tuple[str, ...] | None = None,
    ) -> DiscoveryResult:
        if not brave_search_configured():
            return DiscoveryResult(
                candidates=(),
                raw_rows=(),
                search_outcome=build_search_outcome_from_brave(
                    [],
                    {"response_kind": "no_credentials", "parsed_rows": 0},
                    candidate_count=0,
                    provider=self.id,
                ),
                provider_id=self.id,
            )

        scoped_query, target_site = format_site_bias_query(query, site_bias)
        rows, inspection = search_brave(
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
        outcome = build_search_outcome_from_brave(
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
