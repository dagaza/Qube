"""Wikipedia API discovery provider — official search, no HTML scraping."""

from __future__ import annotations

from core.knowledge.adapters.wikipedia_api import search_wikipedia
from core.knowledge.discovery.types import CandidateUrl, DiscoveryResult
from core.knowledge.search_outcome import build_search_outcome_from_provider_rows


class WikipediaDiscovery:
    id = "wikipedia"

    def discover_full(
        self,
        query: str,
        *,
        max_results: int,
        site_bias: tuple[str, ...] | None = None,
    ) -> DiscoveryResult:
        _ = site_bias  # Wikipedia API does not support DDG-style site: filters.
        rows = search_wikipedia(query, max_results=max_results)
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
        outcome = build_search_outcome_from_provider_rows(
            safe_rows,
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
