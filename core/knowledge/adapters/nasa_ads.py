"""NASA ADS adapter stub — requires ADS API token (not enabled in v1).

ADS is a valuable astrophysics index but requires a personal API key and is not
unrestricted open access. Qube prioritizes arXiv + INSPIRE-HEP + OpenAlex for
physics until a user opts in with ``QUBE_ADS_API_KEY``.
"""

from __future__ import annotations

import logging
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query

logger = logging.getLogger("Qube.Knowledge.NASA_ADS")

ADAPTER_ID = "nasa_ads"
RETRIEVAL_METHOD = "ads_search"


def search_nasa_ads(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Placeholder — catalog stub only; not registered for live retrieval."""
    _ = max_results
    q = sanitize_api_query(query)
    if q:
        logger.debug("[NASA ADS] live search unavailable (requires API key; catalog stub)")
    return []
