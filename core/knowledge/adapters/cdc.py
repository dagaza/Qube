"""CDC adapter — Content Services media search and Open Data catalog discovery."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from core.knowledge.adapters.query_sanitize import sanitize_api_query
from core.knowledge.http_client import BudgetExhaustedError, knowledge_get

logger = logging.getLogger("Qube.Knowledge.CDC")

ADAPTER_ID = "cdc"
RETRIEVAL_METHOD = "cdc_content_search"
MEDIA_URL = "https://tools.cdc.gov/api/v2/resources/media"
CATALOG_URL = "https://data.cdc.gov/api/catalog/v1"
USER_AGENT = "Qube/1.0 (local@qube.app)"
_TOKEN_SPLIT = re.compile(r"\s+")


def _fixture_search_path(name: str) -> Path | None:
    path = (
        Path(__file__).resolve().parents[3]
        / "eval"
        / "fixtures"
        / "knowledge"
        / name
    )
    return path if path.is_file() else None


def _use_fixtures() -> bool:
    return os.environ.get("QUBE_KNOWLEDGE_FIXTURES", "").strip() == "1"


def _headers() -> dict[str, str]:
    return {"User-Agent": USER_AGENT, "Accept": "application/json"}


def _topic_label(search_query: str) -> str:
    words = [
        word
        for word in _TOKEN_SPLIT.split(sanitize_api_query(search_query))
        if word
    ]
    if not words:
        return ""
    return " ".join(word[:1].upper() + word[1:] for word in words[:4])


def _row_from_media(item: dict[str, Any]) -> dict[str, Any] | None:
    title = str(item.get("name") or item.get("title") or "").strip()
    if not title:
        return None
    description = str(item.get("description") or item.get("subTitle") or "").strip()
    snippet = description[:600] if description else title
    media_id = item.get("id")
    source_url = str(item.get("sourceUrl") or item.get("url") or "").strip()
    if source_url.startswith("http"):
        url = source_url
    elif media_id is not None:
        url = f"https://www.cdc.gov/media/releases/{media_id}.html"
    else:
        url = "https://www.cdc.gov/"
    topics = [
        str(tag.get("name") or tag).strip()
        for tag in (item.get("tags") or [])
        if isinstance(tag, dict) and str(tag.get("name") or "").strip()
    ]
    return {
        "title": title,
        "snippet": snippet,
        "full_text": description or None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": ("Centers for Disease Control and Prevention",),
        "venue": "CDC",
        "publication_date": str(item.get("datePublished") or item.get("dateCreated") or "")[:10] or None,
        "document_type": "health_guidance",
        "topics": tuple(topics[:5]),
        "media_type": str(item.get("mediaType") or "").strip() or None,
        "retrieval_method": RETRIEVAL_METHOD,
    }


def _row_from_catalog(item: dict[str, Any]) -> dict[str, Any] | None:
    resource = item.get("resource") if isinstance(item.get("resource"), dict) else item
    if not isinstance(resource, dict):
        return None
    title = str(resource.get("name") or "").strip()
    if not title:
        return None
    description = str(resource.get("description") or "").strip()
    dataset_id = str(resource.get("id") or "").strip()
    url = str(resource.get("link") or resource.get("webUri") or "").strip()
    if not url and dataset_id:
        url = f"https://data.cdc.gov/d/{dataset_id}"
    if not url:
        url = "https://data.cdc.gov/"
    snippet = description[:600] if description else title
    return {
        "title": title,
        "snippet": snippet,
        "full_text": description or None,
        "url": url,
        "_adapter": ADAPTER_ID,
        "authors": ("Centers for Disease Control and Prevention",),
        "venue": "CDC Open Data",
        "publication_date": str(resource.get("updatedAt") or resource.get("dataUpdatedAt") or "")[:10] or None,
        "document_type": "government_publication",
        "dataset_id": dataset_id or None,
        "retrieval_method": "cdc_open_data_search",
    }


def _fetch_media_results(
    search_query: str,
    *,
    max_results: int,
    timeout: float,
) -> list[dict[str, Any]]:
    q = sanitize_api_query(search_query)
    attempts: list[dict[str, str | int]] = [
        {"q": q, "max": max(1, min(max_results, 10))},
    ]
    topic = _topic_label(q)
    if topic and topic.lower() != q.lower():
        attempts.append({"topic": topic, "max": max(1, min(max_results, 10))})

    for params in attempts:
        try:
            resp = knowledge_get(
                MEDIA_URL,
                params=params,
                headers=_headers(),
                timeout=timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
            if not isinstance(payload, dict):
                continue
            results = payload.get("results") or []
            if isinstance(results, list) and results:
                return [item for item in results if isinstance(item, dict)]
        except BudgetExhaustedError:
            raise
        except Exception as exc:
            logger.debug("[CDC] media search attempt failed (%s): %s", params, exc)
    return []


def _fetch_catalog_results(
    search_query: str,
    *,
    max_results: int,
    timeout: float,
) -> list[dict[str, Any]]:
    q = sanitize_api_query(search_query)
    if not q:
        return []
    try:
        resp = knowledge_get(
            CATALOG_URL,
            params={"q": q, "limit": max(1, min(max_results, 10))},
            headers=_headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict):
            results = payload.get("results") or []
            return [item for item in results if isinstance(item, dict)]
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
    except BudgetExhaustedError:
        raise
    except Exception as exc:
        logger.debug("[CDC] catalog search failed: %s", exc)
    return []


def fetch_search_results(
    search_query: str,
    *,
    max_results: int = 3,
    timeout: float = 20.0,
) -> dict[str, Any]:
    q = sanitize_api_query(search_query)
    if _use_fixtures():
        fixture = _fixture_search_path("cdc_search_diabetes.json")
        if fixture is not None:
            try:
                return json.loads(fixture.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("[CDC] fixture load failed: %s", exc)

    if not q:
        return {"results": [], "source": "media"}

    try:
        media = _fetch_media_results(q, max_results=max_results, timeout=timeout)
        if media:
            return {"results": media[: max(1, max_results)], "source": "media"}
        catalog = _fetch_catalog_results(q, max_results=max_results, timeout=timeout)
        return {"results": catalog[: max(1, max_results)], "source": "catalog"}
    except BudgetExhaustedError:
        logger.warning("[CDC] budget exhausted; skipping retry")
        return {"results": [], "source": "media"}
    except Exception as exc:
        logger.warning("[CDC] search failed: %s", exc)
        return {"results": [], "source": "media"}


def search_cdc(
    query: str,
    *,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Search CDC health content and open-data catalog entries."""
    q = sanitize_api_query(query)
    if not q or max_results <= 0:
        return []

    payload = fetch_search_results(q, max_results=max_results)
    source = str(payload.get("source") or "media")
    rows: list[dict[str, Any]] = []
    for item in payload.get("results") or []:
        if not isinstance(item, dict):
            continue
        row = (
            _row_from_catalog(item)
            if source == "catalog"
            else _row_from_media(item)
        )
        if row is not None:
            rows.append(row)
        if len(rows) >= max_results:
            break
    return rows
