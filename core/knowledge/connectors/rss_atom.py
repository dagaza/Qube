"""RSS/Atom feed connector."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Any

from core.knowledge.egress_policy import EgressPolicy
from core.knowledge.http_client import knowledge_get

logger = logging.getLogger("Qube.Knowledge.Connectors.RSS")


def _local(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


class RssAtomConnector:
    id = "rss_atom"

    def execute(
        self,
        query: str,
        *,
        config: dict[str, Any],
        auth: dict[str, Any] | None = None,
        egress_policy: dict[str, Any] | None = None,
        max_results: int = 3,
        timeout: float = 10.0,
    ) -> list[dict[str, Any]]:
        _ = auth
        policy = EgressPolicy.from_dict(egress_policy)
        feed_url = str(config.get("feed_url") or "").strip()
        if not feed_url:
            return []
        q = (query or "").strip().lower()
        adapter_id = str(config.get("adapter_id") or "configured_rss")
        try:
            resp = knowledge_get(feed_url, timeout=timeout, egress_policy=policy)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
        except Exception as exc:
            logger.warning("[RSS] feed fetch failed: %s", exc)
            return []

        rows: list[dict[str, Any]] = []
        for item in root.iter():
            if _local(item.tag) not in {"item", "entry"}:
                continue
            title = ""
            link = ""
            summary = ""
            for child in item:
                name = _local(child.tag)
                text = (child.text or "").strip()
                if name == "title":
                    title = text
                elif name in {"link", "id"} and not link:
                    link = text or child.get("href") or ""
                elif name in {"description", "summary", "content"}:
                    summary = text
            haystack = f"{title} {summary}".lower()
            if q and q not in haystack:
                continue
            if not title and not summary:
                continue
            rows.append(
                {
                    "title": title or summary[:120],
                    "snippet": (summary or title)[:600],
                    "full_text": None,
                    "url": link or None,
                    "_adapter": adapter_id,
                    "retrieval_method": "rss_atom",
                }
            )
            if len(rows) >= max(1, max_results):
                break
        return rows

    def test_connection(
        self,
        *,
        config: dict[str, Any],
        auth: dict[str, Any] | None = None,
        egress_policy: dict[str, Any] | None = None,
        timeout: float = 10.0,
    ) -> tuple[bool, str]:
        rows = self.execute(
            "",
            config=config,
            auth=auth,
            egress_policy=egress_policy,
            max_results=1,
            timeout=timeout,
        )
        if rows:
            return True, "OK — feed parsed"
        return False, "Feed returned no items"
