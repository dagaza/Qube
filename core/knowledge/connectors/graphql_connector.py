"""GraphQL connector — POST query execution."""

from __future__ import annotations

import logging
from typing import Any

from core.knowledge.connectors.json_path import extract_json_path
from core.knowledge.connectors.rest_json import _auth_headers
from core.knowledge.egress_policy import EgressPolicy
from core.knowledge.http_client import knowledge_post

logger = logging.getLogger("Qube.Knowledge.Connectors.GraphQL")


class GraphQLConnector:
    id = "graphql"
    trust_policy = "enterprise"

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
        policy = EgressPolicy.from_dict(egress_policy)
        endpoint = str(config.get("endpoint") or "").strip()
        gql_template = str(config.get("query_template") or "").strip()
        adapter_id = str(config.get("adapter_id") or "configured_graphql")
        if not endpoint or not gql_template:
            return []

        gql = gql_template.replace("{query}", query.replace('"', '\\"'))
        headers = {str(k): str(v) for k, v in (config.get("headers") or {}).items()}
        headers.update(_auth_headers(auth))
        headers.setdefault("Content-Type", "application/json")

        try:
            resp = knowledge_post(
                endpoint,
                json={"query": gql},
                headers=headers,
                timeout=timeout,
                egress_policy=policy,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            logger.warning("[GraphQL] request failed: %s", exc)
            return []

        mapping = config.get("response_mapping") or {}
        items_path = str(mapping.get("items_path") or "$.data.results")
        items = extract_json_path(payload, items_path)
        if not isinstance(items, list):
            items = [items] if items is not None else []

        rows: list[dict[str, Any]] = []
        for item in items[: max(1, max_results)]:
            if not isinstance(item, dict):
                continue
            title = str(extract_json_path(item, str(mapping.get("title") or "$.title")) or "").strip()
            snippet = str(extract_json_path(item, str(mapping.get("snippet") or "$.snippet")) or "").strip()
            item_url = extract_json_path(item, str(mapping.get("url") or "$.url"))
            item_url = str(item_url).strip() if item_url else None
            if not title and not snippet:
                continue
            rows.append(
                {
                    "title": title or snippet[:120],
                    "snippet": snippet[:600],
                    "full_text": None,
                    "url": item_url,
                    "_adapter": adapter_id,
                    "retrieval_method": "graphql",
                }
            )
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
            str(config.get("test_query") or "test"),
            config=config,
            auth=auth,
            egress_policy=egress_policy,
            max_results=1,
            timeout=timeout,
        )
        if rows:
            return True, "OK — GraphQL query returned results"
        return False, "GraphQL query returned no results"
