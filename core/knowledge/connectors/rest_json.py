"""REST/JSON connector — declarative HTTP source execution."""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import urlencode

from core.knowledge.connectors.json_path import extract_json_path
from core.knowledge.credential_resolver import authorization_token
from core.knowledge.egress_policy import EgressPolicy
from core.knowledge.http_client import knowledge_get, knowledge_post

logger = logging.getLogger("Qube.Knowledge.Connectors.REST")


def _render_template(template: str, *, query: str) -> str:
    return (template or "").replace("{query}", query)


def _auth_headers(auth: dict[str, Any] | None) -> dict[str, str]:
    if not auth:
        return {}
    auth_type = str(auth.get("type") or "").strip().lower()
    ref = str(auth.get("credential_ref") or "").strip()
    if not ref:
        return {}
    provider_id = ref if ref.startswith("configured:") else f"configured:{ref}"
    if auth_type == "bearer":
        token = authorization_token(provider_id)
        if token:
            return {"Authorization": f"Bearer {token}"}
    if auth_type == "api_key_header":
        header = str(auth.get("header") or "X-API-Key")
        token = authorization_token(provider_id)
        if token:
            return {header: token}
    if auth_type == "api_key_query":
        return {}
    return {}


def _api_key_query_params(auth: dict[str, Any] | None) -> dict[str, str]:
    if not auth:
        return {}
    if str(auth.get("type") or "").strip().lower() != "api_key_query":
        return {}
    ref = str(auth.get("credential_ref") or "").strip()
    if not ref:
        return {}
    provider_id = ref if ref.startswith("configured:") else f"configured:{ref}"
    token = authorization_token(provider_id)
    param = str(auth.get("param") or "api_key")
    return {param: token} if token else {}


class RestJsonConnector:
    id = "rest_json"

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
        base_url = str(config.get("base_url") or "").rstrip("/")
        search_path = _render_template(str(config.get("search_path") or ""), query=query)
        method = str(config.get("method") or "GET").upper()
        headers = {str(k): str(v) for k, v in (config.get("headers") or {}).items()}
        headers.update(_auth_headers(auth))

        url = f"{base_url}{search_path}" if search_path.startswith("/") else search_path
        if not url.startswith("http"):
            url = f"{base_url}/{search_path.lstrip('/')}"

        extra_params = _api_key_query_params(auth)
        if extra_params and "?" not in url:
            url = f"{url}?{urlencode(extra_params)}"

        try:
            if method == "POST":
                body_template = config.get("body_template")
                body = None
                if isinstance(body_template, dict):
                    body = {
                        str(k): _render_template(str(v), query=query)
                        for k, v in body_template.items()
                    }
                resp = knowledge_post(
                    url,
                    json=body,
                    headers=headers,
                    timeout=timeout,
                    egress_policy=policy,
                )
            else:
                resp = knowledge_get(
                    url,
                    headers=headers,
                    timeout=timeout,
                    egress_policy=policy,
                )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            logger.warning("[REST] request failed: %s", exc)
            return []

        mapping = config.get("response_mapping") or {}
        items_path = str(mapping.get("items_path") or "$")
        items = extract_json_path(payload, items_path)
        if not isinstance(items, list):
            items = [items] if items is not None else []

        rows: list[dict[str, Any]] = []
        adapter_id = str(config.get("adapter_id") or "configured_rest")
        for item in items[: max(1, max_results)]:
            if not isinstance(item, dict):
                continue
            title_path = str(mapping.get("title") or "$.title")
            snippet_path = str(mapping.get("snippet") or "$.snippet")
            url_path = str(mapping.get("url") or "$.url")
            title = str(extract_json_path(item, title_path) or "").strip()
            snippet = str(extract_json_path(item, snippet_path) or "").strip()
            item_url = extract_json_path(item, url_path)
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
                    "retrieval_method": "rest_json",
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
        test_query = str(config.get("test_query") or "test")
        rows = self.execute(
            test_query,
            config=config,
            auth=auth,
            egress_policy=egress_policy,
            max_results=1,
            timeout=timeout,
        )
        if rows:
            return True, f"OK — received {len(rows)} result(s)"
        return False, "No results returned (check URL, mapping, and credentials)"
