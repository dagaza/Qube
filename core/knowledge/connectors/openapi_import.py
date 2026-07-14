"""OpenAPI 3 import — subset for REST JSON source instances."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse


def parse_openapi_document(raw: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw, str):
        doc = json.loads(raw)
    else:
        doc = raw
    if not isinstance(doc, dict):
        raise ValueError("Invalid OpenAPI document")
    return doc


def list_get_endpoints(doc: dict[str, Any]) -> list[dict[str, Any]]:
    servers = doc.get("servers") or []
    base_url = ""
    if servers and isinstance(servers[0], dict):
        base_url = str(servers[0].get("url") or "").rstrip("/")
    paths = doc.get("paths") or {}
    endpoints: list[dict[str, Any]] = []
    if not isinstance(paths, dict):
        return endpoints
    for path, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        get_op = methods.get("get")
        if not isinstance(get_op, dict):
            continue
        endpoints.append(
            {
                "path": str(path),
                "operation_id": str(get_op.get("operationId") or path),
                "summary": str(get_op.get("summary") or ""),
                "base_url": base_url,
            }
        )
    return endpoints


def source_instance_from_openapi(
    doc: dict[str, Any],
    *,
    endpoint_path: str,
    source_id: str,
    label: str,
    knowledge_service: str = "scientific_evidence",
) -> dict[str, Any]:
    servers = doc.get("servers") or []
    base_url = ""
    if servers and isinstance(servers[0], dict):
        base_url = str(servers[0].get("url") or "").rstrip("/")
    path = (endpoint_path or "").strip()
    if not path.startswith("/"):
        path = f"/{path}"

    security = doc.get("security") or []
    auth: dict[str, Any] = {"type": "none"}
    if security:
        auth = {"type": "bearer", "credential_ref": source_id}

    return {
        "id": source_id,
        "label": label,
        "connector_type": "rest_json",
        "knowledge_service": knowledge_service,
        "config": {
            "base_url": base_url,
            "search_path": f"{path}?q={{query}}",
            "method": "GET",
            "adapter_id": source_id,
            "test_query": "test",
            "response_mapping": {
                "items_path": "$",
                "title": "$.title",
                "snippet": "$.description",
                "url": "$.url",
            },
        },
        "auth": auth,
        "egress_policy": {
            "allow_http": False,
            "allow_localhost": False,
            "max_response_bytes": 524288,
        },
        "trust_policy": "standard",
        "version": 1,
    }


def infer_base_url_from_openapi_url(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
