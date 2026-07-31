"""Pure helpers for the Custom sources settings editor."""

from __future__ import annotations

import json
from typing import Any

from core.knowledge.configured_sources import ConfiguredSource
from core.knowledge.types import SERVICE_SCIENTIFIC_EVIDENCE

_DEFAULT_CONNECTOR_ID = "rest_json"
_MCP_CONNECTOR_ID = "mcp"


def parse_mcp_command(text: str) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    if raw.startswith("["):
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError("MCP command must be a JSON array")
        return [str(part) for part in parsed]
    return [part.strip() for part in raw.split(",") if part.strip()]


def configured_source_to_field_values(source: ConfiguredSource) -> dict[str, str]:
    cfg = dict(source.config or {})
    command = cfg.get("command")
    if isinstance(command, list):
        command_text = json.dumps([str(part) for part in command], ensure_ascii=False)
    else:
        command_text = str(command or "")
    return {
        "id": source.id,
        "label": source.label,
        "connector_type": source.connector_type,
        "base_url": str(cfg.get("base_url") or ""),
        "search_path": str(cfg.get("search_path") or ""),
        "mcp_command": command_text,
        "mcp_namespace": str(cfg.get("namespace") or ""),
        "mcp_tool_name": str(cfg.get("tool_name") or ""),
    }


def build_configured_source_from_fields(
    *,
    source_id: str,
    label: str,
    connector_type: str,
    base_url: str = "",
    search_path: str = "",
    mcp_command: str = "",
    mcp_namespace: str = "",
    mcp_tool_name: str = "",
    loaded: ConfiguredSource | None = None,
) -> ConfiguredSource:
    connector = (connector_type or _DEFAULT_CONNECTOR_ID).strip().lower()
    sid = (source_id or "").strip().lower()
    same_record = loaded is not None and loaded.id == sid

    if connector == _MCP_CONNECTOR_ID:
        config = _mcp_config_from_fields(
            sid,
            mcp_command=mcp_command,
            mcp_namespace=mcp_namespace,
            mcp_tool_name=mcp_tool_name,
            loaded=loaded if same_record else None,
        )
        auth = dict(loaded.auth) if same_record else {}
        knowledge_service = loaded.knowledge_service if same_record else SERVICE_SCIENTIFIC_EVIDENCE
        trust_policy = loaded.trust_policy if same_record else "enterprise"
        egress_policy = dict(loaded.egress_policy) if same_record else {}
    elif connector == _DEFAULT_CONNECTOR_ID:
        config = {
            "base_url": base_url.strip(),
            "search_path": search_path.strip() or "/?q={query}",
            "method": "GET",
            "adapter_id": sid,
            "response_mapping": {
                "items_path": "$",
                "title": "$.title",
                "snippet": "$.description",
                "url": "$.url",
            },
        }
        auth = (
            dict(loaded.auth)
            if same_record
            else {"type": "bearer", "credential_ref": sid}
        )
        knowledge_service = (
            loaded.knowledge_service if same_record else SERVICE_SCIENTIFIC_EVIDENCE
        )
        trust_policy = loaded.trust_policy if same_record else "standard"
        egress_policy = dict(loaded.egress_policy) if same_record else {}
    elif same_record and loaded.connector_type == connector:
        config = dict(loaded.config)
        auth = dict(loaded.auth)
        knowledge_service = loaded.knowledge_service
        trust_policy = loaded.trust_policy
        egress_policy = dict(loaded.egress_policy)
    else:
        config = {"adapter_id": sid}
        auth = {}
        knowledge_service = SERVICE_SCIENTIFIC_EVIDENCE
        trust_policy = "standard"
        egress_policy = {}

    return ConfiguredSource(
        id=sid,
        label=label.strip(),
        connector_type=connector,
        knowledge_service=knowledge_service,
        config=config,
        auth=auth,
        egress_policy=egress_policy,
        trust_policy=trust_policy,
        created_at=loaded.created_at if same_record else "",
        version=loaded.version if same_record else 1,
    )


def _mcp_config_from_fields(
    source_id: str,
    *,
    mcp_command: str,
    mcp_namespace: str,
    mcp_tool_name: str,
    loaded: ConfiguredSource | None,
) -> dict[str, Any]:
    cfg = dict(loaded.config) if loaded is not None else {}
    command = parse_mcp_command(mcp_command)
    if command:
        cfg["command"] = command
    namespace = mcp_namespace.strip()
    if namespace:
        cfg["namespace"] = namespace
    tool_name = mcp_tool_name.strip()
    if tool_name:
        cfg["tool_name"] = tool_name
    cfg["adapter_id"] = str(cfg.get("adapter_id") or source_id)
    return cfg
