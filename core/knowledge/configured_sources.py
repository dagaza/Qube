"""Configured source instances — user-defined knowledge sources."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Callable
from pathlib import Path

from core.knowledge.connectors.base import get_connector
from core.knowledge.egress_policy import EgressPolicy
from core.knowledge.types import (
    SERVICE_FINANCE_KNOWLEDGE,
    SERVICE_LEGAL_KNOWLEDGE,
    SERVICE_SCIENTIFIC_EVIDENCE,
)
from core.paths import user_data_root

logger = logging.getLogger("Qube.Knowledge.ConfiguredSources")


def sources_dir() -> Path:
    path = user_data_root() / "knowledge" / "sources"
    path.mkdir(parents=True, exist_ok=True)
    return path

SOURCE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,31}$")
ALLOWED_SERVICES = frozenset(
    {
        SERVICE_SCIENTIFIC_EVIDENCE,
        SERVICE_FINANCE_KNOWLEDGE,
        SERVICE_LEGAL_KNOWLEDGE,
    }
)
TRUST_POLICIES = frozenset({"standard", "local_only", "enterprise"})


@dataclass
class ConfiguredSource:
    id: str
    label: str
    connector_type: str
    knowledge_service: str = SERVICE_SCIENTIFIC_EVIDENCE
    config: dict[str, Any] = field(default_factory=dict)
    auth: dict[str, Any] = field(default_factory=dict)
    egress_policy: dict[str, Any] = field(default_factory=dict)
    trust_policy: str = "standard"
    created_at: str = ""
    version: int = 1

    def __post_init__(self) -> None:
        self.id = (self.id or "").strip().lower()
        self.label = (self.label or self.id).strip()
        self.connector_type = (self.connector_type or "").strip().lower()
        self.knowledge_service = (self.knowledge_service or SERVICE_SCIENTIFIC_EVIDENCE).strip().lower()
        self.trust_policy = (self.trust_policy or "standard").strip().lower()
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.egress_policy:
            if self.connector_type in {"sqlite", "filesystem"}:
                self.egress_policy = EgressPolicy.local_connector_default().__dict__
            else:
                self.egress_policy = EgressPolicy.configured_source_default().__dict__

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ConfiguredSource:
        return cls(
            id=str(raw.get("id") or ""),
            label=str(raw.get("label") or ""),
            connector_type=str(raw.get("connector_type") or ""),
            knowledge_service=str(raw.get("knowledge_service") or SERVICE_SCIENTIFIC_EVIDENCE),
            config=dict(raw.get("config") or {}),
            auth=dict(raw.get("auth") or {}),
            egress_policy=dict(raw.get("egress_policy") or {}),
            trust_policy=str(raw.get("trust_policy") or "standard"),
            created_at=str(raw.get("created_at") or ""),
            version=int(raw.get("version") or 1),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "connector_type": self.connector_type,
            "knowledge_service": self.knowledge_service,
            "config": dict(self.config),
            "auth": dict(self.auth),
            "egress_policy": dict(self.egress_policy),
            "trust_policy": self.trust_policy,
            "created_at": self.created_at,
            "version": self.version,
        }

    def validate(self) -> None:
        if not SOURCE_ID_RE.match(self.id):
            raise ValueError(f"Invalid source id: {self.id!r}")
        if not self.label:
            raise ValueError("Source label is required")
        if get_connector(self.connector_type) is None:
            raise ValueError(f"Unknown connector type: {self.connector_type}")
        if self.knowledge_service not in ALLOWED_SERVICES:
            raise ValueError(f"Unsupported knowledge service: {self.knowledge_service}")
        if self.trust_policy not in TRUST_POLICIES:
            raise ValueError(f"Unknown trust policy: {self.trust_policy}")
        cfg = dict(self.config)
        cfg.setdefault("adapter_id", self.id)
        self.config = cfg

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _source_path(source_id: str) -> Any:
    return sources_dir() / f"{source_id.strip().lower()}.json"


def clear_configured_source_search_cache() -> None:
    """Drop cached search callables after on-disk source changes."""
    _configured_search_fn.cache_clear()


def list_configured_sources() -> list[ConfiguredSource]:
    out: list[ConfiguredSource] = []
    for path in sorted(sources_dir().glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            source = ConfiguredSource.from_dict(raw)
            source.validate()
            out.append(source)
        except Exception:
            continue
    return out


def _mcp_namespace_from_raw(raw: dict[str, Any], *, source_id: str) -> str:
    cfg = dict(raw.get("config") or {})
    return str(cfg.get("namespace") or cfg.get("adapter_id") or source_id).strip().lower()


def inspect_configured_mcp_namespace(namespace: str) -> tuple[str, str, str]:
    """Return ``(state, source_id, detail)`` for an MCP namespace.

    ``state`` is one of ``ok``, ``missing``, or ``invalid``.
    """
    want = (namespace or "").strip().lower()
    if not want:
        return "missing", "", "namespace is empty"

    for source in list_configured_sources():
        if source.connector_type != "mcp":
            continue
        cfg = dict(source.config or {})
        src_ns = str(cfg.get("namespace") or cfg.get("adapter_id") or source.id).strip().lower()
        if src_ns != want:
            continue
        command = cfg.get("command")
        if not isinstance(command, list) or not command:
            return "invalid", source.id, "MCP command is not configured"
        return "ok", source.id, ""

    for path in sorted(sources_dir().glob("*.json")):
        source_id = path.stem.lower()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            if source_id == want or want in path.name.lower():
                return "invalid", source_id, f"JSON parse error: {exc}"
            continue
        connector = str(raw.get("connector_type") or "").strip().lower()
        if connector != "mcp":
            continue
        src_ns = _mcp_namespace_from_raw(raw, source_id=source_id)
        if src_ns != want:
            continue
        try:
            source = ConfiguredSource.from_dict(raw)
            source.validate()
            command = dict(source.config or {}).get("command")
            if not isinstance(command, list) or not command:
                return "invalid", source_id or src_ns, "MCP command is not configured"
        except Exception as exc:
            return "invalid", source_id or src_ns, str(exc)
    return "missing", "", ""


def load_configured_source(source_id: str) -> ConfiguredSource | None:
    path = _source_path(source_id)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        source = ConfiguredSource.from_dict(raw)
        source.validate()
        return source
    except Exception:
        return None


def save_configured_source(source: ConfiguredSource) -> None:
    source.validate()
    path = _source_path(source.id)
    path.write_text(
        json.dumps(source.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    clear_configured_source_search_cache()


def delete_configured_source(source_id: str) -> bool:
    path = _source_path(source_id)
    if path.is_file():
        path.unlink()
        clear_configured_source_search_cache()
        return True
    return False


def test_configured_source(source: ConfiguredSource, *, timeout: float = 10.0) -> tuple[bool, str]:
    connector = get_connector(source.connector_type)
    if connector is None:
        return False, f"Unknown connector: {source.connector_type}"
    return connector.test_connection(
        config=source.config,
        auth=source.auth,
        egress_policy=source.egress_policy,
        timeout=timeout,
    )


def execute_configured_source(
    source_id: str,
    query: str,
    *,
    max_results: int = 3,
    timeout: float = 10.0,
) -> list[dict[str, Any]]:
    source = load_configured_source(source_id)
    if source is None:
        return []
    connector = get_connector(source.connector_type)
    if connector is None:
        return []
    rows = connector.execute(
        query,
        config=source.config,
        auth=source.auth,
        egress_policy=source.egress_policy,
        max_results=max_results,
        timeout=timeout,
    )
    for row in rows:
        row.setdefault("_adapter", source.id)
        row.setdefault("_source_kind", "configured")
        row.setdefault("_connector_type", source.connector_type)
        row.setdefault("_config_hash", source.config_hash())
    return rows


@lru_cache(maxsize=128)
def _configured_search_fn(source_id: str) -> Callable[..., list[dict[str, Any]]] | None:
    if load_configured_source(source_id) is None:
        return None

    def _search(query: str, *, max_results: int = 3, timeout: float = 10.0) -> list[dict[str, Any]]:
        return execute_configured_source(
            source_id,
            query,
            max_results=max_results,
            timeout=timeout,
        )

    return _search


def get_configured_source_fn(source_id: str) -> Callable[..., list[dict[str, Any]]] | None:
    return _configured_search_fn((source_id or "").strip().lower())


def resolve_configured_credential(provider_id: str) -> str | None:
    ref = (provider_id or "").strip()
    if ref.startswith("configured:"):
        ref = ref.split(":", 1)[1]
    secret = resolve_secret(f"configured:{ref}")
    if secret:
        return secret
    from core.app_settings import get_knowledge_provider_credentials

    creds = get_knowledge_provider_credentials()
    entry = creds.get(f"configured:{ref}") or creds.get(ref)
    if isinstance(entry, dict):
        key = str(entry.get("api_key") or "").strip()
        return key or None
    return None
