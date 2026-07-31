"""Export/import integration consent grants for knowledge pack transfer."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from core.integrations.capabilities.persistence import ConsentStore, integrations_dir
from core.paths import user_data_root

logger = logging.getLogger("Qube.Integrations.ConsentExport")

__all__ = [
    "export_integration_consents",
    "import_integration_consents",
    "list_providers_with_consent",
]


def list_providers_with_consent() -> list[str]:
    root = user_data_root() / "integrations"
    if not root.is_dir():
        return []
    providers: list[str] = []
    for child in root.iterdir():
        if child.is_dir() and (child / "consent.json").is_file():
            providers.append(child.name.lower())
    return sorted(set(providers))


def export_integration_consents() -> dict[str, Any]:
    """Serialize all provider consent files (no secrets — grants only)."""
    payload: dict[str, Any] = {"schema_version": 1, "providers": {}}
    for provider_id in list_providers_with_consent():
        path = integrations_dir(provider_id) / "consent.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("[integrations] skip consent export for %s: %s", provider_id, exc)
            continue
        if isinstance(raw, dict):
            payload["providers"][provider_id] = raw
    return payload


def import_integration_consents(
    payload: dict[str, Any] | None,
    *,
    merge: bool = True,
) -> dict[str, Any]:
    """Import consent grants from a knowledge pack fragment."""
    summary: dict[str, Any] = {"providers_imported": 0, "grants_imported": 0, "errors": []}
    if not isinstance(payload, dict):
        return summary
    providers = payload.get("providers")
    if not isinstance(providers, dict):
        providers = payload if any(isinstance(v, dict) for v in payload.values()) else {}
    if not isinstance(providers, dict):
        return summary

    for provider_id, raw in providers.items():
        pid = str(provider_id or "").strip().lower()
        if not pid or not isinstance(raw, dict):
            continue
        grants = raw.get("grants")
        if not isinstance(grants, list):
            summary["errors"].append(f"{pid}: missing grants list")
            continue
        path = integrations_dir(pid) / "consent.json"
        try:
            if merge and path.exists():
                existing = json.loads(path.read_text(encoding="utf-8"))
                existing_grants = existing.get("grants") if isinstance(existing, dict) else []
                if not isinstance(existing_grants, list):
                    existing_grants = []
                merged_by_urn = {
                    str(item.get("urn")): item
                    for item in existing_grants
                    if isinstance(item, dict) and item.get("urn")
                }
                for grant in grants:
                    if isinstance(grant, dict) and grant.get("urn"):
                        merged_by_urn[str(grant["urn"])] = grant
                raw = {
                    "schema_version": raw.get("schema_version", 1),
                    "provider_id": pid,
                    "grants": list(merged_by_urn.values()),
                }
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            summary["providers_imported"] += 1
            summary["grants_imported"] += len(raw.get("grants") or [])
        except Exception as exc:
            summary["errors"].append(f"{pid}: {exc}")
    return summary
