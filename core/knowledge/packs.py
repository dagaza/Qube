"""Knowledge pack validation and installation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from core.knowledge.knowledge_pack import import_knowledge_pack

PACK_FORMAT = "qube_knowledge_pack"
PACK_FORMAT_VERSION = 1


@dataclass(frozen=True)
class KnowledgePackManifest:
    format: str
    version: int
    name: str
    publisher: str
    created_at: str
    signature: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> KnowledgePackManifest:
        return cls(
            format=str(raw.get("format") or PACK_FORMAT),
            version=int(raw.get("version") or PACK_FORMAT_VERSION),
            name=str(raw.get("name") or "Unnamed pack"),
            publisher=str(raw.get("publisher") or "community"),
            created_at=str(raw.get("created_at") or datetime.now(timezone.utc).isoformat()),
            signature=str(raw.get("signature") or "") or None,
        )


def validate_knowledge_pack(pack: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(pack, dict):
        return ["Pack must be a JSON object"]
    manifest_raw = pack.get("manifest")
    if isinstance(manifest_raw, dict):
        manifest = KnowledgePackManifest.from_dict(manifest_raw)
        if manifest.format != PACK_FORMAT:
            errors.append(f"Unsupported pack format: {manifest.format}")
    for key in ("presets", "sources"):
        items = pack.get(key)
        if items is not None and not isinstance(items, list):
            errors.append(f"{key} must be a list")
    return errors


def install_knowledge_pack(pack: dict[str, Any]) -> dict[str, Any]:
    errors = validate_knowledge_pack(pack)
    if errors:
        return {"installed": False, "errors": errors}
    summary = import_knowledge_pack(pack)
    summary["installed"] = not summary.get("errors")
    return summary


def build_enterprise_pack(
    *,
    name: str,
    publisher: str,
    presets: list[dict[str, Any]] | None = None,
    sources: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    from core.knowledge.knowledge_pack import export_knowledge_pack

    pack = export_knowledge_pack()
    pack["manifest"] = {
        "format": PACK_FORMAT,
        "version": PACK_FORMAT_VERSION,
        "name": name,
        "publisher": publisher,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trust_policy": "enterprise",
    }
    if presets is not None:
        pack["presets"] = presets
    if sources is not None:
        pack["sources"] = sources
    return pack


def load_pack_from_json(text: str) -> dict[str, Any]:
    return json.loads(text)
