"""Knowledge pack import/export — sources, presets, and preferences."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.app_settings import (
    get_knowledge_source_preferences,
    set_knowledge_source_preferences,
)
from core.paths import user_data_root

PACK_VERSION = 1


def export_knowledge_pack(*, include_sources: bool = True, include_presets: bool = True) -> dict[str, Any]:
    """Export knowledge configuration with credentials redacted."""
    pack: dict[str, Any] = {
        "pack_version": PACK_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source_preferences": get_knowledge_source_preferences(),
        "presets": [],
        "sources": [],
    }

    if include_presets:
        from core.knowledge.presets import list_presets

        pack["presets"] = [p.to_dict() for p in list_presets()]

    if include_sources:
        from core.knowledge.configured_sources import list_configured_sources

        for src in list_configured_sources():
            d = src.to_dict()
            if "auth" in d and isinstance(d["auth"], dict):
                d["auth"] = {**d["auth"], "credential_ref": d["auth"].get("credential_ref")}
            pack["sources"].append(d)

    return pack


def import_knowledge_pack(
    pack: dict[str, Any],
    *,
    import_preferences: bool = True,
    import_presets: bool = True,
    import_sources: bool = True,
) -> dict[str, Any]:
    """Import a knowledge pack. Returns summary of actions taken."""
    if not isinstance(pack, dict):
        raise ValueError("Invalid knowledge pack")

    summary: dict[str, Any] = {
        "preferences_imported": False,
        "presets_imported": 0,
        "sources_imported": 0,
        "errors": [],
    }

    if import_preferences:
        prefs = pack.get("source_preferences")
        if isinstance(prefs, dict):
            set_knowledge_source_preferences(prefs)
            summary["preferences_imported"] = True

    if import_presets:
        from core.knowledge.presets import save_preset, KnowledgePreset

        for raw in pack.get("presets") or []:
            if not isinstance(raw, dict):
                continue
            try:
                preset = KnowledgePreset.from_dict(raw)
                save_preset(preset)
                summary["presets_imported"] += 1
            except Exception as exc:
                summary["errors"].append(f"preset: {exc}")

    if import_sources:
        from core.knowledge.configured_sources import save_configured_source, ConfiguredSource

        for raw in pack.get("sources") or []:
            if not isinstance(raw, dict):
                continue
            try:
                source = ConfiguredSource.from_dict(raw)
                save_configured_source(source)
                summary["sources_imported"] += 1
            except Exception as exc:
                summary["errors"].append(f"source: {exc}")

    return summary


def export_knowledge_pack_to_file(path: Path) -> None:
    pack = export_knowledge_pack()
    path.write_text(json.dumps(pack, indent=2, ensure_ascii=False), encoding="utf-8")


def import_knowledge_pack_from_file(path: Path) -> dict[str, Any]:
    from core.knowledge.packs import install_knowledge_pack

    raw = json.loads(path.read_text(encoding="utf-8"))
    return install_knowledge_pack(raw)
