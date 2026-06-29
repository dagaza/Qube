"""Optional transparency enrichments for the knowledge graph."""

from __future__ import annotations

from typing import Any

from core.app_settings import research_map_enabled
from core.knowledge.graph.entities import extract_entity_keys_from_bundle
from core.knowledge.graph.service import find_prior_bundles_by_entities
from core.knowledge.types import EvidenceBundle


def enrich_transparency_with_prior_sessions(
    transparency: dict[str, Any],
    *,
    db: Any,
    session_id: str | None,
    bundle: EvidenceBundle | None,
) -> dict[str, Any]:
    """Append read-only cross-session hints when research map is enabled."""
    if not research_map_enabled() or not transparency or bundle is None:
        return transparency

    entity_keys = extract_entity_keys_from_bundle(bundle)
    priors = find_prior_bundles_by_entities(
        db,
        entity_keys=entity_keys,
        exclude_session_id=session_id,
        limit=3,
    )
    if not priors:
        return transparency

    enriched = dict(transparency)
    enriched["prior_sessions"] = priors
    lines = [str(enriched.get("why_summary") or "").rstrip()]
    lines.append("Prior sessions on related topics:")
    for row in priors:
        query = str(row.get("query_resolved") or "").strip()
        service = str(row.get("knowledge_service") or "")
        if query:
            lines.append(f"  • {query}" + (f" ({service})" if service else ""))
    enriched["why_summary"] = "\n".join(line for line in lines if line)
    return enriched
