"""Entity keys for knowledge graph nodes (delegates to entity resolution)."""

from __future__ import annotations

from core.knowledge.entities.resolve import (
    collect_bundle_entity_ids,
    resolve_entities_for_source,
)
from core.knowledge.types import EvidenceBundle, EvidenceObject


def entity_id_for_key(kind: str, key: str) -> str:
    from core.knowledge.entities.ids import make_entity_id

    return make_entity_id(kind, key)


def extract_entity_keys_from_text(text: str) -> set[str]:
    from core.knowledge.entities.resolve import resolve_entities_from_text

    return set(resolve_entities_from_text(text))


def extract_topic_keys_from_query(query: str, *, max_topics: int = 4) -> set[str]:
    import re

    from core.knowledge.entities.ids import make_entity_id

    stopwords = frozenset(
        {"the", "and", "for", "with", "what", "how", "from", "that", "this"}
    )
    tokens = re.findall(r"[a-z0-9]{3,}", (query or "").lower())
    topics: list[str] = []
    for token in tokens:
        if token in stopwords:
            continue
        if token not in topics:
            topics.append(token)
        if len(topics) >= max_topics:
            break
    return {make_entity_id("topic", t) for t in topics}


def extract_entity_keys_from_source(source: EvidenceObject) -> set[str]:
    if source.entity_ids:
        return set(source.entity_ids)
    return set(resolve_entities_for_source(source))


def extract_entity_keys_from_bundle(bundle: EvidenceBundle) -> tuple[str, ...]:
    return collect_bundle_entity_ids(bundle)
