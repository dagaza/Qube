"""Stable entity id helpers."""

from __future__ import annotations

import re

ENTITY_PREFIX = "entity"


def normalize_token(raw: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", (raw or "").strip().lower())
    return re.sub(r"-+", "-", text).strip("-")


def make_entity_id(kind: str, key: str) -> str:
    return f"{ENTITY_PREFIX}:{normalize_token(kind)}:{normalize_token(key)}"


def entity_kind(entity_id: str) -> str:
    parts = str(entity_id or "").split(":")
    if len(parts) >= 3 and parts[0] == ENTITY_PREFIX:
        return parts[1]
    return ""


def is_dedupe_cluster_entity(entity_id: str) -> bool:
    from core.knowledge.entities.policy import is_dedupe_cluster_entity as _policy_is_dedupe

    return _policy_is_dedupe(entity_id)
