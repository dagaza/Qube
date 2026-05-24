"""One-time migration for pre-T3.4 memory ``source`` strings.

Pre-T3.4 rows used ``qube_memory::<category>`` (two segments). T3.4+
rows use ``qube_memory::<tier>::<category>``. The old retrieval
compatibility wildcard ``LIKE 'qube_memory::%'`` matched every tier and
defeated chat-turn isolation — so legacy rows are migrated to an
explicit namespace:

    qube_memory::legacy::<category>

After migration, retrieval can scope ``include_context`` to
``qube_memory::context::%`` + ``qube_memory::legacy::%`` without pulling
knowledge or episode rows on plain chat turns.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("Qube.MemorySourceMigration")

MEMORY_SOURCE_PREFIX = "qube_memory::"
LEGACY_TIER = "legacy"
KNOWN_TIERS = frozenset({"preference", "knowledge", "episode", "context", LEGACY_TIER})


def is_unnamespaced_legacy_source(source: str) -> bool:
    """Return True for pre-T3.4 ``qube_memory::<category>`` sources."""
    if not isinstance(source, str) or not source.startswith(MEMORY_SOURCE_PREFIX):
        return False
    parts = source.split("::")
    # Exactly ``qube_memory`` + one category segment — no tier namespace.
    return len(parts) == 2 and bool(parts[1].strip())


def legacy_namespaced_source(category: str) -> str:
    """Map a bare category to the explicit legacy tier namespace."""
    cat = (category or "context").strip().lower() or "context"
    return f"{MEMORY_SOURCE_PREFIX}{LEGACY_TIER}::{cat}"


def migrate_legacy_memory_sources(store, *, batch_limit: int = 100_000) -> int:
    """Rewrite unnamespaced legacy memory rows in-place.

    Idempotent: rows already under ``qube_memory::legacy::`` or any other
    three-segment tier namespace are left unchanged.

    Returns the number of rows migrated.
    """
    table = getattr(getattr(store, "table", None), "delete", None)
    if table is None:
        return 0

    try:
        rows = (
            store.table.search()
            .where(f"source LIKE '{MEMORY_SOURCE_PREFIX}%'")
            .limit(batch_limit)
            .to_list()
        )
    except Exception as e:
        logger.warning("[MemorySourceMigration] scan failed: %s", e)
        return 0

    migrated = 0
    for row in rows:
        source = str(row.get("source") or "")
        if not is_unnamespaced_legacy_source(source):
            continue

        category = source.split("::", 1)[1].strip().lower() or "context"
        new_source = legacy_namespaced_source(category)
        rid = row.get("id")
        if not rid:
            continue

        try:
            safe_id = str(rid).replace("'", "''")
            store.table.delete(f"id = '{safe_id}'")
            store.table.add([{
                "text": row.get("text"),
                "vector": row.get("vector"),
                "source": new_source,
                "chunk_id": int(row.get("chunk_id") or 0),
            }])
            migrated += 1
        except Exception as e:
            logger.warning(
                "[MemorySourceMigration] failed to migrate row %s: %s",
                rid,
                e,
            )

    if migrated:
        logger.info(
            "[MemorySourceMigration] migrated %d legacy memory row(s) to %s*",
            migrated,
            f"{MEMORY_SOURCE_PREFIX}{LEGACY_TIER}::",
        )
    return migrated


__all__ = [
    "KNOWN_TIERS",
    "LEGACY_TIER",
    "MEMORY_SOURCE_PREFIX",
    "is_unnamespaced_legacy_source",
    "legacy_namespaced_source",
    "migrate_legacy_memory_sources",
]
