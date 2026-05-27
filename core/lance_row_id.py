"""Stable row identity helpers for LanceDB tables without a user ``id`` column."""
from __future__ import annotations

# Request ``_rowid`` whenever callers need a durable key for delete/update.
LANCE_ROW_ID_SELECT = ["_rowid", "vector", "text", "source", "chunk_id"]


def lance_row_id(row: dict | None) -> str | None:
    """Return a string row key from a LanceDB result dict."""
    if not row:
        return None
    rowid = row.get("_rowid")
    if rowid is not None:
        return str(rowid)
    legacy = row.get("id")
    if legacy is not None and str(legacy).strip():
        return str(legacy)
    return None


def lance_row_delete_filter(row_id: str | None) -> str | None:
    """Return a LanceDB delete/where filter for ``row_id``, or None if invalid."""
    if row_id is None:
        return None
    rid = str(row_id).strip()
    if not rid:
        return None
    if rid.isdigit():
        return f"_rowid = {int(rid)}"
    safe = rid.replace("'", "''")
    return f"id = '{safe}'"


__all__ = ["LANCE_ROW_ID_SELECT", "lance_row_delete_filter", "lance_row_id"]
