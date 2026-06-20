"""Global reindex-in-progress flag for retrieval and write gates."""
from __future__ import annotations

_reindex_in_progress: bool = False


def is_reindex_in_progress() -> bool:
    return _reindex_in_progress


def set_reindex_in_progress(value: bool) -> None:
    global _reindex_in_progress
    _reindex_in_progress = bool(value)
