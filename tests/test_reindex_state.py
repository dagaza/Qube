"""Reindex state flag tests."""
from __future__ import annotations

from core.reindex_state import is_reindex_in_progress, set_reindex_in_progress


def test_reindex_flag_roundtrip():
    set_reindex_in_progress(False)
    assert is_reindex_in_progress() is False
    set_reindex_in_progress(True)
    assert is_reindex_in_progress() is True
    set_reindex_in_progress(False)
    assert is_reindex_in_progress() is False
