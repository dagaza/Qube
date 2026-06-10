"""Automated smoke proxies for manual QA Sections 1, 6, and E2E-2.

Full E2E-1 (learn → recall → edit → recall) requires a running GUI session;
these tests validate the underlying settings defaults, export path, negative
list, and recall-intent plumbing that manual testers rely on.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest


def test_s1_promotion_default_off():
    """S1.3 — promotion is opt-in."""
    from core.app_settings import get_enable_memory_promotion

    with mock.patch("core.app_settings._store") as store:
        store.return_value.get.side_effect = lambda key, default=None: default
        assert get_enable_memory_promotion() is False


def test_s1_consolidation_default_off():
    """S1.6 — consolidation staging is opt-in."""
    from core.app_settings import get_enable_memory_consolidation

    with mock.patch("core.app_settings._store") as store:
        store.return_value.get.side_effect = lambda key, default=None: default
        assert get_enable_memory_consolidation() is False


def test_s1_enrichment_default_off():
    from core.app_settings import get_enable_memory_enrichment

    with mock.patch("core.app_settings._store") as store:
        store.return_value.get.side_effect = lambda key, default=None: default
        assert get_enable_memory_enrichment() is False


def test_s1_promotion_preset_default_standard():
    """S1.5 — preset defaults to standard."""
    from core.app_settings import get_memory_promotion_preset

    with mock.patch("core.app_settings._store") as store:
        store.return_value.get.side_effect = lambda key, default=None: default
        assert get_memory_promotion_preset() == "standard"


def test_m6_export_visible_writes_markdown(tmp_path: Path):
    """M6.5 — export produces dated markdown under ~/.qube/exports/."""
    from core.memory_export import default_export_path, export_memories_to_markdown, write_memory_export

    rows = [
        {
            "id": "row-1",
            "source": "qube_memory::preference::preference",
            "payload": {
                "content": "QA export smoke test content",
                "category": "preference",
                "provenance_quote": "I prefer QA exports.",
            },
        }
    ]
    md = export_memories_to_markdown(rows, title="Smoke Export")
    assert "QA export smoke test content" in md
    assert "## Preference" in md or "## preference" in md.lower()

    with mock.patch("core.memory_export.os.path.expanduser", return_value=str(tmp_path)):
        path = write_memory_export(rows)
    assert path.endswith(".md")
    assert Path(path).is_file()
    assert "QA export smoke test content" in Path(path).read_text(encoding="utf-8")

    with mock.patch("core.memory_export.os.path.expanduser", return_value=str(tmp_path)):
        dated = default_export_path()
    assert dated.endswith(".md")
    assert "memory_" in Path(dated).name


def test_m6_negative_list_blocks_similar_reinsert(tmp_path: Path):
    """M6.3 / E2E-2 proxy — deleted vector blocks near-duplicate store."""
    from core.memory_negative_list import MemoryNegativeList

    neg = MemoryNegativeList(path=str(tmp_path / "negatives.json"))
    vector = [0.1, 0.2, 0.3, 0.4]
    neg.add("User prefers teal for QA", vector)
    assert neg.is_negative(vector, threshold=0.20) is True
    assert neg.is_negative([0.9, 0.0, 0.0, 0.0], threshold=0.20) is False


def test_e2e1_recall_intent_detected_for_stored_fact_query():
    """E2E-1 partial — recall phrasing is recognized for later retrieval turn."""
    from core.memory_filters import detect_recall_intent

    assert detect_recall_intent("Remind me about metric units") is True
    assert detect_recall_intent("Suggest a pasta recipe") is False


def test_e2e1_explicit_remember_detection():
    """Supports E2E learn path — explicit remember gate."""
    from core.memory_filters import detect_explicit_remember

    body = detect_explicit_remember("Please remember that my QA codename is Nightjar-7.")
    assert body is not None
    assert "Nightjar" in body
