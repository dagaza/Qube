"""Tests for local GGUF menu listing."""

from __future__ import annotations


def test_list_local_gguf_menu_entries_empty_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("core.local_gguf_library.get_llm_models_dir", lambda: str(tmp_path))
    from core.local_gguf_library import list_local_gguf_menu_entries

    assert list_local_gguf_menu_entries() == []


def test_list_local_gguf_menu_entries_returns_primary_models(tmp_path, monkeypatch):
    models_dir = tmp_path / "llm"
    models_dir.mkdir()
    (models_dir / "demo-model-q4_k_m.gguf").write_bytes(b"gguf")
    monkeypatch.setattr("core.local_gguf_library.get_llm_models_dir", lambda: str(models_dir))

    from core.local_gguf_library import list_local_gguf_menu_entries

    entries = list_local_gguf_menu_entries()
    assert len(entries) == 1
    _label, path = entries[0]
    assert path == str((models_dir / "demo-model-q4_k_m.gguf").resolve())
