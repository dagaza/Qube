"""Memory v7 pre-window salvage path."""
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock


def _load_enrichment_worker():
    path = Path(__file__).resolve().parents[1] / "workers" / "enrichment_worker.py"
    spec = importlib.util.spec_from_file_location("test_enrichment_worker", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_salvage_rate_limit(monkeypatch):
    ew_mod = _load_enrichment_worker()
    monkeypatch.setattr(ew_mod, "get_enable_memory_v7_salvage", lambda: True)
    worker = ew_mod.EnrichmentWorker(
        llm=MagicMock(),
        embedder=MagicMock(),
        store=MagicMock(),
        db=MagicMock(),
    )
    worker._wait_for_chat_llm_idle = lambda: True
    worker._salvage_last_ts["sess"] = ew_mod.time.time()
    worker._process_salvage(
        {"session_id": "sess", "salvage_message_ids": ["m1"], "enrichment_mode": "salvage"}
    )
    worker.db.get_session_history.assert_not_called()


def test_salvage_extracts_when_messages_present(monkeypatch):
    ew_mod = _load_enrichment_worker()
    monkeypatch.setattr(ew_mod, "get_enable_memory_v7_salvage", lambda: True)
    worker = ew_mod.EnrichmentWorker(
        llm=MagicMock(),
        embedder=MagicMock(),
        store=MagicMock(),
        db=MagicMock(),
    )
    worker._wait_for_chat_llm_idle = lambda: True
    worker._salvage_last_ts.clear()
    worker.db.get_session_history.return_value = [
        {"id": "m1", "role": "user", "content": "Remember I prefer tea."},
    ]
    worker._generate_memory = lambda _p: "[]"
    worker._extract_json_facts = lambda _r: []
    worker._store_facts = MagicMock()
    worker._process_salvage(
        {"session_id": "sess", "salvage_message_ids": ["m1"], "enrichment_mode": "salvage"}
    )
    worker.db.get_session_history.assert_called_once()
