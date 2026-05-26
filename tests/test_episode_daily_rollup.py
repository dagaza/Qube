"""Daily episode rollup (v7) — YYYYMMDD source idempotency."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
import unittest
from unittest import mock

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)


def _load_enrichment_worker_module():
    if "PyQt6" not in sys.modules:
        pyqt_mod = types.ModuleType("PyQt6")
        qtcore_mod = types.ModuleType("PyQt6.QtCore")

        class _StubQThread:
            def __init__(self, *a, **kw):
                pass

            def start(self):
                pass

            def msleep(self, _ms):
                pass

        class _StubMutex:
            def lock(self):
                pass

            def unlock(self):
                pass

        class _StubMutexLocker:
            def __init__(self, *_a, **_kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_a):
                return False

        def _stub_pyqt_signal(*_a, **_kw):
            class _Signal:
                def connect(self, *_a, **_kw):
                    pass

                def emit(self, *_a, **_kw):
                    pass

            return _Signal()

        qtcore_mod.QThread = _StubQThread
        qtcore_mod.QMutex = _StubMutex
        qtcore_mod.QMutexLocker = _StubMutexLocker
        qtcore_mod.pyqtSignal = _stub_pyqt_signal
        pyqt_mod.QtCore = qtcore_mod
        sys.modules["PyQt6"] = pyqt_mod
        sys.modules["PyQt6.QtCore"] = qtcore_mod

    mod_path = os.path.join(_WS_ROOT, "workers", "enrichment_worker.py")
    spec = importlib.util.spec_from_file_location("workers.enrichment_worker", mod_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["workers.enrichment_worker"] = module
    spec.loader.exec_module(module)
    return module


_ew = _load_enrichment_worker_module()
EnrichmentWorker = _ew.EnrichmentWorker


class _EpisodeTable:
    def __init__(self, store):
        self.store = store
        self._where = ""

    def search(self, *_a, **_kw):
        return self

    def where(self, clause):
        self._where = clause or ""
        return self

    def limit(self, *_a, **_kw):
        return self

    def to_list(self):
        if "episode" in self._where:
            return list(self.store.rows)
        return []

    def delete(self, where):
        self.store.deletes.append(where)

    def add(self, records):
        for r in records:
            self.store.rows.append(dict(r))


class _EpisodeStore:
    def __init__(self, rows=None):
        self.rows = list(rows or [])
        self.deletes: list[str] = []

    @property
    def table(self):
        return _EpisodeTable(self)


class _FakeEmbedder:
    def embed_query(self, _text):
        return [0.1, 0.2, 0.3]


class _FakeLLM:
    def isRunning(self):
        return False


class TestDailyEpisodeRollup(unittest.TestCase):
    def test_daily_rollup_writes_yyyymmdd_source_and_replaces(self):
        day_key = "20260526"
        session_row = {
            "source": "qube_memory::episode::sess-abc",
            "text": json.dumps(
                {
                    "category": "episode",
                    "content": "Worked on memory pipeline.",
                    "topics": ["memory"],
                }
            ),
        }
        store = _EpisodeStore([session_row])
        worker = EnrichmentWorker(
            llm=_FakeLLM(),
            embedder=_FakeEmbedder(),
            store=store,
            db=None,
        )
        worker._last_daily_rollup_ts = 0.0

        with mock.patch("workers.enrichment_worker.time.strftime", return_value=day_key):
            worker._maybe_daily_episode_rollup()
            worker._maybe_daily_episode_rollup()

        daily = [r for r in store.rows if r.get("source") == f"qube_memory::episode::{day_key}"]
        self.assertEqual(len(daily), 1)
        payload = json.loads(daily[0]["text"])
        self.assertEqual(payload.get("origin"), "daily_rollup")
        self.assertIn("memory pipeline", payload.get("content", "").lower())


if __name__ == "__main__":
    unittest.main()
