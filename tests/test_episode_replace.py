"""T3.2 — episode replace-in-place per session.

The enrichment worker only keeps ONE episode row per session id. When
the summariser fires a second time on the same session, it must delete
the prior row and write the fresh summary. This test exercises that
contract at the ``_replace_episode_row`` level so we do not need the
cadence gate in play.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
import unittest

if "PyQt6" not in sys.modules or not hasattr(sys.modules.get("PyQt6.QtCore", object()), "pyqtSignal"):
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

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)


def _load_enrichment_worker_module():
    if "workers" not in sys.modules or not hasattr(sys.modules["workers"], "__path__"):
        stub_pkg = types.ModuleType("workers")
        stub_pkg.__path__ = [os.path.join(_WS_ROOT, "workers")]
        sys.modules["workers"] = stub_pkg

    mod_path = os.path.join(_WS_ROOT, "workers", "enrichment_worker.py")
    spec = importlib.util.spec_from_file_location(
        "workers.enrichment_worker", mod_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["workers.enrichment_worker"] = module
    spec.loader.exec_module(module)
    return module


_ew_module = _load_enrichment_worker_module()
EnrichmentWorker = _ew_module.EnrichmentWorker


class _FakeLLM:
    def isRunning(self):
        return False

    def generate(self, _prompt):
        return ""


class _FakeDB:
    def get_session_history(self, _session_id):
        return []


class _FakeEmbedder:
    def embed_query(self, _content):
        return [0.0, 0.0, 0.0]


class _Table:
    def __init__(self, parent):
        self.parent = parent

    def add(self, records):
        for r in records:
            self.parent.rows.append(dict(r))

    def delete(self, where):
        self.parent.deletes.append(where)
        if where.startswith("source = '") and where.endswith("'"):
            target = where[len("source = '") : -1]
            self.parent.rows = [r for r in self.parent.rows if r.get("source") != target]

    def search(self, *_a, **_kw):
        return self

    def where(self, *_a, **_kw):
        return self

    def limit(self, *_a, **_kw):
        return self

    def to_list(self):
        return []


class _FakeStore:
    def __init__(self):
        self.rows: list[dict] = []
        self.deletes: list[str] = []

    @property
    def table(self):
        return _Table(self)


def _make_worker():
    return EnrichmentWorker(
        llm=_FakeLLM(),
        embedder=_FakeEmbedder(),
        store=_FakeStore(),
        db=_FakeDB(),
    )


class TestEpisodeReplaceInPlace(unittest.TestCase):
    def test_second_summary_replaces_first_for_same_session(self):
        worker = _make_worker()
        # First summary
        worker._replace_episode_row(
            session_id="sess-AAA",
            summary="User started a refactor.",
            topics=["refactor", "memory"],
            source_message_ids=["u0", "a0"],
            vector=[0.1, 0.2, 0.3],
            reason="cadence",
            turn_count=8,
        )
        self.assertEqual(len(worker.store.rows), 1)
        first_payload = json.loads(worker.store.rows[0]["text"])
        self.assertEqual(first_payload["content"], "User started a refactor.")

        # Second summary on the same session — must replace in place.
        worker._replace_episode_row(
            session_id="sess-AAA",
            summary="User finished the refactor.",
            topics=["refactor", "done"],
            source_message_ids=["u1", "a1"],
            vector=[0.4, 0.5, 0.6],
            reason="idle",
            turn_count=12,
        )
        self.assertEqual(
            len(worker.store.rows), 1,
            "only one episode row should exist per session id",
        )
        new_payload = json.loads(worker.store.rows[0]["text"])
        self.assertEqual(new_payload["content"], "User finished the refactor.")
        self.assertEqual(new_payload["turn_count"], 12)
        self.assertEqual(new_payload["episode_reason"], "idle")
        self.assertTrue(worker.store.rows[0]["source"].endswith("sess-AAA"))

    def test_summary_on_different_session_does_not_touch_first(self):
        worker = _make_worker()
        worker._replace_episode_row(
            session_id="sess-AAA",
            summary="Session A summary.",
            topics=["a"],
            source_message_ids=[],
            vector=[0.0, 0.0, 0.0],
            reason="cadence",
            turn_count=8,
        )
        worker._replace_episode_row(
            session_id="sess-BBB",
            summary="Session B summary.",
            topics=["b"],
            source_message_ids=[],
            vector=[0.0, 0.0, 0.0],
            reason="cadence",
            turn_count=8,
        )
        self.assertEqual(len(worker.store.rows), 2)
        sources = sorted(r["source"] for r in worker.store.rows)
        self.assertEqual(
            sources,
            [
                "qube_memory::episode::sess-AAA",
                "qube_memory::episode::sess-BBB",
            ],
        )


if __name__ == "__main__":
    unittest.main()
