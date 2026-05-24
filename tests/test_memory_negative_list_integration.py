"""Integration tests: delete → negative list → enrichment reject."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

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


class _FakeTable:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.add_calls: list[list[dict]] = []

    def search(self, _vec=None):
        return self

    def where(self, _clause: str):
        return self

    def limit(self, _n: int):
        return self

    def to_list(self):
        return []

    def add(self, records: list[dict]):
        self.add_calls.append(records)


class _FakeStore:
    def __init__(self) -> None:
        self.table = _FakeTable()


class _FakeEmbedder:
    def __init__(self, vector):
        self._vector = vector

    def embed_query(self, _text: str):
        return self._vector


class NegativeListEnrichmentIntegrationTests(unittest.TestCase):
    def test_store_facts_rejects_negative_list_vector(self):
        import numpy as np
        from core.memory_negative_list import MemoryNegativeList

        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            neg = MemoryNegativeList(path=os.path.join(tmp, "memory_negatives.json"))
            neg.add("deleted memory about tea", vec)

            worker = EnrichmentWorker(
                llm=mock.Mock(),
                embedder=_FakeEmbedder(vec),
                store=_FakeStore(),
                db=mock.Mock(),
            )

            fact = {
                "subject": "user",
                "source_role": "user",
                "durability": "long_term",
                "category": "preference",
                "content": "User prefers green tea",
                "provenance_quote": "I prefer green tea",
                "confidence": 0.9,
            }

            with mock.patch.object(
                _ew_module,
                "get_memory_negative_list",
                return_value=neg,
            ):
                worker._store_facts(
                    [fact],
                    turn_context={
                        "session_id": "sess-1",
                        "source_message_ids": [],
                        "rag_chunk_ids": [],
                        "conversation_text": "I prefer green tea",
                    },
                )

            self.assertEqual(worker.store.table.add_calls, [])


class MemoryManagerDeleteNegativeListTests(unittest.TestCase):
    def test_delete_adds_content_and_vector_to_negative_list(self):
        import numpy as np
        from core.memory_negative_list import MemoryNegativeList

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "memory_negatives.json")
            neg = MemoryNegativeList(path=path)
            vec = np.array([0.5, 0.5, 0.0], dtype=np.float32)
            neg.add("my favorite color is blue", vec)

            self.assertTrue(neg.is_negative(vec, threshold=0.20))
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertEqual(len(data["entries"]), 1)
            self.assertEqual(data["entries"][0]["content"], "my favorite color is blue")


if __name__ == "__main__":
    unittest.main()
