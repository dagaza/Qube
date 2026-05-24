"""Master enrichment toggle: enqueue gate, queue drain, maintenance while disabled."""
from __future__ import annotations

import importlib.util
import os
import sys
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

        def sleep(self, _sec):
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


def _load_module(relative_path: str, module_name: str):
    if "workers" not in sys.modules or not hasattr(sys.modules["workers"], "__path__"):
        stub_pkg = types.ModuleType("workers")
        stub_pkg.__path__ = [os.path.join(_WS_ROOT, "workers")]
        sys.modules["workers"] = stub_pkg

    mod_path = os.path.join(_WS_ROOT, relative_path)
    spec = importlib.util.spec_from_file_location(module_name, mod_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_ew_module = _load_module("workers/enrichment_worker.py", "workers.enrichment_worker")
EnrichmentWorker = _ew_module.EnrichmentWorker

_rw_module = _load_module("workers/memory_reflection_worker.py", "workers.memory_reflection_worker")
MemoryReflectionWorker = _rw_module.MemoryReflectionWorker


class _FakeLLM:
    def isRunning(self):
        return False


class _FakeDB:
    def get_session_history(self, _session_id):
        return []


class _FakeEmbedder:
    def embed_query(self, _content):
        return [0.0, 0.0, 0.0]


class _FakeStore:
    @property
    def table(self):
        return self

    def add(self, *_a, **_kw):
        pass

    def delete(self, *_a, **_kw):
        pass

    def search(self, *_a, **_kw):
        return self

    def where(self, *_a, **_kw):
        return self

    def limit(self, *_a, **_kw):
        return self

    def to_list(self):
        return []


def _make_enrichment_worker():
    return EnrichmentWorker(
        llm=_FakeLLM(),
        embedder=_FakeEmbedder(),
        store=_FakeStore(),
        db=_FakeDB(),
    )


class TestEnrichmentEnabledGate(unittest.TestCase):
    def test_enqueue_ignored_when_disabled(self):
        worker = _make_enrichment_worker()
        worker.set_enabled(False)
        worker.enqueue({"session_id": "sess-1"})
        self.assertTrue(worker.queue.empty())

    def test_set_enabled_false_drains_queue(self):
        worker = _make_enrichment_worker()
        worker.enqueue({"session_id": "sess-1"})
        worker.enqueue({"session_id": "sess-2"})
        worker.set_enabled(False)
        self.assertTrue(worker.queue.empty())

    def test_disabled_idle_tick_runs_maintenance(self):
        worker = _make_enrichment_worker()
        worker.set_enabled(False)

        def _stop_after_sleep(_ms):
            worker.is_running = False

        worker.msleep = _stop_after_sleep
        with mock.patch.object(worker, "_maybe_drain_usage_recorder") as drain, mock.patch.object(
            worker, "_maybe_run_decay_sweep"
        ) as decay:
            worker.run()
            drain.assert_called()
            decay.assert_called()

    def test_enabled_processing_still_runs_maintenance_in_finally(self):
        worker = _make_enrichment_worker()
        worker.enqueue({"session_id": "sess-1", "skip_enrichment": True})

        def _stop_after_get(*_a, **_kw):
            worker.is_running = False
            return {"session_id": "sess-1", "skip_enrichment": True}

        with mock.patch.object(worker.queue, "get", side_effect=_stop_after_get), mock.patch.object(
            worker, "_maybe_drain_usage_recorder"
        ) as drain, mock.patch.object(worker, "_maybe_run_decay_sweep") as decay:
            worker.run()
            self.assertGreaterEqual(drain.call_count, 1)
            self.assertGreaterEqual(decay.call_count, 1)


class TestMemoryReflectionEnabledGate(unittest.TestCase):
    def test_disabled_skips_run_cycle_when_due(self):
        worker = MemoryReflectionWorker(llm=_FakeLLM(), store=_FakeStore())
        worker._next_run_at = 0.0
        worker.set_enabled(False)

        def _stop_after_sleep(_sec):
            worker._running = False

        worker.sleep = _stop_after_sleep
        with mock.patch.object(worker, "_run_cycle") as cycle:
            worker.run()
            cycle.assert_not_called()

    def test_enabled_runs_cycle_when_due(self):
        worker = MemoryReflectionWorker(llm=_FakeLLM(), store=_FakeStore())
        worker._next_run_at = 0.0
        worker.set_enabled(True)

        def _stop_after_sleep(_sec):
            worker._running = False

        worker.sleep = _stop_after_sleep
        with mock.patch.object(worker, "_run_cycle") as cycle:
            worker.run()
            cycle.assert_called_once()

    def test_disable_pushes_next_run_forward(self):
        worker = MemoryReflectionWorker(llm=_FakeLLM(), store=_FakeStore())
        worker._next_run_at = 0.0
        before = worker._next_run_at
        worker.set_enabled(False)
        self.assertGreater(worker._next_run_at, before)


if __name__ == "__main__":
    unittest.main()
