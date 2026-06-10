"""Tests for native engine preemption helpers and enrichment reschedule."""
from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

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


def _load_enrichment_worker_module():
    if "workers" not in sys.modules or not hasattr(sys.modules["workers"], "__path__"):
        stub_pkg = types.ModuleType("workers")
        stub_pkg.__path__ = [os.path.join(_WS_ROOT, "workers")]
        sys.modules["workers"] = stub_pkg
    mod_path = os.path.join(_WS_ROOT, "workers", "enrichment_worker.py")
    spec = importlib.util.spec_from_file_location("workers.enrichment_worker", mod_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["workers.enrichment_worker"] = module
    spec.loader.exec_module(module)
    return module


_ew_mod = _load_enrichment_worker_module()
EnrichmentWorker = _ew_mod.EnrichmentWorker
MAX_EXTRACTION_RESCHEDULE_ATTEMPTS = _ew_mod.MAX_EXTRACTION_RESCHEDULE_ATTEMPTS
EXTRACTION_RESCHEDULE_BACKOFF_SEC = _ew_mod.EXTRACTION_RESCHEDULE_BACKOFF_SEC

from core.native_engine_queue import EnginePriority, PriorityCommandQueue


def _should_cancel_background_job(cmd: dict, *, generation_epoch: int, cancel_flag: bool) -> bool:
    """Mirror NativeLlamaEngine._should_cancel_background_job for unit tests."""
    if cmd.get("op") != "chat_once":
        return bool(cancel_flag)
    epoch = int(cmd.get("epoch") or 0)
    return bool(cancel_flag) or epoch < generation_epoch


class PreemptionPolicyTests(unittest.TestCase):
    def test_chat_enqueue_ordering_over_background(self) -> None:
        q = PriorityCommandQueue()
        q.put({"op": "chat_once", "debug_caller": "memory_extraction"}, priority=EnginePriority.background)
        q.put({"op": "generate", "debug_caller": "chat"}, priority=EnginePriority.interactive)
        self.assertEqual(q.get().get("op"), "generate")

    def test_purge_background_before_chat_enqueue_pattern(self) -> None:
        q = PriorityCommandQueue()
        q.put({"op": "chat_once"}, priority=EnginePriority.background)
        q.put({"op": "profile_behavior"}, priority=EnginePriority.maintenance)
        q.put({"op": "generate"}, priority=EnginePriority.interactive)
        removed = q.purge(lambda c: c.get("op") in ("chat_once", "profile_behavior"))
        self.assertEqual(removed, 2)
        self.assertEqual(q.get().get("op"), "generate")

    def test_should_cancel_background_when_epoch_stale(self) -> None:
        cmd = {"op": "chat_once", "epoch": 2}
        self.assertTrue(_should_cancel_background_job(cmd, generation_epoch=3, cancel_flag=False))

    def test_should_not_cancel_chat_on_epoch_mismatch(self) -> None:
        cmd = {"op": "generate", "epoch": 2}
        self.assertFalse(_should_cancel_background_job(cmd, generation_epoch=3, cancel_flag=False))


class EnrichmentRescheduleTests(unittest.TestCase):
    def test_reschedule_turn_requeues_with_incremented_attempt(self) -> None:
        worker = EnrichmentWorker(MagicMock(), MagicMock(), MagicMock())
        payload = {"session_id": "s1", "reschedule_attempt": 0}
        with patch.object(_ew_mod.time, "sleep") as sleep:
            self.assertTrue(worker._reschedule_turn(payload))
            sleep.assert_called_once_with(EXTRACTION_RESCHEDULE_BACKOFF_SEC)
        item = worker.queue.get_nowait()
        self.assertEqual(item["reschedule_attempt"], 1)

    def test_reschedule_cap_returns_false(self) -> None:
        worker = EnrichmentWorker(MagicMock(), MagicMock(), MagicMock())
        payload = {
            "session_id": "s1",
            "reschedule_attempt": MAX_EXTRACTION_RESCHEDULE_ATTEMPTS,
        }
        self.assertFalse(worker._reschedule_turn(payload))
        self.assertTrue(worker.queue.empty())


if __name__ == "__main__":
    unittest.main()
