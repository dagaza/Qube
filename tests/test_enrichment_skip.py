"""T3.3 — tool-aware extraction fences.

Covers the ``skip_enrichment`` / ``enrichment_mode`` plumbing added to
``EnrichmentWorker`` so that pipeline errors, stream-repetition trips,
web-tool failures, and assistant-failure final text do not result in
spurious memories being mined from broken turns.

These tests exercise the public extraction entry points directly
without starting the Qt event loop: ``_process_turn`` for the happy
path + skip path, and ``_extract_and_store(mode=...)`` for the
explicit-remember interaction.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from unittest import mock

# Stub the PyQt6 modules the worker imports at module load time so the
# test suite does not have to depend on the Qt runtime.
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
    """Load ``workers/enrichment_worker.py`` directly so we bypass
    ``workers/__init__.py`` (which drags in pyaudio / openwakeword /
    faster_whisper / etc. we don't need for this unit test)."""
    # Pre-seed a stub ``workers`` package so relative resolution works if
    # the module ever references sibling files. We do NOT run the real
    # package init.
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
    def __init__(self):
        self.generate_calls = 0
        self._running = False

    def isRunning(self):
        return self._running

    def generate(self, _prompt):
        self.generate_calls += 1
        return "[]"


class _FakeDB:
    def __init__(self, messages=None):
        self._messages = messages or []

    def get_session_history(self, _session_id):
        return list(self._messages)


class _FakeEmbedder:
    def embed_query(self, _content):
        return [0.0, 0.0, 0.0]


class _FakeStore:
    def __init__(self):
        self.added: list = []

    class _Table:
        def __init__(self, parent):
            self.parent = parent

        def add(self, records):
            self.parent.added.extend(records)

        def delete(self, _where):
            pass

        def search(self, *_a, **_kw):
            return self

        def where(self, *_a, **_kw):
            return self

        def limit(self, *_a, **_kw):
            return self

        def to_list(self):
            return []

    @property
    def table(self):
        return self._Table(self)


def _make_worker(messages=None):
    llm = _FakeLLM()
    db = _FakeDB(messages=messages)
    worker = EnrichmentWorker(
        llm=llm,
        embedder=_FakeEmbedder(),
        store=_FakeStore(),
        db=db,
    )
    return worker, llm


class TestEnrichmentSkip(unittest.TestCase):
    def test_skip_payload_short_circuits_extraction(self):
        worker, llm = _make_worker(
            messages=[{"id": "u1", "role": "user", "content": "hello"}]
        )
        payload = {
            "session_id": "sess-1",
            "last_user_msg_id": "u1",
            "last_assistant_msg_id": "a1",
            "rag_chunk_ids": [],
            "skip_enrichment": True,
            "enrichment_mode": "skip",
            "skip_reason": "pipeline_error",
        }
        with mock.patch.object(worker, "_extract_and_store") as m_extract:
            # Emulate the relevant branch of run(): _process_turn sees
            # enrichment_mode=="skip" and returns without doing work.
            worker._process_turn(payload)
            m_extract.assert_not_called()
        self.assertEqual(llm.generate_calls, 0)

    def test_full_mode_payload_still_runs_extractor(self):
        worker, llm = _make_worker(
            messages=[
                {"id": "u1", "role": "user", "content": "I love dark roast."},
                {"id": "a1", "role": "assistant", "content": "Noted."},
            ]
        )
        payload = {
            "session_id": "sess-2",
            "last_user_msg_id": "u1",
            "last_assistant_msg_id": "a1",
            "rag_chunk_ids": [],
        }
        with mock.patch.object(worker, "_extract_and_store") as m_extract:
            worker._process_turn(payload)
            m_extract.assert_called_once()
            _, kwargs = m_extract.call_args
            self.assertEqual(kwargs.get("mode"), "full")

    def test_explicit_only_mode_skips_extractor_but_runs_bypass(self):
        messages = [
            {
                "id": "u1",
                "role": "user",
                "content": "please remember that my mom's name is Cornelia",
            },
            {
                "id": "a1",
                "role": "assistant",
                "content": "Got it, I've made a note.",
            },
        ]
        worker, llm = _make_worker(messages=messages)

        captured: list[list[dict]] = []

        def fake_store_facts(facts, turn_context=None):
            captured.append(list(facts))

        with mock.patch.object(worker, "_store_facts", side_effect=fake_store_facts):
            worker._extract_and_store(
                session_id="sess-3",
                messages=messages,
                last_user_msg_id="u1",
                last_assistant_msg_id="a1",
                rag_chunk_ids=[],
                mode="explicit_only",
            )

        self.assertEqual(
            llm.generate_calls,
            0,
            "explicit_only mode must not call the extractor LLM",
        )
        self.assertEqual(len(captured), 1, "exactly one store-facts invocation")
        stored = captured[0]
        self.assertEqual(len(stored), 1, "only the bypass fact should be seeded")
        self.assertTrue(stored[0].get("_explicit_remember"))
        self.assertIn("cornelia", stored[0]["content"].lower())

    def test_explicit_only_mode_no_bypass_means_no_write(self):
        messages = [
            {"id": "u1", "role": "user", "content": "hello there"},
            {"id": "a1", "role": "assistant", "content": "hi!"},
        ]
        worker, llm = _make_worker(messages=messages)

        with mock.patch.object(worker, "_store_facts") as m_store:
            worker._extract_and_store(
                session_id="sess-4",
                messages=messages,
                last_user_msg_id="u1",
                last_assistant_msg_id="a1",
                rag_chunk_ids=[],
                mode="explicit_only",
            )
            m_store.assert_not_called()
        self.assertEqual(llm.generate_calls, 0)

    def test_full_mode_still_invokes_llm_generate(self):
        messages = [
            {"id": "u1", "role": "user", "content": "I am a PhD student."},
            {"id": "a1", "role": "assistant", "content": "Interesting!"},
        ]
        worker, llm = _make_worker(messages=messages)

        with mock.patch.object(worker, "_store_facts"):
            worker._extract_and_store(
                session_id="sess-5",
                messages=messages,
                last_user_msg_id="u1",
                last_assistant_msg_id="a1",
                rag_chunk_ids=[],
                mode="full",
            )
        self.assertEqual(llm.generate_calls, 1)

    def test_unknown_mode_coerced_to_full(self):
        messages = [
            {"id": "u1", "role": "user", "content": "hello"},
            {"id": "a1", "role": "assistant", "content": "hi"},
        ]
        worker, llm = _make_worker(messages=messages)
        with mock.patch.object(worker, "_store_facts"):
            worker._extract_and_store(
                session_id="sess-6",
                messages=messages,
                last_user_msg_id="u1",
                last_assistant_msg_id="a1",
                rag_chunk_ids=[],
                mode="nonsense",
            )
        self.assertEqual(llm.generate_calls, 1)


if __name__ == "__main__":
    unittest.main()
