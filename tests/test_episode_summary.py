"""T3.2 — EnrichmentWorker episode summariser (``_maybe_summarise_session``).

Covers:
- cadence gate: counter must reach ``EPISODE_SUMMARY_TURN_CADENCE`` before
  a summary is written
- idle gate: a big gap since the previous turn also fires the summary
- SKIP response from the LLM means no episode row is written
- valid SUMMARY/TOPICS response writes one row to
  ``qube_memory::episode::<session_id>`` with ``category="episode"``
- assistant-failure summary text is rejected
"""
from __future__ import annotations

import importlib.util
import json
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
EPISODE_SUMMARY_TURN_CADENCE = _ew_module.EPISODE_SUMMARY_TURN_CADENCE


class _FakeLLM:
    def __init__(self, response: str = ""):
        self.response = response
        self.calls: list[str] = []

    def isRunning(self):
        return False

    def generate(self, prompt):
        self.calls.append(prompt)
        return self.response


class _FakeDB:
    def __init__(self, messages=None):
        self._messages = messages or []

    def get_session_history(self, _session_id):
        return list(self._messages)


class _FakeEmbedder:
    def embed_query(self, _content):
        return [0.1, 0.2, 0.3]


class _FakeTable:
    def __init__(self, parent):
        self.parent = parent

    def add(self, records):
        for r in records:
            self.parent.rows.append(dict(r))

    def delete(self, where):
        self.parent.deletes.append(where)
        # Crude filter for tests: support `source = '...'` predicate.
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
        return _FakeTable(self)


def _build_turn_messages(n_turns: int) -> list[dict]:
    msgs: list[dict] = []
    for i in range(n_turns):
        msgs.append({"id": f"u{i}", "role": "user", "content": f"user turn {i}"})
        msgs.append({"id": f"a{i}", "role": "assistant", "content": f"assistant turn {i}"})
    return msgs


def _make_worker(messages=None, llm_response: str = ""):
    llm = _FakeLLM(response=llm_response)
    return (
        EnrichmentWorker(
            llm=llm,
            embedder=_FakeEmbedder(),
            store=_FakeStore(),
            db=_FakeDB(messages=messages or []),
        ),
        llm,
    )


class TestEpisodeSummariser(unittest.TestCase):
    def test_cadence_below_threshold_does_not_fire(self):
        msgs = _build_turn_messages(4)
        worker, llm = _make_worker(
            messages=msgs,
            llm_response="SUMMARY: The user worked on a thing.\nTOPICS: thing, other",
        )
        # Call once; the counter should advance to 1 which is well below
        # EPISODE_SUMMARY_TURN_CADENCE. No LLM call, no row.
        worker._maybe_summarise_session("sess-cadence-low")
        self.assertEqual(llm.calls, [])
        self.assertEqual(worker.store.rows, [])

    def test_cadence_threshold_fires_and_writes_row(self):
        msgs = _build_turn_messages(EPISODE_SUMMARY_TURN_CADENCE)
        worker, llm = _make_worker(
            messages=msgs,
            llm_response=(
                "SUMMARY: The user is refactoring the enrichment pipeline.\n"
                "TOPICS: memory, enrichment, t3.2"
            ),
        )
        for _ in range(EPISODE_SUMMARY_TURN_CADENCE):
            worker._maybe_summarise_session("sess-cadence")

        self.assertEqual(len(llm.calls), 1, "should fire exactly once at threshold")
        self.assertEqual(len(worker.store.rows), 1)
        row = worker.store.rows[0]
        self.assertTrue(row["source"].startswith("qube_memory::episode::"))
        self.assertIn("sess-cadence", row["source"])
        payload = json.loads(row["text"])
        self.assertEqual(payload["category"], "episode")
        self.assertEqual(payload["subject"], "user")
        self.assertEqual(payload["origin"], "session_summary")
        self.assertIn("refactoring", payload["content"].lower())
        self.assertIn("memory", payload["topics"])

    def test_skip_response_writes_nothing(self):
        msgs = _build_turn_messages(EPISODE_SUMMARY_TURN_CADENCE)
        worker, llm = _make_worker(
            messages=msgs,
            llm_response="SUMMARY: SKIP\nTOPICS:",
        )
        for _ in range(EPISODE_SUMMARY_TURN_CADENCE):
            worker._maybe_summarise_session("sess-skip")

        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(worker.store.rows, [])

    def test_empty_llm_response_writes_nothing(self):
        msgs = _build_turn_messages(EPISODE_SUMMARY_TURN_CADENCE)
        worker, _llm = _make_worker(messages=msgs, llm_response="")
        for _ in range(EPISODE_SUMMARY_TURN_CADENCE):
            worker._maybe_summarise_session("sess-empty")
        self.assertEqual(worker.store.rows, [])

    def test_assistant_failure_summary_rejected(self):
        msgs = _build_turn_messages(EPISODE_SUMMARY_TURN_CADENCE)
        worker, _llm = _make_worker(
            messages=msgs,
            llm_response=(
                "SUMMARY: I don't have access to the internet right now.\n"
                "TOPICS: offline"
            ),
        )
        for _ in range(EPISODE_SUMMARY_TURN_CADENCE):
            worker._maybe_summarise_session("sess-fail")
        self.assertEqual(
            worker.store.rows, [],
            "assistant-failure-style summaries must be rejected",
        )

    def test_idle_gate_fires_on_second_turn(self):
        msgs = _build_turn_messages(4)
        worker, llm = _make_worker(
            messages=msgs,
            llm_response=(
                "SUMMARY: The user came back to continue a previous task.\n"
                "TOPICS: resume, task"
            ),
        )
        # Force an idle gap by stamping an old ``last_turn_ts`` before the
        # second call. The helper requires counter >= 2 AND idle > window.
        worker._session_turns_since_summary["sess-idle"] = 1
        worker._session_last_turn_ts["sess-idle"] = 0.0  # epoch = effectively forever ago
        worker._maybe_summarise_session("sess-idle")
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(len(worker.store.rows), 1)


if __name__ == "__main__":
    unittest.main()
