"""Plan D26 (V40): memory v7 salvage receives the messages history windowing drops.

Before D26, ``LLMWorker._bound_session_history`` rebuilt each message as
``{role, content}``, so the ids ``get_session_history`` returns were lost and
salvage never received a message. Both the message-count window and S17's token
fit (``fit_history_to_token_budget``) drop a prefix of what they get, so the
worker now keeps the ids beside the messages, by position, and sends every
dropped id. ``EnrichmentWorker._process_salvage`` takes the oldest ids the
session has not salvaged yet, 24 a run, so each message is salvaged once (the
owner's choice: with the oldest 24 every turn and the 5-minute rate limit, 24 of
110 dropped messages ever reached salvage and one was salvaged 12 times). The
prompt never carries message ids. Salvage stays off by default.
"""

from __future__ import annotations

import importlib.util
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import workers.llm_worker as lw  # noqa: E402
from core.retrieval_budget import HistoryWindow, fit_history_to_token_budget, history_token_budget  # noqa: E402

LLMWorker = lw.LLMWorker


def _load_enrichment_worker():
    path = ROOT / "workers" / "enrichment_worker.py"
    spec = importlib.util.spec_from_file_location("test_d26_enrichment_worker", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _session(n_msgs: int, seed: int = 7) -> list[dict]:
    """Alternating user / assistant messages m0, m1, … ending on a user turn."""
    rnd = random.Random(seed)
    out = []
    for i in range(n_msgs):
        role = "user" if i % 2 == 0 else "assistant"
        size = rnd.randint(80, 400) if role == "user" else rnd.randint(600, 2400)
        out.append({"id": f"m{i}", "role": role, "content": ("word " * size)[:size]})
    if out[-1]["role"] != "user":
        out.append({"id": f"m{n_msgs}", "role": "user", "content": "And what about that?"})
    return out


def _worker(limit: int = 10) -> SimpleNamespace:
    return SimpleNamespace(
        max_history_messages=limit,
        CHAT_HISTORY_SINGLE_MESSAGE_MAX_CHARS=LLMWorker.CHAT_HISTORY_SINGLE_MESSAGE_MAX_CHARS,
        _pending_salvage_message_ids=[],
    )


def _fit(windowed: list[dict], n_ctx: int, retrieval: bool) -> HistoryWindow:
    budget = history_token_budget(
        n_ctx=n_ctx,
        skills_chars=1200,
        reserve_retrieval=retrieval,
        cap_chars=LLMWorker.MAX_TOTAL_RETRIEVAL_CHARS,
    )
    return fit_history_to_token_budget(windowed, budget)


class _SalvageOn(unittest.TestCase):
    ENABLED = True

    def setUp(self):
        p = patch.object(lw, "get_enable_memory_v7_salvage", return_value=self.ENABLED)
        p.start()
        self.addCleanup(p.stop)


# ---------------------------------------------------------------------------
# The worker: ids through both windows
# ---------------------------------------------------------------------------


class TestMessageWindow(_SalvageOn):
    def test_the_window_drops_are_sent_oldest_first(self):
        history = _session(13)  # m0 … m12
        w = _worker(limit=10)
        windowed = LLMWorker._bound_session_history(w, history)
        # The last 10 start on an assistant turn (m3), which goes too.
        self.assertEqual(w._pending_salvage_message_ids, ["m0", "m1", "m2", "m3"])
        self.assertEqual(w._turn_history_ids, [f"m{i}" for i in range(4, 13)])
        self.assertEqual([m["content"] for m in windowed], [m["content"] for m in history[4:]])

    def test_the_prompt_carries_no_ids(self):
        windowed = LLMWorker._bound_session_history(_worker(), _session(13))
        self.assertTrue(all(set(m) == {"role", "content"} for m in windowed))

    def test_nothing_dropped_sends_nothing(self):
        w = _worker(limit=10)
        LLMWorker._bound_session_history(w, _session(7))
        self.assertEqual(w._pending_salvage_message_ids, [])
        self.assertEqual(len(w._turn_history_ids), 7)

    def test_messages_without_an_id_are_skipped(self):
        history = _session(13)
        history[1].pop("id")
        w = _worker(limit=10)
        LLMWorker._bound_session_history(w, history)
        self.assertEqual(w._pending_salvage_message_ids, ["m0", "m2", "m3"])

    def test_an_empty_history_clears_the_previous_turns_ids(self):
        w = _worker(limit=10)
        LLMWorker._bound_session_history(w, _session(13))
        self.assertEqual(LLMWorker._bound_session_history(w, []), [])
        self.assertEqual((w._pending_salvage_message_ids, w._turn_history_ids, w._turn_window_dropped_ids), ([], [], []))


class TestSalvageOff(_SalvageOn):
    ENABLED = False

    def test_off_sends_nothing(self):
        w = _worker(limit=10)
        windowed = LLMWorker._bound_session_history(w, _session(13))
        self.assertEqual(w._pending_salvage_message_ids, [])
        before = list(w._pending_salvage_message_ids)
        LLMWorker._note_token_fit_salvage(w, HistoryWindow(windowed[2:], 2, 0, 100, "trimmed"), len(windowed))
        self.assertEqual(w._pending_salvage_message_ids, before)


class TestTokenFit(_SalvageOn):
    def test_the_token_fit_drops_follow_the_window_drops(self):
        history = _session(13)
        w = _worker(limit=10)
        windowed = LLMWorker._bound_session_history(w, history)
        window = _fit(windowed, n_ctx=2048, retrieval=True)
        self.assertGreater(window.dropped, 0)
        LLMWorker._note_token_fit_salvage(w, window, len(windowed))
        kept_contents = {m["content"] for m in window.messages}
        not_in_prompt = [m["id"] for m in history if m["content"] not in kept_contents]
        self.assertEqual(w._pending_salvage_message_ids, not_in_prompt)
        self.assertEqual(len(not_in_prompt), 4 + window.dropped)

    def test_called_again_it_does_not_repeat_ids(self):
        history = _session(13)
        w = _worker(limit=10)
        windowed = LLMWorker._bound_session_history(w, history)
        window = _fit(windowed, n_ctx=2048, retrieval=True)
        LLMWorker._note_token_fit_salvage(w, window, len(windowed))
        first = list(w._pending_salvage_message_ids)
        LLMWorker._note_token_fit_salvage(w, window, len(windowed))
        self.assertEqual(w._pending_salvage_message_ids, first)

    def test_a_count_mismatch_keeps_only_the_window_drops(self):
        w = _worker(limit=10)
        windowed = LLMWorker._bound_session_history(w, _session(13))
        LLMWorker._note_token_fit_salvage(w, HistoryWindow(windowed[3:], 3, 0, 100, "trimmed"), len(windowed) + 1)
        self.assertEqual(w._pending_salvage_message_ids, ["m0", "m1", "m2", "m3"])

    def test_a_truncated_message_is_kept_not_salvaged(self):
        history = [
            {"id": "a", "role": "user", "content": "word " * 2000},
            {"id": "b", "role": "assistant", "content": "ok"},
            {"id": "c", "role": "user", "content": "And now?"},
        ]
        w = _worker(limit=10)
        windowed = LLMWorker._bound_session_history(w, history)
        window = fit_history_to_token_budget(windowed, 400)
        LLMWorker._note_token_fit_salvage(w, window, len(windowed))
        self.assertTrue(window.truncated)
        self.assertEqual(w._pending_salvage_message_ids, [])

    def test_worker_wiring(self):
        src = (ROOT / "workers" / "llm_worker.py").read_text(encoding="utf-8")
        call = src[src.index("history_window = self._fit_history_for_turn(") :][:500]
        self.assertIn("self._note_token_fit_salvage(history_window,", call)
        self.assertLess(call.index("_note_token_fit_salvage"), call.index("history = history_window.messages"))
        turn = src[src.index("def _execute_llm_turn") :][:1500]
        self.assertIn("self._turn_history_ids", turn)
        payload = src[src.index('"salvage_message_ids": list(') :][:200]
        self.assertIn("_pending_salvage_message_ids", payload)


# ---------------------------------------------------------------------------
# The enrichment worker: each message once
# ---------------------------------------------------------------------------


class TestSalvageOnce(unittest.TestCase):
    def setUp(self):
        self.ew = _load_enrichment_worker()
        self.clock = [10_000.0]
        self.ew.time = SimpleNamespace(time=lambda: self.clock[0])
        self.worker = self.ew.EnrichmentWorker(llm=MagicMock(), embedder=MagicMock(), store=MagicMock(), db=MagicMock())
        self.worker._wait_for_chat_llm_idle = lambda: True
        self.history = _session(40)
        self.worker.db.get_session_history.return_value = self.history
        self.worker._generate_memory = lambda _p: "[]"
        self.worker._extract_json_facts = lambda _r: []
        self.seen: list[list[str]] = []

        def scrub(msgs):
            self.seen.append([m["id"] for m in msgs])
            return msgs

        self.worker._scrub_assistant_failures = scrub

    def _run(self, ids, session="sess"):
        self.worker._process_salvage({"session_id": session, "salvage_message_ids": ids, "enrichment_mode": "salvage"})

    def _later(self):
        self.clock[0] += self.ew.SALVAGE_RATE_LIMIT_SEC

    def test_each_message_once_oldest_first(self):
        ids = [f"m{i}" for i in range(30)]
        self._run(ids)
        self.assertEqual(self.seen, [ids[:24]])
        self._later()
        self._run(ids + ["m30", "m31"])
        self.assertEqual(self.seen[1], ids[24:] + ["m30", "m31"])
        self._later()
        self.worker.db.get_session_history.reset_mock()
        self._run(ids + ["m30", "m31"])
        self.assertEqual(len(self.seen), 2)
        self.worker.db.get_session_history.assert_not_called()

    def test_nothing_new_does_not_spend_the_rate_limit(self):
        self._run(["m0", "m1"])
        self._later()
        self._run(["m0", "m1"])  # nothing new: no stamp
        self.clock[0] += 1
        self._run(["m0", "m1", "m2"])
        self.assertEqual(self.seen, [["m0", "m1"], ["m2"]])

    def test_a_skipped_run_marks_nothing(self):
        self.worker._salvage_last_ts["sess"] = self.clock[0]  # rate-limited
        self._run(["m0"])
        self.assertEqual(self.worker._salvaged_ids, {})
        self._later()
        self.worker._wait_for_chat_llm_idle = lambda: False
        self._run(["m0"])
        self.assertEqual(self.worker._salvaged_ids, {})
        self._later()
        self.worker._wait_for_chat_llm_idle = lambda: True
        self.worker.db.get_session_history.side_effect = RuntimeError("locked")
        self._run(["m0"])
        self.assertEqual(self.worker._salvaged_ids, {})
        self._later()
        self.worker.db.get_session_history.side_effect = None
        self._run(["m0"])
        self.assertEqual(self.seen, [["m0"]])

    def test_sessions_are_separate_and_ids_deduplicated(self):
        self._run(["m0", "m1", "m0"])
        self._run(["m0", "m1"], session="other")
        self.assertEqual(self.seen, [["m0", "m1"], ["m0", "m1"]])

    def test_a_deleted_message_is_not_asked_for_again(self):
        self._run(["gone"])
        self.assertEqual(self.seen, [])
        self.assertEqual(self.worker._salvaged_ids["sess"], {"gone"})


# ---------------------------------------------------------------------------
# The measurement: a growing chat, turn by turn
# ---------------------------------------------------------------------------


class TestGrowingChat(_SalvageOn):
    """60 turns a minute apart, window 10, sources on every other turn (the D26 measurement)."""

    def _chat(self, n_ctx: int):
        ew = _load_enrichment_worker()
        clock = [0.0]
        ew.time = SimpleNamespace(time=lambda: clock[0])
        worker = ew.EnrichmentWorker(llm=MagicMock(), embedder=MagicMock(), store=MagicMock(), db=MagicMock())
        worker._wait_for_chat_llm_idle = lambda: True
        worker._generate_memory = lambda _p: "[]"
        worker._extract_json_facts = lambda _r: []
        full = _session(120, seed=11)
        worker.db.get_session_history.return_value = full
        seen: list[str] = []
        worker._scrub_assistant_failures = lambda msgs: (seen.extend(m["id"] for m in msgs), msgs)[1]
        dropped: set[str] = set()
        fit_dropped = 0
        for turn in range(1, 61):
            history = full[: 2 * turn - 1]
            w = _worker(limit=10)
            windowed = LLMWorker._bound_session_history(w, history)
            window = _fit(windowed, n_ctx=n_ctx, retrieval=bool(turn % 2))
            fit_dropped += window.dropped
            LLMWorker._note_token_fit_salvage(w, window, len(windowed))
            dropped.update(w._pending_salvage_message_ids)
            clock[0] = 1_000_000.0 + turn * 60.0
            worker._process_salvage(
                {"session_id": "s", "salvage_message_ids": w._pending_salvage_message_ids, "enrichment_mode": "salvage"}
            )
        return seen, dropped, fit_dropped

    def test_every_dropped_message_reaches_salvage_once(self):
        # n_ctx, messages ever dropped, messages salvaged. At 2048 the token fit
        # drops more; the 6 left were dropped within the last rate-limit window.
        for n_ctx, n_dropped, n_salvaged in ((8192, 110, 110), (2048, 118, 112)):
            with self.subTest(n_ctx=n_ctx):
                seen, dropped, fit_dropped = self._chat(n_ctx)
                self.assertEqual(len(seen), len(set(seen)))  # each message once
                self.assertEqual((len(dropped), len(seen)), (n_dropped, n_salvaged))
                self.assertLessEqual(set(seen), dropped)
                if n_ctx == 2048:
                    self.assertGreater(fit_dropped, 0)


if __name__ == "__main__":
    unittest.main()
