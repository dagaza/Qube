"""Plan S17 (V20): token-aware chat history windowing.

History windowing counted messages (default 10, each up to 14 000 chars), so
history alone could overflow a 4k window, and the S10 retrieval budget then sat
at its 1000-char floor. ``history_token_budget`` / ``fit_history_to_token_budget``
drop the oldest messages until history fits, keeping room for retrieval on
turns with sources. The matrix reuses the S10 worst-case prompt helpers.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_TESTS = Path(__file__).resolve().parent
for _p in (str(_TESTS.parent), str(_TESTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.retrieval_budget import (  # noqa: E402
    DEFAULT_RETRIEVAL_CHARS,
    TRUNCATION_SUFFIX,
    estimate_tokens,
    fit_history_to_token_budget,
    history_token_budget,
    message_tokens,
    output_reserve_tokens,
    retrieval_char_budget,
)
from test_s10_evidence_pack import _MATRIX, _format, _history, _prompt_tokens, _sources  # noqa: E402

_SKILLS = "S" * 1200
# Up to the overflow cases: V20's ten 1500-char turns, a 30-message window, and
# ten messages at the 14 000-char single-message cap.
_HISTORY_SIZES = ((0, 0), (4, 300), (6, 600), (8, 900), (10, 1500), (30, 1500), (10, 14000))


def _turn(n_ctx: int, history: list[dict], *, has_retrieval: bool = True, count_tokens=None):
    """What the worker does per turn: fit history, then size the block from the kept history."""
    window = fit_history_to_token_budget(
        history,
        history_token_budget(
            n_ctx=n_ctx,
            skills_chars=len(_SKILLS),
            max_output_tokens=4096,
            reserve_retrieval=has_retrieval,
        ),
        count_tokens=count_tokens,
    )
    budget = retrieval_char_budget(
        n_ctx=n_ctx,
        history=window.messages,
        skills_chars=len(_SKILLS),
        max_output_tokens=4096,
        count_tokens=count_tokens,
    )
    return window, budget


def _dense(text: str) -> int:
    """A tokenizer at 2 chars per token (tables, code, logs: the estimate assumes 4)."""
    return (len(text) + 1) // 2


def _msgs(*roles_and_chars: tuple[str, int]) -> list[dict]:
    return [{"role": r, "content": "x" * n, "id": f"m{i}"} for i, (r, n) in enumerate(roles_and_chars)]


class HistoryTokenBudgetTests(unittest.TestCase):
    def test_unknown_n_ctx_means_no_limit(self) -> None:
        for n_ctx in (None, 0, -1, "x"):
            self.assertIsNone(history_token_budget(n_ctx=n_ctx))  # type: ignore[arg-type]
        history = _history(30, 14000)
        window = fit_history_to_token_budget(history, None)
        self.assertEqual(window.reason, "no_ctx")
        self.assertEqual(len(window.messages), len(history))
        self.assertEqual(window.dropped, 0)

    def test_retrieval_turn_splits_the_room(self) -> None:
        # 4k: 4096 - 1024 reserve - 1500 overhead (4800 + 1200 skills chars) - 128 margin = 1444.
        self.assertEqual(history_token_budget(n_ctx=4096, skills_chars=1200, max_output_tokens=4096, reserve_retrieval=False), 1444)
        # Small window: retrieval keeps half (722), history the other half.
        self.assertEqual(history_token_budget(n_ctx=4096, skills_chars=1200, max_output_tokens=4096), 722)
        # Large window: retrieval keeps its full cap (8000 chars = 2000 tokens), history the rest.
        room_32k = 32768 - 4096 - 1500 - 128
        self.assertEqual(history_token_budget(n_ctx=32768, skills_chars=1200, max_output_tokens=4096), room_32k - 2000)

    def test_extra_context_comes_out_of_the_history_budget(self) -> None:
        # 32k: retrieval keeps its full cap either way, so history pays the whole 1000 tokens.
        base = history_token_budget(n_ctx=32768, skills_chars=1200, max_output_tokens=4096)
        with_attachment = history_token_budget(
            n_ctx=32768, skills_chars=1200, max_output_tokens=4096, extra_context_chars=4000
        )
        self.assertEqual(base - with_attachment, 1000)
        # 8k: the room drops below twice the cap, so the cost is shared with retrieval.
        base_8k = history_token_budget(n_ctx=8192, skills_chars=1200, max_output_tokens=4096)
        with_attachment_8k = history_token_budget(
            n_ctx=8192, skills_chars=1200, max_output_tokens=4096, extra_context_chars=4000
        )
        self.assertEqual((base_8k, with_attachment_8k), (4516 - 2000, 3516 - 3516 // 2))

    def test_budget_never_negative(self) -> None:
        self.assertEqual(history_token_budget(n_ctx=1024, skills_chars=20000, max_output_tokens=4096), 0)


class FitHistoryTests(unittest.TestCase):
    def test_fits_returns_the_same_messages(self) -> None:
        history = _msgs(("user", 400), ("assistant", 400), ("user", 40))
        window = fit_history_to_token_budget(history, 1000)
        self.assertEqual(window.reason, "fits")
        self.assertEqual(window.dropped, 0)
        self.assertEqual(len(window.messages), 3)
        for kept, orig in zip(window.messages, history):
            self.assertIs(kept, orig)  # ids and any other keys survive

    def test_drops_oldest_first_and_keeps_a_contiguous_tail(self) -> None:
        history = _msgs(("user", 800), ("assistant", 800), ("user", 800), ("assistant", 800), ("user", 40))
        per = message_tokens(history[0])  # 208
        window = fit_history_to_token_budget(history, 2 * per + message_tokens(history[-1]) + 5)
        self.assertEqual(window.reason, "trimmed")
        self.assertEqual([m["id"] for m in window.messages], ["m2", "m3", "m4"])
        self.assertEqual(window.dropped, 2)
        self.assertLessEqual(window.tokens, window.budget_tokens)

    def test_leading_assistant_is_dropped(self) -> None:
        history = _msgs(("user", 800), ("assistant", 800), ("user", 40))
        # Room for the last two only: the tail would start with the assistant turn.
        window = fit_history_to_token_budget(history, message_tokens(history[1]) + message_tokens(history[2]))
        self.assertEqual([m["id"] for m in window.messages], ["m2"])
        self.assertEqual(window.messages[0]["role"], "user")
        self.assertEqual(window.dropped, 2)

    def test_current_question_is_always_kept(self) -> None:
        history = _msgs(("user", 400), ("assistant", 400), ("user", 20000))
        window = fit_history_to_token_budget(history, 100)
        self.assertEqual(window.reason, "over_budget")
        self.assertEqual([m["id"] for m in window.messages], ["m2"])
        self.assertGreater(window.tokens, 100)

    def test_trace(self) -> None:
        window = fit_history_to_token_budget(_msgs(("user", 4000), ("assistant", 4000), ("user", 40)), 500)
        self.assertEqual(
            window.as_trace(),
            {
                "kept": 1,
                "dropped": 2,
                "tokens": window.tokens,
                "budget_tokens": 500,
                "reason": "trimmed",
                "truncated": False,
            },
        )

    def test_boundary_user_message_is_kept_truncated(self) -> None:
        history = _msgs(("user", 4000), ("assistant", 800), ("user", 40))
        budget = message_tokens(history[1]) + message_tokens(history[2]) + 300
        window = fit_history_to_token_budget(history, budget)
        self.assertEqual(window.reason, "trimmed")
        self.assertTrue(window.truncated)
        self.assertEqual([m["id"] for m in window.messages], ["m0", "m1", "m2"])
        head = window.messages[0]
        self.assertTrue(head["content"].endswith(TRUNCATION_SUFFIX))
        self.assertLess(len(head["content"]), len(history[0]["content"]))
        self.assertEqual(head["role"], "user")
        self.assertIsNot(head, history[0])  # a copy; the session history is not modified
        self.assertEqual(len(history[0]["content"]), 4000)
        self.assertLessEqual(window.tokens, budget)
        self.assertGreater(window.tokens, budget - 10)  # the room is used, not wasted
        self.assertEqual(window.dropped, 0)

    def test_little_room_drops_instead_of_truncating(self) -> None:
        history = _msgs(("user", 4000), ("assistant", 800), ("user", 40))
        budget = message_tokens(history[1]) + message_tokens(history[2]) + 100  # < 128 left
        window = fit_history_to_token_budget(history, budget)
        self.assertFalse(window.truncated)
        self.assertEqual([m["id"] for m in window.messages], ["m2"])  # leading assistant dropped too

    def test_boundary_assistant_message_is_not_truncated(self) -> None:
        # Kept truncated, it would lead the history; templates expect a user turn first.
        history = _msgs(("user", 400), ("assistant", 4000), ("user", 40))
        window = fit_history_to_token_budget(history, message_tokens(history[2]) + 500)
        self.assertFalse(window.truncated)
        self.assertEqual([m["id"] for m in window.messages], ["m2"])

    def test_model_counter_is_used_for_every_cost(self) -> None:
        history = _history(10, 1500)
        by_estimate = fit_history_to_token_budget(history, 2000)
        by_model = fit_history_to_token_budget(history, 2000, count_tokens=_dense)
        self.assertLess(len(by_model.messages), len(by_estimate.messages))
        self.assertEqual(by_model.tokens, sum(_dense(m["content"]) + 8 for m in by_model.messages))
        self.assertLessEqual(by_model.tokens, 2000)

    def test_messages_older_than_the_cut_are_not_tokenized(self) -> None:
        seen: list[int] = []

        def counting(text: str) -> int:
            seen.append(len(text))
            return estimate_tokens(text)

        history = _msgs(*([("user", 1000), ("assistant", 1000)] * 40), ("user", 40))
        fit_history_to_token_budget(history, 1500, count_tokens=counting)
        self.assertLess(len(seen), 30)  # 81 messages; only the tail and the truncation search


class HistoryMatrixTests(unittest.TestCase):
    """Acceptance: on 4k / 8k / 32k, prompt + answer reserve fit the window, history included."""

    def test_matrix_prompt_never_exceeds_n_ctx(self) -> None:
        for n_ctx in _MATRIX:
            reserve = output_reserve_tokens(n_ctx, 4096)
            for turns, chars in _HISTORY_SIZES:
                for has_retrieval in (True, False):
                    with self.subTest(n_ctx=n_ctx, turns=turns, chars=chars, retrieval=has_retrieval):
                        window, budget = _turn(n_ctx, _history(turns, chars), has_retrieval=has_retrieval)
                        self.assertNotEqual(window.reason, "over_budget")
                        self.assertEqual(window.messages[-1]["role"], "user")
                        body = _format(_sources(), budget.chars) if has_retrieval else ""
                        if has_retrieval:
                            # Neither starves: retrieval never falls to the floor.
                            self.assertNotEqual(budget.reason, "floor")
                        prompt = _prompt_tokens(window.messages, body, skills=_SKILLS, service="scientific_evidence")
                        self.assertLessEqual(prompt + reserve, n_ctx)

    def test_v20_case_overflowed_before_and_fits_now(self) -> None:
        history = _history(10, 1500)
        before = retrieval_char_budget(n_ctx=4096, history=history, skills_chars=len(_SKILLS), max_output_tokens=4096)
        self.assertEqual(before.reason, "floor")
        untrimmed = _prompt_tokens(history, _format(_sources(), before.chars), skills=_SKILLS, service="general_web")
        self.assertGreater(untrimmed + output_reserve_tokens(4096, 4096), 4096)

        window, after = _turn(4096, history)
        self.assertEqual(window.reason, "trimmed")
        self.assertGreater(after.chars, before.chars)
        trimmed = _prompt_tokens(window.messages, _format(_sources(), after.chars), skills=_SKILLS, service="general_web")
        self.assertLessEqual(trimmed + output_reserve_tokens(4096, 4096), 4096)

    def test_everything_fits_is_unchanged(self) -> None:
        """When history and a full retrieval block fit, S17 changes nothing (S10's budget too)."""
        cases = [(32768, size) for size in _HISTORY_SIZES[:5]] + [(8192, size) for size in _HISTORY_SIZES[:4]]
        for n_ctx, (turns, chars) in cases:
            for has_retrieval in (True, False):
                with self.subTest(n_ctx=n_ctx, turns=turns, chars=chars, retrieval=has_retrieval):
                    history = _history(turns, chars)
                    window, budget = _turn(n_ctx, history, has_retrieval=has_retrieval)
                    self.assertEqual(window.reason, "fits")
                    self.assertEqual(window.messages, history)
                    s10 = retrieval_char_budget(
                        n_ctx=n_ctx, history=history, skills_chars=len(_SKILLS), max_output_tokens=4096
                    )
                    self.assertEqual(budget, s10)

    def test_small_window_keeps_the_last_exchange(self) -> None:
        """4k with sources and 1500-char turns: the previous exchange survives, cut to fit."""
        window, budget = _turn(4096, _history(10, 1500))
        self.assertEqual([m["role"] for m in window.messages], ["user", "assistant", "user"])
        self.assertTrue(window.truncated)
        self.assertNotEqual(budget.reason, "floor")

    def test_dense_text_with_the_model_counter(self) -> None:
        """Text at 2 chars/token: with the model's own count the matrix still fits."""
        for n_ctx in _MATRIX:
            reserve = output_reserve_tokens(n_ctx, 4096)
            for turns, chars in _HISTORY_SIZES:
                for has_retrieval in (True, False):
                    with self.subTest(n_ctx=n_ctx, turns=turns, chars=chars, retrieval=has_retrieval):
                        window, budget = _turn(
                            n_ctx, _history(turns, chars), has_retrieval=has_retrieval, count_tokens=_dense
                        )
                        self.assertNotEqual(window.reason, "over_budget")
                        if has_retrieval:
                            self.assertNotEqual(budget.reason, "floor")
                        history_real = sum(_dense(m["content"]) + 8 for m in window.messages)
                        # Fixed costs and the block as the S10 budget assumes them.
                        overhead = (4800 + len(_SKILLS) + 3) // 4
                        block = (budget.tokens or 0) if has_retrieval and budget.reason != "floor" else 0
                        self.assertLessEqual(history_real + overhead + block + reserve + 128, n_ctx)

    def test_no_retrieval_turn_keeps_more_history(self) -> None:
        history = _history(10, 1500)
        with_sources, _ = _turn(8192, history, has_retrieval=True)
        chat_only, _ = _turn(8192, history, has_retrieval=False)
        self.assertGreater(len(chat_only.messages), len(with_sources.messages))


class WorkerHistoryWiringTests(unittest.TestCase):
    def _fake(self, **kw):
        from workers.llm_worker import LLMWorker

        fake = SimpleNamespace(
            engine_mode="internal",
            _native_engine=SimpleNamespace(loaded_n_ctx=lambda: 4096),
            context_window=32000,
            prompt="How do heat pumps compare with gas boilers on running cost?",
            MAX_TOTAL_RETRIEVAL_CHARS=DEFAULT_RETRIEVAL_CHARS,
            _max_tokens_completion=lambda native: 4096,
        )
        for k, v in kw.items():
            setattr(fake, k, v)
        fake._active_context_window = lambda: LLMWorker._active_context_window(fake)
        return fake, LLMWorker

    def _skills(self):
        return patch(
            "workers.llm_worker.get_skill_settings",
            return_value=SimpleNamespace(total_prompt_char_budget=len(_SKILLS)),
        )

    def test_loaded_4k_window_trims_and_logs(self) -> None:
        fake, worker = self._fake()
        history = _history(10, 1500)
        with self._skills(), self.assertLogs("Qube.LLM", level="INFO") as logs:
            window = worker._fit_history_for_turn(fake, history, has_retrieval=True)
        self.assertEqual(window.reason, "trimmed")
        self.assertEqual(window.budget_tokens, 722)
        self.assertIn("token-fitted", "\n".join(logs.output))
        budget = worker._retrieval_budget_for_turn(fake, window.messages)
        self.assertEqual(budget.reason, "n_ctx")

    def test_external_32k_setting_leaves_history_alone(self) -> None:
        fake, worker = self._fake(engine_mode="external")
        history = _history(10, 1500)
        with self._skills():
            window = worker._fit_history_for_turn(fake, history, has_retrieval=True)
        self.assertEqual(window.reason, "fits")
        self.assertEqual(window.messages, history)

    def test_native_counter_uses_the_loaded_tokenizer(self) -> None:
        from workers.llm_worker import _turn_token_counter
        from workers.native_llama_engine import NativeLlamaEngine

        engine = NativeLlamaEngine.__new__(NativeLlamaEngine)
        engine._llama = None
        self.assertIsNone(engine.count_tokens("hello"))
        engine._llama = SimpleNamespace(tokenize=lambda b, add_bos=False, special=True: list(b))  # 1 token/byte
        self.assertEqual(engine.count_tokens("hello"), 5)

        fake, _ = self._fake(_native_engine=engine)
        count = _turn_token_counter(fake)
        self.assertIsNotNone(count)
        self.assertEqual(count("abcdefgh"), 8)  # the model's count, not the estimate (2)

    def test_counter_falls_back_to_the_estimate(self) -> None:
        from workers.llm_worker import _turn_token_counter

        fake, _ = self._fake(engine_mode="external")
        self.assertIsNone(_turn_token_counter(fake))
        fake, _ = self._fake(_native_engine=SimpleNamespace(count_tokens=lambda text: None))
        self.assertIsNone(_turn_token_counter(fake))  # model not loaded
        fake, _ = self._fake(_native_engine=SimpleNamespace(loaded_n_ctx=lambda: 4096))
        self.assertIsNone(_turn_token_counter(fake))  # engine without count_tokens

        calls = {"n": 0}

        def flaky(text: str):
            calls["n"] += 1
            if calls["n"] > 1:
                raise RuntimeError("tokenizer broke")
            return 1

        fake, _ = self._fake(_native_engine=SimpleNamespace(count_tokens=flaky))
        count = _turn_token_counter(fake)
        self.assertEqual(count("abcdefgh"), 2)  # this text falls back to the estimate

    def test_worker_fits_with_the_turn_counter(self) -> None:
        fake, worker = self._fake()
        history = _history(10, 1500)
        with self._skills():
            by_estimate = worker._fit_history_for_turn(fake, history, has_retrieval=False)
            by_model = worker._fit_history_for_turn(fake, history, has_retrieval=False, count_tokens=_dense)
            budget = worker._retrieval_budget_for_turn(fake, by_model.messages, count_tokens=_dense)
        self.assertLess(len(by_model.messages), len(by_estimate.messages))
        s10 = worker._retrieval_budget_for_turn(fake, by_model.messages)
        self.assertLessEqual(budget.chars, s10.chars)  # dense history leaves less room

    def test_routing_debug_snapshot_carries_the_window(self) -> None:
        from mcp.routing_debug import build_retrieval_outcome_snapshot

        trace = {"kept": 3, "dropped": 8, "tokens": 700, "budget_tokens": 722, "reason": "trimmed"}
        snap = build_retrieval_outcome_snapshot(
            decision={"history_window": trace},
            execution_route_pre_downgrade="WEB",
            execution_route_final="WEB",
            memory_hits=0,
            rag_hits=0,
            web_hits=3,
        )
        self.assertEqual(snap["history_window"], trace)
        empty = build_retrieval_outcome_snapshot(
            decision={},
            execution_route_pre_downgrade="NONE",
            execution_route_final="NONE",
            memory_hits=0,
            rag_hits=0,
            web_hits=0,
        )
        self.assertEqual(empty["history_window"], {})


if __name__ == "__main__":
    unittest.main()
