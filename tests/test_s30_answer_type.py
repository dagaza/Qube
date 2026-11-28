"""Plan S30: which kind of source answers a question sent to Evidence (shadow mode, D67).

The sidecar answers GUIDE / REFERENCE / RESEARCH (prompt v4, chosen on the
knowledge-quality fixtures); a word rule built from those fixtures' how-to wordings
backs it. One ``[AnswerType] shadow`` line per Evidence turn, with no question
text; nothing changes the service, the sources or the answer.
"""
from __future__ import annotations

import json
import logging
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.answer_type_llm import (  # noqa: E402
    ANSWER_TYPE_QUESTION_CHARS,
    ANSWER_TYPE_SYSTEM,
    GATE_DISABLED,
    GATE_NO_QUESTION,
    GATE_NOT_EVIDENCE,
    GATE_SCHEDULED,
    GATE_SIDECAR_BUSY,
    GATE_SIDECAR_UNAVAILABLE,
    MODE_DEFAULT,
    MODE_EXPLICIT,
    answer_type_gate,
    answer_type_mode,
    build_answer_type_prompt,
    classify_answer_type,
    parse_answer_type_output,
    rule_answer_type,
    run_shadow_answer_type,
)
from core.cognition_prompt_adapter import QWEN3_EMPTY_THINK  # noqa: E402
from core.sidecar_prompts import build_prompt_for_task, parse_task_output, task_inference_params  # noqa: E402
from core.sidecar_types import SidecarResult, SidecarTask  # noqa: E402

_RECORDED = ROOT / "tests" / "fixtures" / "answer_type_recorded_s30.json"
_HELDOUT = ROOT / "eval" / "retrieval_corpus" / "source_kind_v1.json"
_QWEN = "/models/cognition/Qwen3-1.7B-Q6_K.gguf"
_EVIDENCE = "scientific_evidence"
_RADIATOR = "How do I bleed a radiator?"


class _ReplayClient:
    """Sidecar client that answers from a recorded completion via the production parser."""

    available = True

    def __init__(self, raw: str = "", *, error: str = "", raises: bool = False) -> None:
        self.raw, self.error, self.raises = raw, error, raises
        self.calls: list[tuple[SidecarTask, dict]] = []

    def complete(self, task, *, timeout_sec=0.0, **payload):
        self.calls.append((task, payload))
        if self.raises:
            raise RuntimeError("boom")
        if self.error:
            return SidecarResult(ok=False, error=self.error, task=task)
        return parse_task_output(task, self.raw, **payload)


def _flagged(kind) -> bool:
    return kind in ("practical", "reference")


class TestParser(unittest.TestCase):
    def test_the_first_word_names_the_kind(self):
        for raw, expected in (
            ("GUIDE", "practical"),
            (" RESEARCH\n", "scholarly"),
            ("REFERENCE.", "reference"),
            ("guide", "practical"),
            ("<think>\n\n</think>\n\nGUIDE", "practical"),
            ("", None),
            ("Maybe GUIDE", None),
        ):
            with self.subTest(raw=raw):
                self.assertEqual(parse_answer_type_output(raw), expected)

    def test_sidecar_task_wiring(self):
        self.assertEqual(task_inference_params(SidecarTask.answer_type), {"max_tokens": 4, "temperature": 0.0})
        res = parse_task_output(SidecarTask.answer_type, "GUIDE")
        self.assertTrue(res.ok)
        self.assertEqual(res.parsed, {"answer_type": "practical"})
        bad = parse_task_output(SidecarTask.answer_type, "unsure")
        self.assertFalse(bad.ok)
        self.assertEqual(bad.error, "parse_fail")


class TestPrompt(unittest.TestCase):
    def test_qwen3_prompt_prefills_the_empty_think_block(self):
        p = build_answer_type_prompt(question=_RADIATOR, model_path=_QWEN)
        self.assertTrue(p.endswith("<|im_start|>assistant\n" + QWEN3_EMPTY_THINK))
        self.assertIn(ANSWER_TYPE_SYSTEM, p)
        self.assertIn(f"Question: {_RADIATOR}", p)

    def test_non_qwen_prompt_has_no_think_block(self):
        p = build_answer_type_prompt(question=_RADIATOR, chat_format="llama-3", model_path="/m/llama-3.gguf")
        self.assertNotIn("<think>", p)

    def test_build_prompt_for_task_routes_here(self):
        self.assertEqual(
            build_prompt_for_task(SidecarTask.answer_type, chat_format="chatml", model_path=_QWEN, question=_RADIATOR),
            build_answer_type_prompt(question=_RADIATOR, chat_format="chatml", model_path=_QWEN),
        )

    def test_the_question_is_capped(self):
        p = build_answer_type_prompt(question="x" * 1000, model_path=_QWEN)
        self.assertIn("x" * ANSWER_TYPE_QUESTION_CHARS, p)
        self.assertNotIn("x" * (ANSWER_TYPE_QUESTION_CHARS + 1), p)


class TestRule(unittest.TestCase):
    def test_how_to_wordings(self):
        for q, expected in (
            ("how do I undo the last git commit but keep my changes", "practical"),
            ("how to descale an electric kettle with vinegar", "practical"),
            ("What is a sourdough starter and how do you maintain it?", "practical"),
            ("How does the OAuth 2.0 authorization code flow work?", None),
            ("How does self-attention work in transformer models?", None),
        ):
            with self.subTest(q=q):
                self.assertEqual(rule_answer_type(q), expected)

    def test_on_the_frozen_held_out_set(self):
        cases = json.loads(_HELDOUT.read_text(encoding="utf-8"))["cases"]
        flagged = {c["id"] for c in cases if rule_answer_type(c["prompt"])}
        practical = {c["id"] for c in cases if c["source_kind"] == "practical"}
        self.assertEqual(len(flagged), 9)
        self.assertLessEqual(flagged, practical)
        self.assertEqual(practical - flagged, {"comp_kerberos_ticket", "health_bp_home", "home_store_herbs"})


class TestGate(unittest.TestCase):
    def test_each_reason(self):
        client = _ReplayClient("GUIDE")
        for kwargs, expected in (
            (dict(service=_EVIDENCE, question=_RADIATOR, sidecar_client=client, enabled=False), GATE_DISABLED),
            (dict(service="general_web", question=_RADIATOR, sidecar_client=client, enabled=True), GATE_NOT_EVIDENCE),
            (dict(service=_EVIDENCE, question="  ", sidecar_client=client, enabled=True), GATE_NO_QUESTION),
            (dict(service=_EVIDENCE, question=_RADIATOR, sidecar_client=None, enabled=True), GATE_SIDECAR_UNAVAILABLE),
            (dict(service=_EVIDENCE, question=_RADIATOR, sidecar_client=SimpleNamespace(available=False, complete=None),
                  enabled=True), GATE_SIDECAR_UNAVAILABLE),
            (dict(service=_EVIDENCE, question=_RADIATOR, sidecar_client=client, enabled=True, queue_depth=1), GATE_SIDECAR_BUSY),
            (dict(service=_EVIDENCE, question=_RADIATOR, sidecar_client=client, enabled=True), GATE_SCHEDULED),
        ):
            with self.subTest(expected=expected):
                kw = dict(kwargs)
                self.assertEqual(answer_type_gate(kw.pop("service"), kw.pop("question"), kw.pop("sidecar_client"), **kw), expected)

    def test_mode(self):
        for tool in ("evidence", "science", "pubmed", "arxiv", " Evidence "):
            self.assertEqual(answer_type_mode(tool), MODE_EXPLICIT)
        for tool in (None, "", "internet", "source:openalex"):
            self.assertEqual(answer_type_mode(tool), MODE_DEFAULT)


class TestShadow(unittest.TestCase):
    def _run(self, **kw):
        seen: list[dict] = []
        args = dict(question=_RADIATOR, service=_EVIDENCE, composer_tool="evidence", enabled=True, session_id="s1",
                    on_done=seen.append, background=False)
        args.update(kw)
        with self.assertLogs("Qube.AnswerType", level=logging.INFO) as logs:
            gate = run_shadow_answer_type(**args)
        lines = [m for m in logs.output if "[AnswerType] shadow" in m]
        return gate, seen, lines

    def test_scheduled_line_has_both_answers_and_no_question_text(self):
        client = _ReplayClient("GUIDE")
        gate, seen, lines = self._run(sidecar_client=client)
        self.assertEqual(gate, GATE_SCHEDULED)
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0].split("shadow ", 1)[1])
        self.assertEqual(record, seen[0])
        self.assertEqual(record["mode"], MODE_EXPLICIT)
        self.assertEqual(record["rule"], "practical")
        self.assertEqual(record["llm"]["status"], "ok")
        self.assertEqual(record["llm"]["answer_type"], "practical")
        self.assertEqual(client.calls, [(SidecarTask.answer_type, {"question": _RADIATOR})])
        self.assertNotIn("radiator", lines[0].lower())

    def test_without_the_sidecar_the_rule_alone_is_logged(self):
        for client in (None, SimpleNamespace(available=False, complete=self.fail)):
            with self.subTest(client=client), patch("core.answer_type_llm.threading.Thread") as thread:
                gate, seen, lines = self._run(sidecar_client=client, composer_tool=None, background=True)
                self.assertEqual(gate, GATE_SIDECAR_UNAVAILABLE)
                thread.assert_not_called()
                self.assertEqual(len(lines), 1)
                self.assertEqual((seen[0]["mode"], seen[0]["rule"], seen[0]["llm"]), (MODE_DEFAULT, "practical", None))

    def test_a_busy_sidecar_is_not_asked(self):
        client = _ReplayClient("GUIDE")
        gate, seen, _lines = self._run(sidecar_client=client, queue_depth=2)
        self.assertEqual(gate, GATE_SIDECAR_BUSY)
        self.assertEqual(client.calls, [])
        self.assertIsNone(seen[0]["llm"])

    def test_nothing_is_logged_off_evidence_or_when_disabled(self):
        for kw in (dict(service="general_web", enabled=True), dict(service=_EVIDENCE, enabled=False)):
            with self.subTest(**kw), patch("core.answer_type_llm.logger") as log:
                client = _ReplayClient("GUIDE")
                run_shadow_answer_type(question=_RADIATOR, composer_tool=None, sidecar_client=client, background=False, **kw)
                log.info.assert_not_called()
                self.assertEqual(client.calls, [])

    def test_runs_off_the_calling_thread(self):
        started, release, done = threading.Event(), threading.Event(), threading.Event()

        class _Slow(_ReplayClient):
            def complete(self, task, **kw):
                started.set()
                release.wait(5)
                return super().complete(task, **kw)

        gate = run_shadow_answer_type(
            question=_RADIATOR, service=_EVIDENCE, composer_tool=None, sidecar_client=_Slow("GUIDE"), enabled=True,
            on_done=lambda _r: done.set(),
        )
        self.assertEqual(gate, GATE_SCHEDULED)
        self.assertTrue(started.wait(5))
        self.assertFalse(done.is_set())
        release.set()
        self.assertTrue(done.wait(5))

    def test_failures_never_raise(self):
        for client, status in (
            (_ReplayClient(raises=True), "sidecar_error"),
            (_ReplayClient(error="timeout"), "timeout"),
            (_ReplayClient(error="empty_output"), "parse_fail"),
            (_ReplayClient("I think it is a guide"), "parse_fail"),
        ):
            with self.subTest(status=status):
                out = classify_answer_type(_RADIATOR, client, timeout_sec=1)
                self.assertEqual((out.status, out.answer_type), (status, None))


class TestRecordedReplay(unittest.TestCase):
    """Real Qwen3-1.7B completions (tests/fixtures/answer_type_recorded_s30.json) through the production parser."""

    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads(_RECORDED.read_text(encoding="utf-8"))["cases"]

    def test_replay_matches_recording(self):
        self.assertEqual(len(self.cases), 93)
        for case in self.cases:
            with self.subTest(case=case["id"]):
                out = classify_answer_type(case["question"], _ReplayClient(case["raw"]), timeout_sec=1)
                self.assertEqual(out.status, "ok")
                self.assertEqual(out.answer_type, case["answer_type"])

    def test_the_held_out_cases_are_the_frozen_set(self):
        frozen = {c["id"]: c for c in json.loads(_HELDOUT.read_text(encoding="utf-8"))["cases"]}
        held = {c["id"]: c for c in self.cases if c["set"] == "heldout"}
        self.assertEqual(set(held), set(frozen))
        for cid, case in held.items():
            self.assertEqual((case["question"], case["gold"]), (frozen[cid]["prompt"], frozen[cid]["source_kind"]))

    def test_the_measured_rates(self):
        for which, how_tos, non_scholarly, wrong in (("dev", (8, 8), (27, 32), (2, 31)), ("heldout", (11, 12), (16, 18), (1, 12))):
            rows = [c for c in self.cases if c["set"] == which]
            with self.subTest(set=which):
                self.assertEqual(
                    (sum(_flagged(c["answer_type"]) for c in rows if c["gold"] == "practical"),
                     sum(c["gold"] == "practical" for c in rows)), how_tos)
                self.assertEqual(
                    (sum(_flagged(c["answer_type"]) for c in rows if c["gold"] != "scholarly"),
                     sum(c["gold"] != "scholarly" for c in rows)), non_scholarly)
                self.assertEqual(
                    (sum(_flagged(c["answer_type"]) for c in rows if c["gold"] == "scholarly"),
                     sum(c["gold"] == "scholarly" for c in rows)), wrong)

    def test_the_known_misses(self):
        by_id = {(c["set"], c["id"]): c for c in self.cases}
        # The protocol question shaped like OAuth's is answered RESEARCH (V95).
        self.assertEqual(by_id[("heldout", "comp_kerberos_ticket")]["answer_type"], "scholarly")
        self.assertEqual(by_id[("heldout", "learn_bullying")]["answer_type"], "practical")
        self.assertEqual(by_id[("dev", "sci_cs_oauth_flow")]["answer_type"], "practical")


class TestSettings(unittest.TestCase):
    def test_on_by_default_and_independent_of_the_sidecar_switch(self):
        from core import app_settings

        store = SimpleNamespace(get=lambda _k, default=None: default)
        with patch.object(app_settings, "_store", return_value=store), \
                patch.object(app_settings, "get_sidecar_enabled", return_value=False):
            self.assertTrue(app_settings.get_knowledge_answer_type_shadow_enabled())
        off = SimpleNamespace(get=lambda k, default=None: False if k == app_settings.KEY_KNOWLEDGE_ANSWER_TYPE_SHADOW else default)
        with patch.object(app_settings, "_store", return_value=off):
            self.assertFalse(app_settings.get_knowledge_answer_type_shadow_enabled())


class TestWorkerWiring(unittest.TestCase):
    def test_the_shadow_runs_after_the_service_is_chosen_and_before_retrieval(self):
        src = (ROOT / "workers" / "llm_worker.py").read_text(encoding="utf-8")
        chosen = src.index("turn_knowledge_service = resolve_turn_knowledge_service(")
        shadow = src.index("run_shadow_answer_type(")
        retrieval = src.index("_web_outcome = run_web_retrieval(")
        self.assertLess(chosen, shadow)
        self.assertLess(shadow, retrieval)
        call = src[shadow: src.index(")", src.index("session_id=", shadow))]
        self.assertIn("service=turn_knowledge_service", call)
        self.assertIn("enabled=get_knowledge_answer_type_shadow_enabled()", call)

    def test_the_gate_reaches_the_routing_debug_record(self):
        from mcp.routing_debug import build_retrieval_outcome_snapshot

        snap = build_retrieval_outcome_snapshot(
            decision={"answer_type_gate": GATE_SIDECAR_BUSY},
            execution_route_pre_downgrade="WEB",
            execution_route_final="WEB",
            memory_hits=0,
            rag_hits=0,
            web_hits=3,
        )
        self.assertEqual(snap["answer_type_gate"], GATE_SIDECAR_BUSY)


if __name__ == "__main__":
    unittest.main()
