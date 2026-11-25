"""Gated LLM slot extraction for follow-ups (plan S9, shadow mode)."""

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

from core.discourse_intent import FollowUpClassification, FollowUpKind  # noqa: E402
from core.query_resolution_evaluation import build_discourse_resolution  # noqa: E402
from core.query_slots_llm import (  # noqa: E402
    GATE_DETERMINISTIC_RESOLVED,
    GATE_DISABLED,
    GATE_NO_HISTORY,
    GATE_NOT_FOLLOW_UP,
    GATE_SCHEDULED,
    GATE_SIDECAR_BUSY,
    GATE_SIDECAR_UNAVAILABLE,
    QWEN3_EMPTY_THINK,
    QuerySlots,
    build_slot_prompt,
    compose_slot_query,
    extract_query_slots,
    parse_slot_output,
    run_shadow_slot_extraction,
    slot_gate,
    validate_slots,
)
from core.sidecar_prompts import build_prompt_for_task, parse_task_output, task_inference_params  # noqa: E402
from core.sidecar_types import SidecarResult, SidecarTask  # noqa: E402

_RECORDED = ROOT / "tests" / "fixtures" / "query_slots_recorded_s9.json"
_QWEN = "/models/cognition/Qwen3-1.7B-Q6_K.gguf"


def _u(text: str) -> dict[str, str]:
    return {"role": "user", "content": text}


def _a(text: str) -> dict[str, str]:
    return {"role": "assistant", "content": text}


_ASTHMA = [
    _u("My daughter was just diagnosed with asthma."),
    _a("Asthma is common in children and usually well controlled with treatment."),
]


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


class ParserTests(unittest.TestCase):
    def test_parses_completion_after_prefilled_subject(self) -> None:
        self.assertEqual(
            parse_slot_output(" asthma\nFACET: triggers in kids"),
            QuerySlots("asthma", "triggers in kids"),
        )

    def test_facet_none_and_junk(self) -> None:
        self.assertEqual(parse_slot_output(' "Canberra."  \nFACET: none'), QuerySlots("Canberra", None))
        self.assertEqual(parse_slot_output("SUBJECT: OAuth\nFACET: n/a"), QuerySlots("OAuth", None))

    def test_think_block_is_ignored(self) -> None:
        self.assertEqual(parse_slot_output("<think>\n\n</think>\n\nSUBJECT: Go\nFACET: GC"), QuerySlots("Go", "GC"))

    def test_rejects_empty_or_rambling_subject(self) -> None:
        self.assertIsNone(parse_slot_output(""))
        self.assertIsNone(parse_slot_output(" none\nFACET: cost"))
        self.assertIsNone(parse_slot_output("<think>"))
        self.assertIsNone(parse_slot_output(" the user is asking about something they said earlier in the chat"))

    def test_sidecar_task_wiring(self) -> None:
        self.assertEqual(task_inference_params(SidecarTask.query_slots), {"max_tokens": 32, "temperature": 0.0})
        res = parse_task_output(SidecarTask.query_slots, " heat pump\nFACET: installation cost")
        self.assertTrue(res.ok)
        self.assertEqual(res.parsed, {"subject": "heat pump", "facet": "installation cost"})
        bad = parse_task_output(SidecarTask.query_slots, "<think>")
        self.assertFalse(bad.ok)


class PromptTests(unittest.TestCase):
    def test_qwen3_prompt_prefills_empty_think_and_subject(self) -> None:
        p = build_slot_prompt(history=_ASTHMA, prompt="What triggers attacks?", model_path=_QWEN)
        self.assertTrue(p.endswith("<|im_start|>assistant\n" + QWEN3_EMPTY_THINK + "SUBJECT:"))
        self.assertIn("/no_think", p)
        self.assertIn("user (last message): What triggers attacks?", p)

    def test_non_qwen_prompt_has_no_think_block(self) -> None:
        p = build_slot_prompt(history=_ASTHMA, prompt="Why?", chat_format="llama-3", model_path="/m/llama-3.gguf")
        self.assertNotIn("<think>", p)
        self.assertTrue(p.endswith("SUBJECT:"))

    def test_current_turn_not_repeated_and_window_capped(self) -> None:
        history = [_u(f"turn {i}") for i in range(6)] + [_u("And now?")]
        p = build_slot_prompt(history=history, prompt="And now?", model_path=_QWEN)
        self.assertEqual(p.count("And now?"), 1)
        self.assertNotIn("turn 1\n", p)
        self.assertIn("turn 5", p)

    def test_build_prompt_for_task_routes_to_slot_prompt(self) -> None:
        kwargs = dict(history=_ASTHMA, prompt="Why?")
        self.assertEqual(
            build_prompt_for_task(SidecarTask.query_slots, chat_format="chatml", model_path=_QWEN, **kwargs),
            build_slot_prompt(chat_format="chatml", model_path=_QWEN, **kwargs),
        )

    def test_stop_token_no_longer_truncates_the_answer(self) -> None:
        """S9 audit: the "\\n\\n" stop ended Qwen3 completions at "<think>"."""
        from core.cognition_prompt_adapter import apply_qwen3_no_think_to_prompt, cognition_stop_tokens

        p = apply_qwen3_no_think_to_prompt(
            build_prompt_for_task(SidecarTask.query_slots, chat_format="chatml", model_path=_QWEN,
                                  history=_ASTHMA, prompt="Why?"),
            _QWEN,
        )
        self.assertIn("\n\n", cognition_stop_tokens("chatml"))
        # The empty think block is already in the prompt, so generation starts at the slot value.
        self.assertTrue(p.endswith("</think>\n\nSUBJECT:"))


class ValidationTests(unittest.TestCase):
    def test_grounded_subject_accepted(self) -> None:
        self.assertEqual(validate_slots(QuerySlots("asthma"), _ASTHMA, "Triggers?"), (True, "ok"))
        self.assertEqual(validate_slots(QuerySlots("Asthma attacks"), _ASTHMA, "Triggers?"), (True, "ok"))

    def test_unanchored_subjects_rejected(self) -> None:
        self.assertEqual(
            validate_slots(QuerySlots("Cutting Costs"), _ASTHMA, "How do we cut the bill?"),
            (False, "unanchored_proper_noun"),
        )
        self.assertEqual(validate_slots(QuerySlots("diabetes"), _ASTHMA, "Why?"), (False, "unanchored_subject"))
        self.assertEqual(validate_slots(QuerySlots("it"), _ASTHMA, "Why?"), (False, "subject_is_reference"))
        self.assertEqual(validate_slots(QuerySlots("MS"), _ASTHMA, "Why?"), (False, "no_content_word"))

    def test_compose_uses_the_s6_composer(self) -> None:
        self.assertEqual(
            compose_slot_query("What usually triggers attacks in kids?", QuerySlots("asthma", "triggers")),
            "asthma What usually triggers attacks in kids?",
        )
        # A turn that already names the subject is used as it stands.
        self.assertEqual(compose_slot_query("And asthma in adults?", QuerySlots("asthma")), "asthma in adults?")


class GateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _ReplayClient(" asthma\nFACET: triggers")
        self.fu, _state, self.resolved = build_discourse_resolution("What usually triggers attacks in kids?", _ASTHMA)

    def _gate(self, **kw):
        args = dict(follow_up=self.fu, resolved=self.resolved, history=_ASTHMA,
                    prompt="What usually triggers attacks in kids?", sidecar_client=self.client)
        enabled = kw.pop("enabled", True)
        depth = kw.pop("queue_depth", 0)
        args.update(kw)
        return slot_gate(args.pop("follow_up"), args.pop("resolved"), args.pop("history"), args.pop("prompt"),
                         args.pop("sidecar_client"), enabled=enabled, queue_depth=depth)

    def test_unresolved_follow_up_is_scheduled(self) -> None:
        # Classified as a follow-up below the activation threshold, and S6 found no subject.
        self.assertFalse(self.fu.active)
        self.assertNotEqual(self.fu.kind, FollowUpKind.NONE)
        self.assertEqual(self._gate(), GATE_SCHEDULED)

    def test_each_skip_reason(self) -> None:
        self.assertEqual(self._gate(enabled=False), GATE_DISABLED)
        self.assertEqual(self._gate(follow_up=FollowUpClassification(FollowUpKind.NONE, 0.0)), GATE_NOT_FOLLOW_UP)
        self.assertEqual(self._gate(follow_up=None), GATE_NOT_FOLLOW_UP)
        self.assertEqual(self._gate(history=[]), GATE_NO_HISTORY)
        self.assertEqual(self._gate(sidecar_client=None), GATE_SIDECAR_UNAVAILABLE)
        self.assertEqual(self._gate(sidecar_client=SimpleNamespace(available=False, complete=lambda *a, **k: None)),
                         GATE_SIDECAR_UNAVAILABLE)
        self.assertEqual(self._gate(queue_depth=2), GATE_SIDECAR_BUSY)

    def test_deterministically_resolved_turn_is_skipped(self) -> None:
        history = [_u("What is OAuth?"), _a("OAuth is an authorization framework.")]
        fu, _s, resolved = build_discourse_resolution("How is it different from SAML?", history)
        self.assertEqual(
            slot_gate(fu, resolved, history, "How is it different from SAML?", self.client, enabled=True),
            GATE_DETERMINISTIC_RESOLVED,
        )


class ExtractionAndShadowTests(unittest.TestCase):
    def test_extract_ok(self) -> None:
        out = extract_query_slots("What usually triggers attacks in kids?", _ASTHMA,
                                  _ReplayClient(" asthma\nFACET: triggers"), timeout_sec=5)
        self.assertEqual((out.status, out.subject, out.composed_query),
                         ("ok", "asthma", "asthma What usually triggers attacks in kids?"))

    def test_extract_failures_never_raise(self) -> None:
        self.assertEqual(extract_query_slots("Why?", _ASTHMA, _ReplayClient(error="timeout"), timeout_sec=1).status, "timeout")
        self.assertEqual(extract_query_slots("Why?", _ASTHMA, _ReplayClient("<think>"), timeout_sec=1).status, "parse_fail")
        self.assertEqual(extract_query_slots("Why?", _ASTHMA, _ReplayClient(raises=True), timeout_sec=1).status,
                         "sidecar_error")
        rejected = extract_query_slots("Why?", _ASTHMA, _ReplayClient(" diabetes"), timeout_sec=1)
        self.assertEqual((rejected.status, rejected.reason, rejected.composed_query),
                         ("rejected", "unanchored_subject", None))

    def test_shadow_logs_comparison_and_calls_back(self) -> None:
        fu, _s, resolved = build_discourse_resolution("What usually triggers attacks in kids?", _ASTHMA)
        seen: list[dict] = []
        with self.assertLogs("Qube.QuerySlots", level=logging.INFO) as logs:
            gate = run_shadow_slot_extraction(
                prompt="What usually triggers attacks in kids?", history=_ASTHMA, follow_up=fu,
                resolved=resolved, sidecar_client=_ReplayClient(" asthma\nFACET: triggers"),
                enabled=True, timeout_sec=5, session_id="s1", on_done=seen.append, background=False,
            )
        self.assertEqual(gate, GATE_SCHEDULED)
        record = seen[0]
        self.assertEqual(record["deterministic"]["web_text"], resolved.web_text)
        self.assertEqual(record["slots"]["subject"], "asthma")
        self.assertEqual(record["slots"]["status"], "ok")
        line = next(m for m in logs.output if "[QuerySlots] shadow" in m)
        self.assertEqual(json.loads(line.split("shadow ", 1)[1])["session_id"], "s1")

    def test_shadow_runs_off_the_calling_thread(self) -> None:
        fu, _s, resolved = build_discourse_resolution("What usually triggers attacks in kids?", _ASTHMA)
        started, release, done = threading.Event(), threading.Event(), threading.Event()

        class _Slow(_ReplayClient):
            def complete(self, task, **kw):
                started.set()
                release.wait(5)
                return super().complete(task, **kw)

        gate = run_shadow_slot_extraction(
            prompt="What usually triggers attacks in kids?", history=_ASTHMA, follow_up=fu, resolved=resolved,
            sidecar_client=_Slow(" asthma"), enabled=True, timeout_sec=5,
            on_done=lambda _r: done.set(),
        )
        # Returned while the sidecar call is still blocked.
        self.assertEqual(gate, GATE_SCHEDULED)
        self.assertTrue(started.wait(5))
        self.assertFalse(done.is_set())
        release.set()
        self.assertTrue(done.wait(5))

    def test_zero_impact_when_sidecar_unavailable(self) -> None:
        fu, _s, resolved = build_discourse_resolution("What usually triggers attacks in kids?", _ASTHMA)
        for client in (None, SimpleNamespace(available=False, complete=self.fail)):
            with self.subTest(client=client), patch("core.query_slots_llm.threading.Thread") as thread:
                gate = run_shadow_slot_extraction(
                    prompt="What usually triggers attacks in kids?", history=_ASTHMA, follow_up=fu,
                    resolved=resolved, sidecar_client=client, enabled=True, timeout_sec=5,
                )
                self.assertEqual(gate, GATE_SIDECAR_UNAVAILABLE)
                thread.assert_not_called()


class RecordedOutputReplayTests(unittest.TestCase):
    """Real Qwen3 completions (tests/fixtures/query_slots_recorded_s9.json) through parse + validate + compose."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = json.loads(_RECORDED.read_text(encoding="utf-8"))["cases"]

    def test_replay_matches_recording(self) -> None:
        self.assertGreaterEqual(len(self.cases), 39)
        for case in self.cases:
            with self.subTest(case=case["id"]):
                out = extract_query_slots(case["prompt"], case["history"], _ReplayClient(case["raw"]), timeout_sec=5)
                exp = case["expect"]
                self.assertEqual(out.status, exp["status"])
                self.assertEqual(out.subject, exp["subject"])
                self.assertEqual(out.facet, exp["facet"])
                self.assertEqual(out.composed_query, exp["composed_query"])

    def test_known_failure_modes_are_caught_by_validation(self) -> None:
        by_id = {c["id"]: c for c in self.cases}
        # The original ANT failure: the model names the facet as the subject.
        self.assertEqual(by_id["mt_ant_followup_live"]["expect"]["status"], "rejected")
        self.assertEqual(by_id["fs_statement_kubernetes"]["expect"]["reason"], "unanchored_proper_noun")


class SettingsTests(unittest.TestCase):
    def test_defaults(self) -> None:
        from core import app_settings

        store = SimpleNamespace(get=lambda _k, default=None: default)
        with patch.object(app_settings, "_store", return_value=store), \
                patch.object(app_settings, "get_sidecar_enabled", return_value=True):
            self.assertFalse(app_settings.get_sidecar_query_rewrite_enabled())
            self.assertTrue(app_settings.get_sidecar_query_slots_shadow_enabled())
            self.assertEqual(app_settings.get_sidecar_query_slots_timeout_ms(), 8000)
        with patch.object(app_settings, "get_sidecar_enabled", return_value=False):
            self.assertFalse(app_settings.get_sidecar_query_slots_shadow_enabled())

    def test_timeout_clamped(self) -> None:
        from core import app_settings

        for raw, expected in ((10, 1000), (99999, 30000), ("x", 8000)):
            store = SimpleNamespace(get=lambda _k, default=None, _v=raw: _v)
            with self.subTest(raw=raw), patch.object(app_settings, "_store", return_value=store):
                self.assertEqual(app_settings.get_sidecar_query_slots_timeout_ms(), expected)

    def test_rewrite_can_be_turned_back_on(self) -> None:
        from core import app_settings

        store = SimpleNamespace(get=lambda k, default=None: True if k == app_settings.KEY_SIDECAR_QUERY_REWRITE else default)
        with patch.object(app_settings, "_store", return_value=store), \
                patch.object(app_settings, "get_sidecar_enabled", return_value=True):
            self.assertTrue(app_settings.get_sidecar_query_rewrite_enabled())


class TelemetryAndRoutingDebugTests(unittest.TestCase):
    def test_queue_depth_accessor(self) -> None:
        from core.sidecar_telemetry import SidecarTelemetryBrain

        brain = SidecarTelemetryBrain()
        self.assertEqual(brain.queue_depth(), 0)
        brain.set_queue_depth(3)
        self.assertEqual(brain.queue_depth(), 3)


if __name__ == "__main__":
    unittest.main()
