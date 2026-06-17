"""Tests for chat exchange / inference scope markers in llm_debug.log."""
from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from core.llm_debug_markers import (
    log_chat_exchange_begin,
    log_chat_exchange_end,
    log_engine_job_timing,
    log_inference_scope_begin,
    log_inference_scope_end,
    log_inference_token_begin,
    log_inference_token_end,
    next_exchange_id,
)


class LlmDebugMarkersTests(unittest.TestCase):
    def test_next_exchange_id_increments(self) -> None:
        a = next_exchange_id()
        b = next_exchange_id()
        self.assertGreater(b, a)

    def test_exchange_begin_emits_json_and_banner(self) -> None:
        with patch.dict(os.environ, {"QUBE_LLM_DEBUG": "1"}, clear=False):
            with self.assertLogs("Qube.NativeLLM.Debug", level="INFO") as captured:
                log_chat_exchange_begin(
                    exchange_id=7,
                    session_id="sess-abc",
                    user_prompt="What is the capital of France?",
                    engine_mode="internal",
                )
        messages = [r.getMessage() for r in captured.records]
        json_line = next(m for m in messages if m.startswith("{"))
        payload = json.loads(json_line)
        self.assertEqual(payload["event"], "llm_debug_exchange_begin")
        self.assertEqual(payload["exchange_id"], 7)
        self.assertIn("France", payload["user_prompt_preview"])
        self.assertTrue(any("[QUBE CHAT EXCHANGE BEGIN] id=7" in m for m in messages))

    def test_exchange_end_emits_json_and_banner(self) -> None:
        with patch.dict(os.environ, {"QUBE_LLM_DEBUG": "1"}, clear=False):
            with self.assertLogs("Qube.NativeLLM.Debug", level="INFO") as captured:
                log_chat_exchange_end(
                    exchange_id=3,
                    session_id="sess-abc",
                    route="RAG",
                    success=True,
                    presented_text="Paris is the capital.",
                )
        messages = [r.getMessage() for r in captured.records]
        json_line = next(m for m in messages if m.startswith("{"))
        payload = json.loads(json_line)
        self.assertEqual(payload["event"], "llm_debug_exchange_end")
        self.assertEqual(payload["route"], "RAG")
        self.assertTrue(payload["success"])
        self.assertTrue(any("[QUBE CHAT EXCHANGE END] id=3" in m for m in messages))

    def test_inference_scope_markers(self) -> None:
        with patch.dict(os.environ, {"QUBE_LLM_DEBUG": "1"}, clear=False):
            with self.assertLogs("Qube.NativeLLM.Debug", level="INFO") as captured:
                log_inference_scope_begin(
                    caller="chat", exchange_id=2, stream=True
                )
                log_inference_scope_end(caller="chat", exchange_id=2)
        messages = [r.getMessage() for r in captured.records]
        self.assertTrue(any("[QUBE INFERENCE BEGIN] caller=chat exchange=2" in m for m in messages))
        self.assertTrue(any("[QUBE INFERENCE END] caller=chat exchange=2" in m for m in messages))

    def test_exchange_end_includes_timing_fields(self) -> None:
        with patch.dict(os.environ, {"QUBE_LLM_DEBUG": "1"}, clear=False):
            with self.assertLogs("Qube.NativeLLM.Debug", level="INFO") as captured:
                log_chat_exchange_end(
                    exchange_id=9,
                    session_id="sess",
                    route="NONE",
                    success=True,
                    presented_text="Paris.",
                    worker_prep_ms=1200,
                    engine_queue_wait_ms=500,
                    engine_inference_ms=2800,
                    exchange_total_ms=4500,
                )
        messages = [r.getMessage() for r in captured.records]
        payload = json.loads(next(m for m in messages if m.startswith("{")))
        self.assertEqual(payload["worker_prep_ms"], 1200)
        self.assertEqual(payload["engine_queue_wait_ms"], 500)
        self.assertEqual(payload["engine_inference_ms"], 2800)
        self.assertEqual(payload["exchange_total_ms"], 4500)

    def test_engine_job_timing_event(self) -> None:
        with patch.dict(os.environ, {"QUBE_LLM_DEBUG": "1"}, clear=False):
            with self.assertLogs("Qube.NativeLLM.Debug", level="INFO") as captured:
                log_engine_job_timing(
                    {
                        "task_type": "chat",
                        "queue_wait_ms": 100,
                        "inference_ms": 50,
                        "total_ms": 200,
                    }
                )
        payload = json.loads(captured.records[0].getMessage())
        self.assertEqual(payload["event"], "llm_engine_job_timing")
        self.assertEqual(payload["queue_wait_ms"], 100)

    def test_inference_token_markers(self) -> None:
        with patch.dict(os.environ, {"QUBE_LLM_DEBUG": "1"}, clear=False):
            with self.assertLogs("Qube.NativeLLM.Debug", level="INFO") as captured:
                log_inference_token_begin(caller="chat", exchange_id=4, stream=True)
                log_inference_token_end(caller="chat", exchange_id=4)
        messages = [r.getMessage() for r in captured.records]
        self.assertTrue(any("[QUBE INFERENCE TOKEN BEGIN]" in m for m in messages))
        self.assertTrue(any("[QUBE INFERENCE TOKEN END]" in m for m in messages))

    def test_exchange_begin_includes_execution_policy_fields(self) -> None:
        with patch.dict(os.environ, {"QUBE_LLM_DEBUG": "1"}, clear=False):
            with self.assertLogs("Qube.NativeLLM.Debug", level="INFO") as captured:
                log_chat_exchange_begin(
                    exchange_id=8,
                    session_id="sess",
                    user_prompt="hi",
                    engine_mode="internal",
                    execution_policy={
                        "allow_thinking_tokens": False,
                        "strip_thinking_output": True,
                        "reasoning_mode": "disabled",
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
        payload = json.loads(
            next(r.getMessage() for r in captured.records if r.getMessage().startswith("{"))
        )
        self.assertFalse(payload["allow_thinking_tokens"])
        self.assertEqual(payload["reasoning_mode"], "disabled")
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})

    def test_noop_when_debug_disabled(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with patch("core.llm_debug_markers.logger") as mock_logger:
                log_chat_exchange_begin(
                    exchange_id=1,
                    session_id="x",
                    user_prompt="hi",
                )
                log_inference_scope_begin(caller="chat")
        mock_logger.info.assert_not_called()


if __name__ == "__main__":
    unittest.main()
