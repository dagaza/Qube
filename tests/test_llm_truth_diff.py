"""Tests for opt-in 3-layer LLM truth diff logging."""
from __future__ import annotations

import json
import os
import unittest
from typing import Any
from unittest.mock import patch

from core.llm_truth_diff import (
    LLMTruthDiffLogger,
    bind_llm_worker_truth_diff_hooks,
    clear_llm_worker_truth_diff_hooks,
    emit_l2_prompt,
    llm_truth_diff_enabled,
    llm_truth_diff_max_chars,
)


class LLMTruthDiffTests(unittest.TestCase):
    def test_disabled_by_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(llm_truth_diff_enabled())

    def test_enabled_with_env(self) -> None:
        with patch.dict(os.environ, {"ENABLE_LLM_TRUTH_DIFF_LOGGING": "1"}, clear=False):
            self.assertTrue(llm_truth_diff_enabled())

    def test_default_max_chars(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(llm_truth_diff_max_chars(), 20_000)

    def test_l1_raw_request_payload(self) -> None:
        logger = LLMTruthDiffLogger(enabled=True)
        with patch.object(logger, "_emit") as emit:
            logger.log_l1_raw_request(
                {"prompt": "hello", "session_id": "s1"},
                {"exchange_id": 7, "session_id": "s1", "model_name": "test.gguf"},
            )
        payload = emit.call_args[0][0]
        self.assertEqual(payload["layer"], "L1")
        self.assertEqual(payload["pipeline_stage"], "raw_request")
        self.assertEqual(payload["exchange_id"], 7)
        self.assertEqual(payload["request"]["prompt"], "hello")
        self.assertIn("fingerprint", payload)
        self.assertEqual(len(payload["fingerprint"]["sha256"]), 64)
        self.assertEqual(payload["fingerprint"]["short"], payload["fingerprint"]["sha256"][:12])

    def test_l1_truncates_long_strings_in_request(self) -> None:
        with patch.dict(os.environ, {"LLM_TRUTH_DIFF_MAX_CHARS": "10"}, clear=False):
            logger = LLMTruthDiffLogger(enabled=True)
            with patch.object(logger, "_emit") as emit:
                logger.log_l1_raw_request(
                    {"prompt": "01234567890123456789"},
                    {"exchange_id": 1},
                )
        req = emit.call_args[0][0]["request"]
        self.assertTrue(req["prompt"]["_truncated"])
        self.assertEqual(req["prompt"]["_full_len"], 20)

    def test_l2_prompt_metadata(self) -> None:
        logger = LLMTruthDiffLogger(enabled=True)
        with patch.object(logger, "_emit") as emit:
            logger.log_l2_prompt(
                "system\nuser",
                {
                    "exchange_id": 3,
                    "template_source": "harmony",
                    "chat_format_mode": "structured",
                    "execution_mode": "direct",
                    "prompt_contract_mode": "rendered",
                },
            )
        payload = emit.call_args[0][0]
        self.assertEqual(payload["layer"], "L2")
        self.assertEqual(payload["template_source"], "harmony")
        self.assertEqual(payload["prompt"], "system\nuser")
        self.assertIn("fingerprint", payload)
        self.assertEqual(payload["fingerprint"]["length"], len("system\nuser"))

    def test_l2_truncates_prompt(self) -> None:
        with patch.dict(os.environ, {"LLM_TRUTH_DIFF_MAX_CHARS": "4"}, clear=False):
            logger = LLMTruthDiffLogger(enabled=True)
            with patch.object(logger, "_emit") as emit:
                logger.log_l2_prompt("0123456789", {"exchange_id": 1})
        payload = emit.call_args[0][0]
        self.assertTrue(payload["truncated"])
        self.assertEqual(payload["prompt"], "0123")
        self.assertEqual(payload["prompt_full_len"], 10)

    def test_l3_model_io_stages(self) -> None:
        logger = LLMTruthDiffLogger(enabled=True)
        with patch.object(logger, "_emit") as emit:
            logger.log_l3_model_io(
                raw="raw planning Hello.",
                after_stages=["Hello.", "Hello!"],
                final="Hello!",
                metadata={"exchange_id": 9, "engine_mode": "internal"},
            )
        payload = emit.call_args[0][0]
        self.assertEqual(payload["layer"], "L3")
        self.assertEqual(len(payload["after_stages"]), 2)
        self.assertFalse(payload["raw_equals_final"])
        self.assertEqual(payload["final"], "Hello!")
        self.assertIn("fingerprint", payload)
        self.assertIn("final_fingerprint", payload)
        self.assertIn("fingerprint", payload["after_stages"][0])

    def test_l1_engine_includes_canonical_fingerprint(self) -> None:
        logger = LLMTruthDiffLogger(enabled=True)
        with patch.object(logger, "_emit") as emit:
            logger.log_l1_engine_request(
                {
                    "model": "demo.gguf",
                    "messages": [{"role": "user", "content": "hi"}],
                    "temperature": 0.2,
                },
                {"exchange_id": 1},
            )
        payload = emit.call_args[0][0]
        self.assertIn("fingerprint", payload)
        self.assertIn("fingerprint_raw", payload)

    def test_log_emits_wrapped_json(self) -> None:
        logger = LLMTruthDiffLogger(enabled=True)
        with self.assertLogs("Qube.NativeLLM.Debug", level="INFO") as captured:
            logger.log_l1_engine_request({"messages": []}, {"exchange_id": 2})
        json_line = next(m for m in [r.getMessage() for r in captured.records] if m.startswith("{"))
        parsed = json.loads(json_line)
        self.assertIn("llm_truth_diff", parsed)
        self.assertEqual(parsed["llm_truth_diff"]["pipeline_stage"], "engine_request")
        self.assertIn("fingerprint", parsed["llm_truth_diff"])

    def test_emit_routes_through_worker_hook(self) -> None:
        seen: dict[str, Any] = {}

        def l2_hook(prompt: str, metadata: dict) -> None:
            seen["prompt"] = prompt
            seen["metadata"] = metadata

        bind_llm_worker_truth_diff_hooks(l2_prompt=l2_hook)
        try:
            emit_l2_prompt("hello", {"exchange_id": 5})
        finally:
            clear_llm_worker_truth_diff_hooks()
        self.assertEqual(seen["prompt"], "hello")
        self.assertEqual(seen["metadata"]["exchange_id"], 5)

    def test_emit_falls_back_when_hook_raises(self) -> None:
        def bad_hook(_prompt: str, _metadata: dict) -> None:
            raise RuntimeError("boom")

        bind_llm_worker_truth_diff_hooks(l2_prompt=bad_hook)
        logger = LLMTruthDiffLogger(enabled=True)
        try:
            with patch.object(logger, "_emit") as emit:
                with patch(
                    "core.llm_truth_diff.get_llm_truth_diff_logger", return_value=logger
                ):
                    emit_l2_prompt("fallback", {"exchange_id": 1})
            emit.assert_called_once()
        finally:
            clear_llm_worker_truth_diff_hooks()

    def test_log_never_raises(self) -> None:
        logger = LLMTruthDiffLogger(enabled=True)
        with patch.object(logger, "_emit", side_effect=RuntimeError("fail")):
            logger.log_l3_model_io("raw", [], "final", {})


    def test_noop_when_disabled(self) -> None:
        logger = LLMTruthDiffLogger(enabled=False)
        with patch("core.llm_truth_diff.logger") as mock_logger:
            logger.log_l2_prompt("x", {})
            logger.log_l3_model_io("a", [], "b", {})
        mock_logger.info.assert_not_called()


if __name__ == "__main__":
    unittest.main()
