"""Tests for provider-agnostic canonical LLM request export."""
from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from core.canonical_request import (
    CanonicalRequestExporter,
    canonical_trace_export_enabled,
    log_canonical_request_trace,
)
from core.canonical_request_adapters import (
    LMStudioAdapter,
    OpenAICompatAdapter,
    VLLMAdapter,
)


class CanonicalRequestExporterTests(unittest.TestCase):
    def test_normalizes_messages_and_sampling(self) -> None:
        internal = {
            "model": "demo.gguf",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ],
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "presence_penalty": 0.5,
            "stop": ["<|end|>"],
            "stream": True,
            "max_tokens": 512,
        }
        canonical = CanonicalRequestExporter.export_canonical_request(internal)
        self.assertEqual(canonical.model, "demo.gguf")
        self.assertEqual(len(canonical.messages), 2)
        self.assertEqual(canonical.messages[0].role, "system")
        self.assertEqual(canonical.sampling.temperature, 0.2)
        self.assertEqual(canonical.sampling.top_k, 40)
        self.assertEqual(canonical.stop, ["<|end|>"])
        self.assertEqual(canonical.metadata.get("stream"), True)
        self.assertEqual(canonical.metadata.get("max_tokens"), 512)

    def test_prompt_only_completion_goes_to_metadata(self) -> None:
        internal = {
            "prompt": "Hello world",
            "temperature": 0.7,
            "stop": ["END"],
        }
        canonical = CanonicalRequestExporter.export_canonical_request(internal)
        self.assertEqual(canonical.messages, [])
        self.assertEqual(canonical.metadata.get("input_mode"), "completion_prompt")
        self.assertEqual(canonical.metadata.get("prompt"), "Hello world")

    def test_unknown_role_preserved_in_metadata(self) -> None:
        internal = {
            "messages": [{"role": "tool", "content": "{}"}],
        }
        canonical = CanonicalRequestExporter.export_canonical_request(internal)
        self.assertEqual(canonical.messages[0].role, "user")
        notes = canonical.metadata.get("role_normalization") or []
        self.assertEqual(notes[0].get("original_role"), "tool")

    def test_stop_aliases(self) -> None:
        for key in ("stop", "stop_tokens", "stops"):
            canonical = CanonicalRequestExporter.export_canonical_request(
                {key: ["A", "B"]}
            )
            self.assertEqual(canonical.stop, ["A", "B"])


class CanonicalRequestAdaptersTests(unittest.TestCase):
    def _sample(self):
        return CanonicalRequestExporter.export_canonical_request(
            {
                "model": "m1",
                "messages": [{"role": "user", "content": "ping"}],
                "temperature": 0.3,
                "top_p": 0.8,
                "top_k": 10,
                "repeat_penalty": 1.05,
                "stop": ["STOP"],
                "max_tokens": 128,
                "stream": False,
            }
        )

    def test_openai_compat_serialization(self) -> None:
        body = OpenAICompatAdapter.serialize(self._sample())
        self.assertEqual(body["model"], "m1")
        self.assertEqual(body["messages"][0]["role"], "user")
        self.assertEqual(body["temperature"], 0.3)
        self.assertEqual(body["top_k"], 10)
        self.assertEqual(body["stop"], ["STOP"])
        self.assertEqual(body["max_tokens"], 128)

    def test_lmstudio_passthrough_cache_prompt(self) -> None:
        req = self._sample()
        req.metadata["cache_prompt"] = False
        body = LMStudioAdapter.serialize(req)
        self.assertIn("cache_prompt", body)
        self.assertFalse(body["cache_prompt"])

    def test_vllm_passthrough_min_p(self) -> None:
        req = self._sample()
        req.metadata["min_p"] = 0.05
        body = VLLMAdapter.serialize(req)
        self.assertEqual(body["min_p"], 0.05)

    def test_adapters_do_not_invent_transport_fields(self) -> None:
        req = self._sample()
        body = OpenAICompatAdapter.serialize(req)
        self.assertNotIn("cache_prompt", body)
        self.assertNotIn("min_p", body)


class CanonicalRequestTraceTests(unittest.TestCase):
    def test_disabled_by_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(canonical_trace_export_enabled())

    def test_log_emits_wrapped_event(self) -> None:
        with patch.dict(os.environ, {"ENABLE_CANONICAL_TRACE_EXPORT": "1"}, clear=False):
            with self.assertLogs("Qube.NativeLLM.Debug", level="INFO") as captured:
                log_canonical_request_trace(
                    {"messages": [{"role": "user", "content": "hi"}]},
                    context={"session_id": "s1", "exchange_id": 4},
                )
        json_line = next(m for m in [r.getMessage() for r in captured.records] if m.startswith("{"))
        parsed = json.loads(json_line)
        self.assertIn("canonical_request_trace", parsed)
        trace = parsed["canonical_request_trace"]
        self.assertEqual(trace["event"], "canonical_request_trace")
        self.assertEqual(trace["session_id"], "s1")
        self.assertEqual(trace["exchange_id"], 4)

    def test_log_noop_when_disabled(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with patch("core.canonical_request.logger") as mock_logger:
                log_canonical_request_trace({"messages": []})
        mock_logger.info.assert_not_called()

    def test_log_never_raises(self) -> None:
        with patch.dict(os.environ, {"ENABLE_CANONICAL_TRACE_EXPORT": "1"}, clear=False):
            with patch(
                "core.canonical_request.build_canonical_request_trace_payload",
                side_effect=RuntimeError("boom"),
            ):
                log_canonical_request_trace({"messages": []})


if __name__ == "__main__":
    unittest.main()
