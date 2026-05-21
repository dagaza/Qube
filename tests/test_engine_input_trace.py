"""Tests for core/engine_input_trace capture wrapper."""

from __future__ import annotations

import unittest
from typing import Any

from core.engine_input_trace import (
    EngineInputTrace,
    EngineInputTracer,
    install_create_completion_capture,
)


def _reset_tracer() -> None:
    EngineInputTracer._instance = None  # type: ignore[attr-defined]


class _FakeLlama:
    """Minimal stand-in for llama-cpp-python ``Llama`` completion boundary."""

    def create_completion(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"choices": [{"text": "x"}]}

    def create_chat_completion(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.create_completion(
            prompt=[10, 20, 30],
            temperature=0.0,
            max_tokens=1,
            stream=False,
            echo=False,
            stop=[],
        )


class _FakeLlamaPositional(_FakeLlama):
    def create_chat_completion(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.create_completion([7, 8], temperature=0.0, max_tokens=1, stream=False, echo=False, stop=[])


class _FakeLlamaExploding(_FakeLlama):
    def create_chat_completion(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.create_completion(
            prompt=[1],
            temperature=0.0,
            max_tokens=1,
            stream=False,
            echo=False,
            stop=[],
        )
        raise RuntimeError("boom")


class InstallCreateCompletionCaptureTests(unittest.TestCase):
    def tearDown(self) -> None:
        _reset_tracer()

    def test_capture_prompt_token_ids(self) -> None:
        llama = _FakeLlama()
        with install_create_completion_capture(llama) as cap:
            llama.create_chat_completion(messages=[{"role": "user", "content": "hi"}])
        self.assertTrue(cap.get("captured"))
        self.assertEqual(cap.get("prompt"), [10, 20, 30])
        self.assertEqual(getattr(llama.create_completion, "__name__", ""), "create_completion")

    def test_restore_on_exception(self) -> None:
        llama = _FakeLlamaExploding()
        with self.assertRaises(RuntimeError):
            with install_create_completion_capture(llama):
                llama.create_chat_completion(messages=[])
        self.assertEqual(getattr(llama.create_completion, "__name__", ""), "create_completion")

    def test_positional_prompt(self) -> None:
        llama = _FakeLlamaPositional()
        with install_create_completion_capture(llama) as cap:
            llama.create_chat_completion(messages=[])
        self.assertEqual(cap.get("prompt"), [7, 8])
        self.assertEqual(getattr(llama.create_completion, "__name__", ""), "create_completion")


class EngineInputTracerTests(unittest.TestCase):
    def tearDown(self) -> None:
        _reset_tracer()

    def test_log_and_get_last(self) -> None:
        t = EngineInputTrace(
            model_name="m.gguf",
            timestamp=1.0,
            input_mode="completion",
            messages=None,
            prompt="x",
            serialized_input="x",
            chat_format=None,
            stop_tokens=[],
            source="llama_cpp_completion",
        )
        tr = EngineInputTracer()
        tr.log(t)
        last = tr.get_last()
        assert last is not None
        self.assertEqual(last.model_name, "m.gguf")
        self.assertGreater(last.trace_id, 0)


if __name__ == "__main__":
    unittest.main()
