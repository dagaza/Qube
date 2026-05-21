from __future__ import annotations

import threading
import unittest
from unittest.mock import patch
import importlib.util
import pathlib
import sys
import types

from core.model_chat_contract import ChatContract
from core.prompt_template_router import RenderPromptBundle
from core.prompt_contract import PromptContract, PromptContractResolution, stops_for_format


def _fake_build_prompt_bundle(
    prompt: str = "<<stub>>",
    *,
    stop_tokens: list[str] | None = None,
):
    st = stop_tokens if stop_tokens is not None else ["<|im_end|>"]

    def _inner(*_a, **_k):
        return (
            RenderPromptBundle(
                prompt=prompt,
                chat_format="chatml",
                stop_tokens=list(st),
                template_type="chatml",
                reasoning_mode="disabled",
            ),
            "stub",
            None,
        )

    return _inner


def _load_native_engine_class():
    if "PyQt6" not in sys.modules:
        pyqt6_mod = types.ModuleType("PyQt6")
        qtcore_mod = types.ModuleType("PyQt6.QtCore")

        class _QThread:
            def __init__(self, *args, **kwargs):
                pass

        class _QSettings:
            def __init__(self, *args, **kwargs):
                self._store = {}

            def value(self, key, default=None, type=None):
                val = self._store.get(key, default)
                if type is not None and val is not None:
                    try:
                        return type(val)
                    except Exception:
                        return default
                return val

            def setValue(self, key, value):
                self._store[key] = value

            def contains(self, key):
                return key in self._store

            def remove(self, key):
                self._store.pop(key, None)

            def sync(self):
                return None

        def _signal(*_args, **_kwargs):
            class _DummySignal:
                def emit(self, *_a, **_k):
                    return None

            return _DummySignal()

        qtcore_mod.QThread = _QThread
        qtcore_mod.QSettings = _QSettings
        qtcore_mod.pyqtSignal = _signal
        pyqt6_mod.QtCore = qtcore_mod
        sys.modules["PyQt6"] = pyqt6_mod
        sys.modules["PyQt6.QtCore"] = qtcore_mod

    file_path = pathlib.Path(__file__).resolve().parents[1] / "workers" / "native_llama_engine.py"
    spec = importlib.util.spec_from_file_location("native_llama_engine_test_module", file_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["native_llama_engine_test_module"] = mod
    spec.loader.exec_module(mod)
    return mod.NativeLlamaEngine


NativeLlamaEngine = _load_native_engine_class()
_NATIVE_ENG_MOD = sys.modules["native_llama_engine_test_module"]


class _FakeTokenizerModel:
    def token_get_text(self, _tid: int) -> str:
        return ""


class _FakeLlama:
    def __init__(self) -> None:
        self.metadata = {"general.name": "test-model"}
        self._chat_handlers = {"chatml": object()}
        self.chat_format = "chatml"
        self.model_path = "/tmp/test-model.gguf"
        self._model = _FakeTokenizerModel()
        self.calls: list[tuple[str, dict]] = []

    def token_eos(self) -> int:
        return 2

    def token_bos(self) -> int:
        return 1

    def create_completion(self, **kwargs):
        self.calls.append(("completion", kwargs))
        return {"choices": [{"text": "ok"}]}


class _FakeLlamaLockSkew(_FakeLlama):
    """Handlers include GGUF default so a locked contract can map to chat_template.default."""

    def __init__(self) -> None:
        super().__init__()
        self._chat_handlers = {"chatml": object(), "chat_template.default": object()}
        self.metadata = {
            "general.name": "oss-model",
            "tokenizer.chat_template": "<|channel|> evil jinja",
        }


class TestNativePromptContractRuntime(unittest.TestCase):
    def test_chat_once_messages_mode_uses_completion_only(self) -> None:
        eng = NativeLlamaEngine()
        eng._llama = _FakeLlama()
        out: list[str] = []
        ev = threading.Event()
        cmd = {
            "out": out,
            "done_event": ev,
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.1,
            "max_tokens": 32,
        }
        with patch(
            "native_llama_engine_test_module.build_prompt_bundle",
            side_effect=_fake_build_prompt_bundle("<<stub prompt>>"),
        ):
            eng._do_chat_once(cmd)
        self.assertTrue(ev.is_set())
        self.assertEqual(out, ["ok"])
        self.assertEqual(len(eng._llama.calls), 1)
        kind, kwargs = eng._llama.calls[0]
        self.assertEqual(kind, "completion")
        self.assertEqual(kwargs.get("prompt"), "<<stub prompt>>")
        self.assertNotIn("messages", kwargs)

    def test_chat_once_rendered_mode_uses_completion_only(self) -> None:
        eng = NativeLlamaEngine()
        eng._llama = _FakeLlama()
        out: list[str] = []
        ev = threading.Event()
        cmd = {
            "out": out,
            "done_event": ev,
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.1,
            "max_tokens": 32,
        }
        rendered = PromptContract(
            mode="rendered",
            chat_format="chatml",
            prompt="USER: hello\nASSISTANT:",
            messages=None,
            stop=["<|im_end|>"],
            template_source="fallback",
            confidence="low",
        )
        with patch(
            "native_llama_engine_test_module.resolve_prompt_contract",
            return_value=PromptContractResolution(contract=rendered),
        ):
            eng._do_chat_once(cmd)
        self.assertTrue(ev.is_set())
        self.assertEqual(out, ["ok"])
        self.assertEqual(len(eng._llama.calls), 1)
        kind, kwargs = eng._llama.calls[0]
        self.assertEqual(kind, "completion")
        self.assertIn("prompt", kwargs)
        self.assertNotIn("messages", kwargs)

    def test_chat_once_applies_retry_output_when_retry_used(self) -> None:
        eng = NativeLlamaEngine()
        eng._llama = _FakeLlama()
        out: list[str] = []
        ev = threading.Event()
        cmd = {
            "out": out,
            "done_event": ev,
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.1,
            "max_tokens": 32,
        }
        base = PromptContract(
            mode="messages",
            chat_format="chatml",
            prompt=None,
            messages=[{"role": "user", "content": "Hello"}],
            stop=["<|im_end|>"],
            template_source="override",
            confidence="medium",
        )
        retried = PromptContract(
            mode="rendered",
            chat_format=None,
            prompt="### Instruction:\nHello\n\n### Response:\n",
            messages=None,
            stop=[],
            template_source="fallback",
            confidence="low",
        )
        with patch(
            "native_llama_engine_test_module.resolve_prompt_contract",
            return_value=PromptContractResolution(contract=base),
        ), patch(
            "native_llama_engine_test_module.maybe_retry",
            return_value=("retry answer", retried, True),
        ), patch(
            "native_llama_engine_test_module.build_prompt_bundle",
            side_effect=_fake_build_prompt_bundle("<<stub>>"),
        ):
            eng._do_chat_once(cmd)
        self.assertTrue(ev.is_set())
        self.assertEqual(out, ["retry answer"])
        self.assertEqual(eng._last_prompt_contract, retried)

    def test_chat_once_runs_response_quality_advisory(self) -> None:
        eng = NativeLlamaEngine()
        eng._llama = _FakeLlama()
        out: list[str] = []
        ev = threading.Event()
        cmd = {
            "out": out,
            "done_event": ev,
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.1,
            "max_tokens": 32,
        }
        with patch("native_llama_engine_test_module.evaluate_response_quality") as mock_quality, patch(
            "native_llama_engine_test_module.build_prompt_bundle",
            side_effect=_fake_build_prompt_bundle("<<stub>>"),
        ):
            mock_quality.return_value = type(
                "_Q",
                (),
                {"score": 0.9, "issues": [], "confidence": "high", "reasoning": None},
            )()
            eng._do_chat_once(cmd)
        self.assertTrue(ev.is_set())
        self.assertEqual(out, ["ok"])
        self.assertTrue(mock_quality.called)

    def test_model_router_runs_before_prompt_contract(self) -> None:
        from core.model_router import RoutingDecision
        from core.prompt_contract import PromptContractResolution

        call_order: list[str] = []

        def _track_route(*_a, **_k):
            call_order.append("route")
            return RoutingDecision(
                selected_model="router-pick",
                confidence=0.9,
                reasoning=["test"],
                task="general_chat",
                scores={"router-pick": 1.0},
            )

        base = PromptContract(
            mode="messages",
            chat_format="chatml",
            prompt=None,
            messages=[{"role": "user", "content": "Hello"}],
            stop=["<|im_end|>"],
            template_source="override",
            confidence="medium",
        )

        def _track_resolve(*_a, **_k):
            call_order.append("resolve")
            return PromptContractResolution(contract=base)

        eng = NativeLlamaEngine()
        eng._llama = _FakeLlama()
        eng._model_path = "/tmp/registry-model.gguf"
        out: list[str] = []
        ev = threading.Event()
        cmd = {
            "out": out,
            "done_event": ev,
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.1,
            "max_tokens": 32,
        }
        with patch.object(_NATIVE_ENG_MOD, "route_model", side_effect=_track_route), patch.object(
            _NATIVE_ENG_MOD, "resolve_prompt_contract", side_effect=_track_resolve
        ), patch(
            "native_llama_engine_test_module.build_prompt_bundle",
            side_effect=_fake_build_prompt_bundle("<<stub>>"),
        ):
            eng._do_chat_once(cmd)
        self.assertTrue(ev.is_set())
        self.assertEqual(call_order, ["route", "resolve"])
        self.assertIsNotNone(eng._last_router_decision)
        assert eng._last_router_decision is not None
        self.assertEqual(eng._last_router_decision.selected_model, "router-pick")

    def test_prepare_skips_chat_lock_for_unsafe_gguf_contract(self) -> None:
        """Locked chat_format must not override PromptContract when template is unsafe."""
        from types import SimpleNamespace

        unsafe = PromptContract(
            mode="messages",
            chat_format="chatml",
            prompt=None,
            messages=[{"role": "user", "content": "Hi"}],
            stop=stops_for_format("chatml"),
            template_source="fallback_unsafe_gguf",
            confidence="medium",
        )
        res = PromptContractResolution(
            contract=unsafe,
            handler_available=True,
            template_safety={"unsafe": True, "reasons": ["contains <|channel|>"]},
        )
        eng = NativeLlamaEngine()
        eng._llama = _FakeLlamaLockSkew()
        eng._model_path = "/tmp/oss.gguf"
        eng._chat_contract = ChatContract(
            model_name="oss",
            format_name="chat_template.default",
            tokenizer_template="x",
            source="gguf",
            locked=True,
        )
        pv = SimpleNamespace(assistant_anchor_present=True)
        with patch.object(
            _NATIVE_ENG_MOD, "route_model", side_effect=RuntimeError("skip router")
        ), patch.object(_NATIVE_ENG_MOD, "resolve_prompt_contract", return_value=res), patch(
            "native_llama_engine_test_module.build_prompt_bundle",
            side_effect=_fake_build_prompt_bundle(
                "<<stub>>", stop_tokens=stops_for_format("chatml")
            ),
        ), patch(
            "native_llama_engine_test_module.validate_chat_inference", return_value=pv
        ), patch("native_llama_engine_test_module.log_prompt_validation_jsonlines"), patch(
            "native_llama_engine_test_module.log_native_inference_request"
        ), patch(
            "native_llama_engine_test_module.load_lm_studio_reference_from_env",
            return_value=None,
        ), self.assertLogs("Qube.NativeLLM", level="INFO") as log_cm:
            out_contract, _cc_kw = eng._prepare_validation_and_logs(
                [{"role": "user", "content": "Hi"}],
                temperature=0.2,
                max_tokens=32,
                stream=False,
            )
        self.assertEqual(out_contract.chat_format, "chatml")
        self.assertNotEqual(out_contract.chat_format, "chat_template.default")
        self.assertEqual(out_contract.template_source, "fallback_unsafe_gguf")
        self.assertEqual(getattr(eng._llama, "chat_format", None), "chatml")
        self.assertTrue(eng._last_chat_contract_lock_skipped)
        msgs = " ".join(r.getMessage() for r in log_cm.records)
        self.assertIn("[PromptContractLock]", msgs)
        self.assertIn("after_lock_block", msgs)
        self.assertIn("per_request_lock_skipped=True", msgs)
