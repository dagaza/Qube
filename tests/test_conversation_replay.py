"""Tests for sequential conversation replay and canonical trace capture."""
from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from core.conversation_replay import (
    ConversationReplayEngine,
    EXECUTION_PATH_EXTERNAL_HTTP,
    EXECUTION_PATH_QUBE_NATIVE,
    ReplayMessage,
    Scenario,
    TurnTrace,
    build_input_state,
    build_replay_input_state,
    diff_turn_traces,
    infer_execution_path_from_turn,
    qube_execution_path_for_engine_mode,
    scenario_from_dict,
    scenario_user_messages,
    session_execution_path,
    user_turn_indices,
)


class ConversationReplayHelpersTests(unittest.TestCase):
    def test_user_turn_indices_skips_empty_user(self) -> None:
        messages = [
            ReplayMessage("user", "hello"),
            ReplayMessage("assistant", "hi"),
            ReplayMessage("user", "   "),
            ReplayMessage("user", "bye"),
        ]
        self.assertEqual(user_turn_indices(messages), [0, 3])

    def test_build_input_state_includes_current_user(self) -> None:
        messages = [
            ReplayMessage("user", "one"),
            ReplayMessage("assistant", "two"),
            ReplayMessage("user", "three"),
        ]
        state = build_input_state(messages, up_to_index=2)
        self.assertEqual(len(state), 3)
        self.assertEqual(state[-1]["content"], "three")

    def test_build_replay_input_state_uses_prior_outputs(self) -> None:
        users = [
            ReplayMessage("user", "one"),
            ReplayMessage("user", "two"),
            ReplayMessage("user", "three"),
        ]
        state = build_replay_input_state(
            users,
            turn_index=2,
            prior_outputs=["answer one", "answer two"],
        )
        self.assertEqual(
            state,
            [
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "answer one"},
                {"role": "user", "content": "two"},
                {"role": "assistant", "content": "answer two"},
                {"role": "user", "content": "three"},
            ],
        )

    def test_scenario_user_messages_ignores_scripted_assistant(self) -> None:
        messages = [
            ReplayMessage("user", "a"),
            ReplayMessage("assistant", "fixture should be ignored"),
            ReplayMessage("user", "b"),
        ]
        self.assertEqual(len(scenario_user_messages(messages)), 2)

    def test_scenario_from_dict(self) -> None:
        scenario = scenario_from_dict(
            {
                "name": "demo",
                "messages": [{"role": "user", "content": "ping"}],
                "backend": "external",
            }
        )
        self.assertEqual(scenario.name, "demo")
        self.assertEqual(scenario.messages[0].content, "ping")
        self.assertEqual(scenario.backend, "external")


class ConversationReplayExternalTests(unittest.TestCase):
    @patch("core.conversation_replay.requests.post")
    def test_external_replay_captures_turn_traces(self, mock_post) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "assistant reply"}}]
        }
        mock_post.return_value = mock_resp

        scenario = Scenario(
            name="parity",
            messages=[
                ReplayMessage("user", "first"),
                ReplayMessage("assistant", "expected"),
                ReplayMessage("user", "second"),
            ],
            backend="external",
            model="test-model",
            external_api_url="http://127.0.0.1:1234/v1/chat/completions",
        )
        engine = ConversationReplayEngine()
        traces = engine.replay(scenario, backend="external")

        self.assertEqual(len(traces), 2)
        self.assertEqual(traces[0].turn_index, 0)
        self.assertEqual(traces[0].user_message, "first")
        self.assertEqual(traces[0].backend_used, "external")
        self.assertEqual(traces[0].output, "assistant reply")
        self.assertEqual(len(traces[0].input_state), 1)
        self.assertTrue(traces[0].prompt)
        self.assertEqual(traces[0].trace.output, "assistant reply")

        self.assertEqual(traces[1].turn_index, 1)
        self.assertEqual(traces[1].user_message, "second")
        self.assertEqual(len(traces[1].input_state), 3)
        payload = mock_post.call_args_list[1].kwargs["json"]
        self.assertEqual(len(payload["messages"]), 3)
        self.assertEqual(payload["messages"][1]["role"], "assistant")
        self.assertEqual(payload["messages"][1]["content"], "assistant reply")
        self.assertEqual(payload["messages"][-1]["content"], "second")

    @patch("core.conversation_replay.requests.post")
    def test_external_prompt_is_lm_studio_serialized_body(self, mock_post) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_post.return_value = mock_resp

        scenario = Scenario(
            messages=[ReplayMessage("user", "hello")],
            backend="external",
            model="m",
        )
        traces = ConversationReplayEngine().replay(scenario, backend="external")
        parsed = json.loads(traces[0].prompt)
        self.assertEqual(parsed["model"], "m")
        self.assertEqual(parsed["messages"][0]["content"], "hello")


class ConversationReplayDiffTests(unittest.TestCase):
    def test_diff_turn_traces_match(self) -> None:
        from core.canonical_request import CanonicalMessage, CanonicalRequest, CanonicalSampling
        from core.golden_trace_capture import build_golden_trace

        canonical = build_golden_trace(
            request=CanonicalRequest(
                model="m",
                messages=[CanonicalMessage(role="user", content="hi")],
                sampling=CanonicalSampling(),
            ),
            prompt="p",
            output="o",
        )
        trace_a = TurnTrace(
            turn_index=0,
            user_message="hi",
            input_state=[],
            prompt="p",
            output="o",
            backend_used="qube",
            trace=canonical,
        )
        trace_b = TurnTrace(
            turn_index=0,
            user_message="hi",
            input_state=[],
            prompt="p",
            output="o",
            backend_used="external",
            trace=canonical,
        )
        diff = diff_turn_traces(trace_a, trace_b)
        self.assertIsNone(diff.first_divergence)
        self.assertTrue(diff.output_match)


class ConversationReplayQubeTests(unittest.TestCase):
    def test_qube_requires_worker_and_db(self) -> None:
        scenario = Scenario(
            messages=[ReplayMessage("user", "hi")],
            backend="qube",
        )
        engine = ConversationReplayEngine()
        with self.assertRaises(ValueError):
            engine.replay(scenario, backend="qube")

    def test_qube_replay_uses_worker_trace_capture(self) -> None:
        from core.canonical_request import CanonicalMessage, CanonicalRequest, CanonicalSampling
        from core.golden_trace_capture import build_golden_trace

        db = MagicMock()
        db.create_session.return_value = "sess-1"

        worker = MagicMock()
        worker.engine_mode = "internal"
        captured: dict[str, str] = {}

        def _generate(text: str, session_id: str) -> None:
            captured["text"] = text
            captured["session_id"] = session_id
            worker._turn_engine_request = {"messages": [{"role": "user", "content": text}]}
            worker._turn_rendered_prompt = f"rendered:{text}"
            worker.build_last_turn_canonical_trace.return_value = build_golden_trace(
                request=CanonicalRequest(
                    model="demo.gguf",
                    messages=[CanonicalMessage(role="user", content=text)],
                    sampling=CanonicalSampling(),
                ),
                prompt=f"rendered:{text}",
                output="model answer",
            )
            for cb in worker.response_finished._callbacks:
                cb(session_id, "model answer")

        worker.generate_response.side_effect = _generate
        worker.response_finished = MagicMock()
        worker.response_finished.connect.side_effect = (
            lambda cb: worker.response_finished._callbacks.append(cb)
        )
        worker.response_finished._callbacks = []
        worker.response_finished.disconnect.side_effect = lambda _cb: None

        scenario = Scenario(
            messages=[
                ReplayMessage("user", "hello"),
                ReplayMessage("assistant", "prior"),
                ReplayMessage("user", "follow up"),
            ],
            backend="qube",
        )
        traces = ConversationReplayEngine(llm_worker=worker, db_manager=db).replay(
            scenario, backend="qube"
        )

        self.assertEqual(len(traces), 2)
        self.assertEqual(traces[0].backend_used, "qube")
        self.assertEqual(traces[0].execution_path, EXECUTION_PATH_QUBE_NATIVE)
        self.assertEqual(traces[0].output, "model answer")
        self.assertEqual(traces[0].prompt, "rendered:hello")
        self.assertEqual(traces[1].user_message, "follow up")
        self.assertEqual(db.add_message.call_count, 2)
        db.add_message.assert_any_call("sess-1", "user", "hello")
        db.add_message.assert_any_call("sess-1", "assistant", "model answer")


class ExecutionPathTests(unittest.TestCase):
    def test_qube_execution_path_for_engine_mode(self) -> None:
        self.assertEqual(qube_execution_path_for_engine_mode("internal"), EXECUTION_PATH_QUBE_NATIVE)
        self.assertEqual(
            qube_execution_path_for_engine_mode("external"),
            "qube_external_http",
        )

    def test_infer_harmony_prompt_as_qube_native(self) -> None:
        from core.canonical_request import CanonicalMessage, CanonicalRequest, CanonicalSampling
        from core.golden_trace_capture import build_golden_trace

        trace = build_golden_trace(
            request=CanonicalRequest(
                model="m",
                messages=[CanonicalMessage(role="user", content="hi")],
                sampling=CanonicalSampling(),
            ),
            prompt="<|start|>system<|message|>You are helpful",
            output="ok",
        )
        turn = TurnTrace(
            turn_index=0,
            user_message="hi",
            input_state=[],
            prompt=trace.prompt,
            output="ok",
            backend_used="qube",
            trace=trace,
        )
        self.assertEqual(infer_execution_path_from_turn(turn), EXECUTION_PATH_QUBE_NATIVE)
        self.assertEqual(
            session_execution_path(backend="qube", traces=[turn]),
            EXECUTION_PATH_QUBE_NATIVE,
        )

    def test_external_backend_maps_to_external_http(self) -> None:
        from core.golden_trace_capture import build_golden_trace

        turn = TurnTrace(
            turn_index=0,
            user_message="hi",
            input_state=[],
            prompt="p",
            output="o",
            backend_used="external",
            trace=build_golden_trace(
                request={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                prompt="p",
                output="o",
            ),
        )
        self.assertEqual(infer_execution_path_from_turn(turn), EXECUTION_PATH_EXTERNAL_HTTP)
        self.assertEqual(
            session_execution_path(backend="external", traces=[turn]),
            EXECUTION_PATH_EXTERNAL_HTTP,
        )


if __name__ == "__main__":
    unittest.main()
