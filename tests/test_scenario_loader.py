"""Tests for test_scenarios JSON loading, serial replay, and offline compare."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.canonical_request import CanonicalMessage, CanonicalRequest, CanonicalSampling
from core.conversation_replay import ReplayMessage, Scenario, TurnTrace, diff_turn_traces
from core.golden_trace_capture import build_golden_trace
from core.scenario_loader import (
    BackendSession,
    SESSION_SCHEMA,
    compare_sessions,
    first_diverging_turn_index,
    load_backend_session,
    load_scenario,
    load_scenario_run_pair,
    list_scenario_files,
    load_scenario_dict,
    replay_traces_dir,
    run_scenario_serial,
    save_backend_session,
    save_scenario_diff,
    scenario_run_pair_from_dict,
    scenario_run_pair_to_dict,
    session_file_path,
    test_scenarios_dir,
    turn_trace_to_dict,
    validate_scenario_dict,
)


class ScenarioLoaderValidationTests(unittest.TestCase):
    def test_validate_rejects_empty_messages(self) -> None:
        errors = validate_scenario_dict({"name": "x", "messages": []})
        self.assertTrue(any("non-empty" in e for e in errors))

    def test_validate_accepts_minimal_scenario(self) -> None:
        self.assertEqual(
            validate_scenario_dict(
                {"messages": [{"role": "user", "content": "hi"}]}
            ),
            [],
        )


class ScenarioLoaderFileTests(unittest.TestCase):
    def test_nepal_fixture_loads(self) -> None:
        path = test_scenarios_dir() / "nepal_follow_up_chain.json"
        self.assertTrue(path.is_file(), f"missing fixture: {path}")
        scenario = load_scenario(path)
        self.assertEqual(scenario.name, "Nepal follow-up chain")
        self.assertEqual(len(scenario.messages), 11)

    def test_list_scenario_files_includes_fixture(self) -> None:
        files = list_scenario_files()
        names = {p.name for p in files}
        self.assertIn("nepal_follow_up_chain.json", names)


class ScenarioSerialPersistenceTests(unittest.TestCase):
    def _sample_trace(self, *, backend: str = "external", turn_index: int = 0) -> TurnTrace:
        trace = build_golden_trace(
            request=CanonicalRequest(
                model="demo",
                messages=[CanonicalMessage(role="user", content="hi")],
                sampling=CanonicalSampling(),
            ),
            prompt="prompt",
            output="answer",
        )
        return TurnTrace(
            turn_index=turn_index,
            user_message="hi",
            input_state=[{"role": "user", "content": "hi"}],
            prompt="prompt",
            output="answer",
            backend_used=backend,
            trace=trace,
            execution_path="external_http" if backend == "external" else "",
        )

    def test_session_file_path_pattern(self) -> None:
        path = session_file_path("nepal_follow_up_chain", "qube")
        self.assertEqual(path.name, "nepal_follow_up_chain_qube.json")
        self.assertEqual(path.parent, replay_traces_dir())

    def test_save_and_load_backend_session(self) -> None:
        session = BackendSession(
            scenario_id="demo",
            scenario_name="Demo",
            backend="external",
            traces=[self._sample_trace()],
            execution_path="external_http",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = save_backend_session(session, output_dir=tmp)
            self.assertTrue(path.is_file())
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(raw["schema"], SESSION_SCHEMA)
            self.assertEqual(raw["execution_path"], "external_http")
            self.assertEqual(raw["traces"][0]["execution_path"], "external_http")
            loaded = load_backend_session(path)
        self.assertEqual(loaded.backend, "external")
        self.assertEqual(loaded.execution_path, "external_http")
        self.assertEqual(loaded.traces[0].execution_path, "external_http")
        self.assertEqual(len(loaded.traces), 1)

    def test_legacy_session_infers_execution_path(self) -> None:
        trace = self._sample_trace(backend="external")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.json"
            payload = {
                "schema": SESSION_SCHEMA,
                "scenario_id": "demo",
                "scenario_name": "Demo",
                "backend": "external",
                "traces": [turn_trace_to_dict(trace)],
            }
            del payload["traces"][0]["execution_path"]
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_backend_session(path)
        self.assertEqual(loaded.execution_path, "external_http")
        self.assertEqual(loaded.traces[0].execution_path, "external_http")

    def test_turn_trace_serializes_history_output(self) -> None:
        trace = self._sample_trace()
        trace.history_output = "[Previous assistant response suppressed due to degeneration detection]"
        data = turn_trace_to_dict(trace)
        self.assertEqual(data["history_output"], trace.history_output)
        restored = scenario_run_pair_from_dict(
            {
                "scenario_id": "x",
                "scenario_name": "x",
                "backends": ["qube"],
                "runs": {"qube": [data]},
                "diffs": [],
            }
        )
        self.assertEqual(
            restored.runs["qube"][0].history_output,
            trace.history_output,
        )

    def test_compare_sessions_offline(self) -> None:
        base = self._sample_trace(backend="qube")
        ext = self._sample_trace(backend="external")
        ext.trace = build_golden_trace(
            request=ext.trace.request,
            prompt="different prompt",
            output="other answer",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path_a = save_backend_session(
                BackendSession("demo", "Demo", "qube", [base]), output_dir=tmp
            )
            path_b = save_backend_session(
                BackendSession("demo", "Demo", "external", [ext]), output_dir=tmp
            )
            pair = compare_sessions(path_a, path_b, save=True, output_dir=tmp)
            diff_path = Path(tmp) / "demo.json"
            self.assertTrue(diff_path.is_file())
        self.assertEqual(pair.backends, ["qube", "external"])
        self.assertEqual(len(pair.diffs), 1)
        self.assertEqual(pair.diffs[0].first_divergence, "PROMPT")

    def test_first_diverging_turn_index(self) -> None:
        from core.conversation_replay import ScenarioRunPair, TurnPairDiff

        pair = ScenarioRunPair(
            scenario_id="x",
            scenario_name="x",
            backends=["qube", "external"],
            runs={"qube": [], "external": []},
            diffs=[
                TurnPairDiff(
                    turn_index=0,
                    user_message="a",
                    baseline_backend="qube",
                    compare_backend="external",
                    first_divergence=None,
                    diff_summary="match",
                    request_match=True,
                    prompt_match=True,
                    output_match=True,
                ),
                TurnPairDiff(
                    turn_index=1,
                    user_message="b",
                    baseline_backend="qube",
                    compare_backend="external",
                    first_divergence="OUTPUT",
                    diff_summary="output mismatch",
                    request_match=True,
                    prompt_match=True,
                    output_match=False,
                ),
            ],
        )
        self.assertEqual(first_diverging_turn_index(pair), 1)

    def test_legacy_single_backend_log_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.json"
            path.write_text(
                json.dumps(
                    {
                        "scenario": {
                            "name": "Legacy",
                            "messages": [{"role": "user", "content": "hi"}],
                            "backend": "external",
                        },
                        "traces": [turn_trace_to_dict(self._sample_trace())],
                    }
                ),
                encoding="utf-8",
            )
            loaded = load_backend_session(path)
        self.assertEqual(loaded.backend, "external")
        self.assertEqual(len(loaded.traces), 1)

    @patch("core.scenario_loader.save_backend_session")
    def test_run_scenario_serial(self, mock_save) -> None:
        mock_save.return_value = Path("/tmp/demo_external.json")
        engine = MagicMock()
        engine.replay.return_value = [self._sample_trace()]
        scenario = Scenario(messages=[ReplayMessage("user", "hi")], name="Demo")

        result = run_scenario_serial(scenario, "external", engine, log_traces=True)

        engine.replay.assert_called_once()
        mock_save.assert_called_once()
        saved_session = mock_save.call_args[0][0]
        self.assertEqual(saved_session.execution_path, "external_http")
        self.assertEqual(result.backend, "external")


class ScenarioDiffSerializationTests(unittest.TestCase):
    def test_round_trip_dict(self) -> None:
        from core.conversation_replay import ScenarioRunPair, TurnPairDiff

        pair = ScenarioRunPair(
            scenario_id="demo",
            scenario_name="Demo",
            backends=["qube", "external"],
            runs={"qube": [], "external": []},
            diffs=[
                TurnPairDiff(
                    turn_index=0,
                    user_message="hi",
                    baseline_backend="qube",
                    compare_backend="external",
                    first_divergence="REQUEST",
                    diff_summary="request mismatch",
                    request_match=False,
                    prompt_match=True,
                    output_match=True,
                )
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = save_scenario_diff(pair, output_dir=tmp)
            loaded = load_scenario_run_pair(path)
        self.assertEqual(loaded.diffs[0].first_divergence, "REQUEST")


if __name__ == "__main__":
    unittest.main()
