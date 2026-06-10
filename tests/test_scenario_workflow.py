"""Tests for guided scenario comparison workflow helpers."""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.conversation_replay import ReplayMessage, Scenario
from core.scenario_workflow import (
    build_external_replay_command,
    models_url_from_chat_completions,
    qube_pathway_ready,
    suggested_external_model_name,
)


class ScenarioWorkflowHelperTests(unittest.TestCase):
    def test_models_url_from_chat_completions(self) -> None:
        self.assertEqual(
            models_url_from_chat_completions("http://localhost:1234/v1/chat/completions"),
            "http://localhost:1234/v1/models",
        )

    def test_qube_pathway_ready_requires_loaded_native_model(self) -> None:
        window = MagicMock()
        window._native_model_loading = False
        window._native_model_unloading = False
        window._native_model_loaded_success = False
        window._native_engine = MagicMock()
        window._native_engine.get_model_reasoning_telemetry.return_value = {"loaded": False}

        with patch("core.app_settings.get_engine_mode", return_value="internal"):
            ready, message = qube_pathway_ready(window)

        self.assertFalse(ready)
        self.assertIn("Load a GGUF model", message)

    def test_qube_pathway_ready_when_model_loaded(self) -> None:
        window = MagicMock()
        window._native_model_loading = False
        window._native_model_unloading = False
        window._native_model_loaded_success = True
        window._native_engine = MagicMock()
        window._native_engine.get_model_reasoning_telemetry.return_value = {"loaded": True}

        with patch("core.app_settings.get_engine_mode", return_value="internal"):
            ready, message = qube_pathway_ready(window)

        self.assertTrue(ready)
        self.assertEqual(message, "")

    def test_build_external_replay_command(self) -> None:
        cmd = build_external_replay_command(
            "test_scenarios/demo.json",
            repo_root=Path("/repo"),
            model="gpt-oss-20b",
            api_url="http://localhost:1234/v1/chat/completions",
            qube_session_path="/repo/debug/replay_traces/demo_qube.json",
            wait_for_api_seconds=120,
        )
        self.assertIn("tools.run_scenario_replay", cmd)
        self.assertIn("--wait-for-api", cmd)
        self.assertIn("--compare-with", cmd)
        self.assertIn("gpt-oss-20b", cmd)

    def test_suggested_external_model_prefers_scenario_model(self) -> None:
        scenario = Scenario(
            messages=[ReplayMessage("user", "hi")],
            model="custom-model",
        )
        self.assertEqual(
            suggested_external_model_name(MagicMock(), scenario),
            "custom-model",
        )


if __name__ == "__main__":
    unittest.main()
