"""Tests for boot argument parsing."""

from __future__ import annotations

import unittest

from core.boot_args import parse_boot_args


class ParseBootArgsTests(unittest.TestCase):
    def test_routing_debug_flag_defaults_false(self) -> None:
        args = parse_boot_args([])
        self.assertFalse(args.routing_debug)

    def test_routing_debug_flag_enabled(self) -> None:
        args = parse_boot_args(["--routing-debug"])
        self.assertTrue(args.routing_debug)

    def test_trace_diff_debug_flag_enabled(self) -> None:
        args = parse_boot_args(["--trace-diff-debug"])
        self.assertTrue(args.trace_diff_debug)

    def test_run_scenario_flag(self) -> None:
        args = parse_boot_args(
            ["--run-scenario", "test_scenarios/nepal_follow_up_chain.json"]
        )
        self.assertEqual(args.run_scenario, "test_scenarios/nepal_follow_up_chain.json")

    def test_scenario_backend_flag(self) -> None:
        args = parse_boot_args(["--scenario-backend", "external"])
        self.assertEqual(args.scenario_backend, "external")

    def test_scenario_single_phase_flag(self) -> None:
        args = parse_boot_args(
            ["--run-scenario", "test_scenarios/demo.json", "--scenario-single-phase"]
        )
        self.assertTrue(args.scenario_single_phase)

    def test_compare_sessions_flag(self) -> None:
        args = parse_boot_args(
            [
                "--compare-sessions",
                "debug/replay_traces/a_qube.json",
                "debug/replay_traces/a_external.json",
            ]
        )
        self.assertEqual(len(args.compare_sessions), 2)

    def test_mock_bootstrap_download_flag(self) -> None:
        args = parse_boot_args(["--mock-bootstrap-download"])
        self.assertTrue(args.mock_bootstrap_download)

    def test_winget_validation_flag(self) -> None:
        args = parse_boot_args(["--winget-validation"])
        self.assertTrue(args.winget_validation)

    def test_bootstrap_trace_flag(self) -> None:
        args = parse_boot_args(["--bootstrap-trace"])
        self.assertTrue(args.bootstrap_trace)


if __name__ == "__main__":
    unittest.main()
