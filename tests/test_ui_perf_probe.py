"""Unit tests for Phase A tools-pane UI performance probe env parsing."""

from __future__ import annotations

import json
import logging
import unittest
from unittest.mock import patch

from core import ui_perf_probe as probe


class UiPerfProbeEnvTests(unittest.TestCase):
    def setUp(self) -> None:
        probe.reset_ui_perf_probe_state()

    def tearDown(self) -> None:
        probe.reset_ui_perf_probe_state()

    def test_ui_perf_disabled_by_default(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            probe.reset_ui_perf_probe_state()
            self.assertFalse(probe.ui_perf_enabled())

    def test_ui_perf_enabled_variants(self) -> None:
        for value in ("1", "true", "yes", "on"):
            with self.subTest(value=value):
                with patch.dict("os.environ", {"QUBE_UI_PERF": value}, clear=True):
                    self.assertTrue(probe.ui_perf_enabled())

    def test_tools_pane_anim_ms_default(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(probe.tools_pane_anim_ms(), 350)

    def test_tools_pane_anim_ms_zero_instant(self) -> None:
        with patch.dict("os.environ", {"QUBE_TOOLS_PANE_ANIM_MS": "0"}, clear=True):
            self.assertEqual(probe.tools_pane_anim_ms(), 0)

    def test_toggle_summary_json_logged(self) -> None:
        with patch.dict("os.environ", {"QUBE_UI_PERF": "1"}, clear=True):
            probe.begin_tools_pane_toggle(
                target_expanded=False,
                animated=False,
                nominal_anim_ms=0,
            )
            probe.ui_perf_count("wallpaper.paint", 3)
            with self.assertLogs("Qube.UiPerf", level="INFO") as captured:
                probe.finish_tools_pane_toggle()
        summary_line = next(
            line for line in captured.output if "tools_pane_toggle_summary" in line
        )
        payload = json.loads(summary_line.split("tools_pane_toggle_summary ", 1)[1])
        self.assertEqual(payload["event"], "tools_pane_toggle_end")
        self.assertEqual(payload["counters"]["wallpaper.paint"], 3)

    def test_window_resize_session_summary_json_logged(self) -> None:
        with patch.dict("os.environ", {"QUBE_UI_PERF": "1"}, clear=True):
            probe.reset_ui_perf_probe_state()
            probe.begin_window_resize_session(coalesce_interval_ms=120)
            probe.record_window_resize_frame(width=1200, height=900)
            probe.record_window_resize_frame(width=1210, height=905)
            probe.ui_perf_count("wallpaper.background_cache_build", 1)
            with self.assertLogs("Qube.UiPerf", level="INFO") as captured:
                probe.finish_window_resize_session()
        summary_line = next(
            line for line in captured.output if "window_resize_session_summary" in line
        )
        payload = json.loads(summary_line.split("window_resize_session_summary ", 1)[1])
        self.assertEqual(payload["event"], "window_resize_session_end")
        self.assertEqual(payload["main_resize_events"], 2)
        self.assertEqual(payload["start_size"], [1200, 900])
        self.assertEqual(payload["end_size"], [1210, 905])
        self.assertEqual(payload["counters"]["wallpaper.background_cache_build"], 1)


if __name__ == "__main__":
    unittest.main()