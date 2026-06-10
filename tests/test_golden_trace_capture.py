"""Tests for golden baseline trace capture."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.canonical_request import CanonicalMessage, CanonicalRequest, CanonicalSampling
from core.canonical_trace_diff import find_first_divergence, traces_equal
from core.golden_trace_capture import (
    build_golden_trace,
    golden_trace_capture_mode_enabled,
    load_golden_trace,
    maybe_capture_golden_trace,
    reset_golden_trace_capture_for_tests,
    save_golden_trace,
)


class GoldenTraceCaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_golden_trace_capture_for_tests()

    def tearDown(self) -> None:
        reset_golden_trace_capture_for_tests()

    def test_disabled_by_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(golden_trace_capture_mode_enabled())

    def test_build_and_load_round_trip(self) -> None:
        trace = build_golden_trace(
            request=CanonicalRequest(
                model="demo.gguf",
                messages=[CanonicalMessage(role="user", content="hi")],
                sampling=CanonicalSampling(temperature=0.5),
            ),
            prompt="rendered",
            output="answer",
            metadata={"exchange_id": 1},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "baseline.json"
            save_golden_trace(trace, path)
            loaded = load_golden_trace(path)
        self.assertEqual(loaded.prompt, "rendered")
        self.assertEqual(loaded.output, "answer")
        self.assertEqual(loaded.request.model, "demo.gguf")
        self.assertIn("request", loaded.fingerprints)
        self.assertIn("prompt", loaded.fingerprints)
        self.assertIn("output", loaded.fingerprints)

    def test_capture_once_per_process(self) -> None:
        trace = build_golden_trace(
            request={"messages": [{"role": "user", "content": "x"}]},
            prompt="p",
            output="o",
        )
        with tempfile.TemporaryDirectory() as tmp:
            traces_dir = Path(tmp)
            with patch.dict(os.environ, {"GOLDEN_TRACE_CAPTURE_MODE": "1"}, clear=False):
                first = maybe_capture_golden_trace(trace, traces_dir=traces_dir)
                second = maybe_capture_golden_trace(trace, traces_dir=traces_dir)
            self.assertIsNotNone(first)
            self.assertIsNone(second)
            self.assertEqual(len(list(traces_dir.glob("*.json"))), 1)

    def test_loaded_trace_usable_for_regression_diff(self) -> None:
        baseline = build_golden_trace(
            request={"model": "m", "messages": [{"role": "user", "content": "q"}]},
            prompt="prompt",
            output="out",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "golden.json"
            save_golden_trace(baseline, path)
            loaded = load_golden_trace(path)
        same = build_golden_trace(
            request={"model": "m", "messages": [{"role": "user", "content": "q"}]},
            prompt="prompt",
            output="out",
        )
        self.assertTrue(traces_equal(loaded, same))
        drift = build_golden_trace(
            request={"model": "m", "messages": [{"role": "user", "content": "q"}]},
            prompt="prompt",
            output="different",
        )
        report = find_first_divergence(loaded, drift)
        self.assertEqual(report["first_divergence_level"], "OUTPUT")

    def test_capture_noop_when_flag_off(self) -> None:
        trace = build_golden_trace(
            request={"messages": []},
            prompt="",
            output="",
        )
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {}, clear=True):
                path = maybe_capture_golden_trace(trace, traces_dir=Path(tmp))
            self.assertIsNone(path)
            self.assertEqual(list(Path(tmp).glob("*.json")), [])


if __name__ == "__main__":
    unittest.main()
