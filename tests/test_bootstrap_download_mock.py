"""Tests for mock bootstrap download simulation."""

from __future__ import annotations

import os
import time
import unittest
from unittest.mock import patch

from core.bootstrap_download import (
    bootstrap_download_mock_enabled,
    bootstrap_download_should_mock,
    estimate_mock_download_seconds,
    run_bootstrap_model_download,
    settings_bootstrap_download_should_mock,
    simulate_bootstrap_downloads,
)
from core.bootstrap_manifest import BootstrapModelId, default_selection


class BootstrapDownloadMockTests(unittest.TestCase):
    def test_mock_enabled_from_env(self) -> None:
        with patch.dict(os.environ, {"QUBE_BOOTSTRAP_MOCK_DOWNLOAD": "1"}, clear=False):
            self.assertTrue(bootstrap_download_mock_enabled())
        with patch.dict(os.environ, {"QUBE_BOOTSTRAP_MOCK_DOWNLOAD": "0"}, clear=False):
            self.assertFalse(bootstrap_download_mock_enabled())

    def test_estimate_seconds_scales_with_selection_size(self) -> None:
        small = {BootstrapModelId.SIDECAR_QWEN05}
        large = default_selection(advanced=False)
        self.assertLess(
            estimate_mock_download_seconds(small, speed_multiplier=10),
            estimate_mock_download_seconds(large, speed_multiplier=10),
        )

    def test_simulate_emits_monotonic_progress(self) -> None:
        selected = {BootstrapModelId.SIDECAR_QWEN05}
        events: list[tuple[str, int]] = []

        def on_progress(step_label: str, filename: str, percent: int, _source: str) -> None:
            events.append((filename, percent))

        with patch.dict(os.environ, {"QUBE_BOOTSTRAP_MOCK_DOWNLOAD_SPEED": "50"}, clear=False):
            started = time.monotonic()
            errors = simulate_bootstrap_downloads(selected, on_progress, speed_multiplier=50)
            elapsed = time.monotonic() - started

        self.assertEqual(errors, [])
        self.assertGreaterEqual(len(events), 3)
        self.assertEqual(events[-1][1], 100)
        percents = [pct for _, pct in events]
        self.assertEqual(percents, sorted(percents))
        self.assertGreaterEqual(elapsed, 0.2)

    def test_should_not_mock_without_explicit_flag(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("QUBE_BOOTSTRAP_REAL_DOWNLOAD", None)
            os.environ.pop("QUBE_BOOTSTRAP_MOCK_DOWNLOAD", None)
            self.assertFalse(bootstrap_download_should_mock())
            self.assertFalse(bootstrap_download_should_mock(explicit_mock=False))

    def test_should_mock_with_explicit_flag_or_env(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("QUBE_BOOTSTRAP_MOCK_DOWNLOAD", None)
            self.assertTrue(bootstrap_download_should_mock(explicit_mock=True))
        with patch.dict(os.environ, {"QUBE_BOOTSTRAP_MOCK_DOWNLOAD": "1"}, clear=False):
            self.assertTrue(bootstrap_download_should_mock())

    def test_real_download_forced_disables_mock(self) -> None:
        with patch.dict(
            os.environ,
            {"QUBE_BOOTSTRAP_REAL_DOWNLOAD": "1", "QUBE_BOOTSTRAP_MOCK_DOWNLOAD": "1"},
            clear=False,
        ):
            self.assertFalse(bootstrap_download_should_mock())
            self.assertFalse(bootstrap_download_should_mock(explicit_mock=True))

    def test_settings_mock_only_when_explicit_env(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("QUBE_BOOTSTRAP_MOCK_DOWNLOAD", None)
            os.environ.pop("QUBE_BOOTSTRAP_REAL_DOWNLOAD", None)
            self.assertFalse(settings_bootstrap_download_should_mock())
        with patch.dict(os.environ, {"QUBE_BOOTSTRAP_MOCK_DOWNLOAD": "1"}, clear=False):
            self.assertTrue(settings_bootstrap_download_should_mock())

    def test_run_bootstrap_model_download_uses_mock_when_env_set(self) -> None:
        selected = {BootstrapModelId.SIDECAR_QWEN05}
        with patch.dict(os.environ, {"QUBE_BOOTSTRAP_MOCK_DOWNLOAD": "1"}, clear=False):
            with patch(
                "core.bootstrap_download.simulate_bootstrap_downloads",
                return_value=[],
            ) as simulate:
                with patch("core.bootstrap_download.download_bootstrap_models") as real:
                    errors, used_mock = run_bootstrap_model_download(selected, lambda *_: None)
        self.assertEqual(errors, [])
        self.assertTrue(used_mock)
        simulate.assert_called_once()
        real.assert_not_called()


if __name__ == "__main__":
    unittest.main()
