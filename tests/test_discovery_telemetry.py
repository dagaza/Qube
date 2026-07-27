"""Tests for web discovery Telemetry snapshot (R10 / Theme B)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from core.knowledge.discovery_telemetry import (
    discovery_telemetry_snapshot,
    format_discovery_health_status,
)


class DiscoveryTelemetryTests(unittest.TestCase):
    def test_snapshot_includes_tier_and_provider(self) -> None:
        snap = discovery_telemetry_snapshot()
        self.assertIn("privacy_tier_label", snap)
        self.assertIn("primary_provider_label", snap)
        self.assertIsInstance(snap["policy_summary_lines"], list)
        self.assertTrue(snap["policy_summary_lines"])

    def test_health_stable_by_default(self) -> None:
        self.assertIn("stable", format_discovery_health_status().lower())

    def test_health_backoff_flag(self) -> None:
        status = format_discovery_health_status({"backoff_active": True})
        self.assertIn("backoff", status.lower())

    @patch(
        "core.knowledge.discovery_telemetry.is_conservative_mode_active",
        return_value=True,
    )
    def test_health_conservative(self, _mock: object) -> None:
        status = format_discovery_health_status({"conservative_mode": True})
        self.assertIn("conservative", status.lower())


if __name__ == "__main__":
    unittest.main()
