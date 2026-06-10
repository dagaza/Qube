"""Tests for ambient context (daypart, season)."""

from __future__ import annotations

import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from core.companion_cognition.ambient_context import resolve_daypart, resolve_season


class TestCompanionAmbientContext(unittest.TestCase):
    def test_daypart_morning(self) -> None:
        tz = ZoneInfo("UTC")
        ts = datetime(2026, 6, 2, 8, 0, tzinfo=tz).timestamp()
        self.assertEqual(resolve_daypart(ts), "morning")

    def test_daypart_late_night(self) -> None:
        tz = ZoneInfo("UTC")
        ts = datetime(2026, 6, 2, 23, 30, tzinfo=tz).timestamp()
        self.assertEqual(resolve_daypart(ts), "late_night")

    def test_season_north_winter(self) -> None:
        tz = ZoneInfo("UTC")
        ts = datetime(2026, 1, 15, 12, 0, tzinfo=tz).timestamp()
        self.assertEqual(resolve_season(ts, hemisphere="north"), "winter")

    def test_season_south_flip(self) -> None:
        tz = ZoneInfo("UTC")
        ts = datetime(2026, 1, 15, 12, 0, tzinfo=tz).timestamp()
        self.assertEqual(resolve_season(ts, hemisphere="south"), "summer")


if __name__ == "__main__":
    unittest.main()
