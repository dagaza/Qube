"""Tests for support website helpers."""

from __future__ import annotations

import unittest

from core.support_feedback import QUBE_WEBSITE_URL


class SupportFeedbackTests(unittest.TestCase):
    def test_website_url_constant(self) -> None:
        self.assertEqual(QUBE_WEBSITE_URL, "https://www.qubeapp.eu")


if __name__ == "__main__":
    unittest.main()
