"""Tests for support website helpers."""

from __future__ import annotations

import unittest

from core.support_feedback import (
    GITHUB_RELEASES_URL,
    QUBE_WEBSITE_URL,
    manual_update_download_message,
)


class SupportFeedbackTests(unittest.TestCase):
    def test_website_url_constant(self) -> None:
        self.assertEqual(QUBE_WEBSITE_URL, "https://www.qubeapp.eu")

    def test_github_releases_url_constant(self) -> None:
        self.assertEqual(GITHUB_RELEASES_URL, "https://github.com/dagaza/Qube/releases")

    def test_manual_update_download_message_mentions_both_links(self) -> None:
        message = manual_update_download_message()
        self.assertIn(QUBE_WEBSITE_URL, message)
        self.assertIn(GITHUB_RELEASES_URL, message)


if __name__ == "__main__":
    unittest.main()
