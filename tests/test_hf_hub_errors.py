"""Tests for Hugging Face Hub error classification."""

from __future__ import annotations

import unittest

import requests

from core.hf_hub_errors import (
    HubErrorKind,
    classify_hf_error,
    classify_hf_http_status,
    coerce_hub_error,
)


class HfHubErrorsTests(unittest.TestCase):
    def test_classify_http_503_is_retryable_server_error(self) -> None:
        info = classify_hf_http_status(503)
        self.assertEqual(info.kind, HubErrorKind.SERVER)
        self.assertTrue(info.retryable)
        self.assertTrue(info.show_status_link)
        self.assertIn("server error", info.message.lower())

    def test_classify_http_404_not_retryable(self) -> None:
        info = classify_hf_http_status(404)
        self.assertEqual(info.kind, HubErrorKind.NOT_FOUND)
        self.assertFalse(info.retryable)

    def test_classify_connection_error(self) -> None:
        info = classify_hf_error(requests.ConnectionError("Connection refused"))
        self.assertEqual(info.kind, HubErrorKind.CONNECTION)
        self.assertTrue(info.is_platform_outage)
        self.assertTrue(info.inline_only)

    def test_classify_timeout_error(self) -> None:
        info = classify_hf_error(requests.Timeout("HTTPSConnectionPool read timed out"))
        self.assertEqual(info.kind, HubErrorKind.TIMEOUT)
        self.assertTrue(info.retryable)

    def test_classify_embedded_http_code_in_message(self) -> None:
        info = classify_hf_error("HTTP 502 — bad gateway")
        self.assertEqual(info.kind, HubErrorKind.SERVER)

    def test_coerce_hub_error_accepts_legacy_string(self) -> None:
        info = coerce_hub_error("Connection refused")
        self.assertEqual(info.kind, HubErrorKind.CONNECTION)

    def test_validation_error_not_platform_outage(self) -> None:
        info = classify_hf_error(ValueError("Invalid repository id format."))
        self.assertEqual(info.kind, HubErrorKind.VALIDATION)
        self.assertFalse(info.is_platform_outage)


if __name__ == "__main__":
    unittest.main()
