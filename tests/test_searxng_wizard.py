"""Tests for SearXNG setup wizard helpers."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from core.knowledge.discovery.searxng_wizard import (
    LOCAL_SEARXNG_CANDIDATES,
    normalize_searxng_base_url,
    probe_searxng_base_url,
    scan_local_searxng_candidates,
)


class SearXNGWizardTests(unittest.TestCase):
    def test_normalize_adds_scheme_and_strips_slash(self) -> None:
        self.assertEqual(
            normalize_searxng_base_url("127.0.0.1:8080"),
            "http://127.0.0.1:8080",
        )
        self.assertEqual(
            normalize_searxng_base_url("https://search.example.org/"),
            "https://search.example.org",
        )

    def test_normalize_rejects_invalid(self) -> None:
        self.assertEqual(normalize_searxng_base_url(""), "")
        self.assertEqual(normalize_searxng_base_url("ftp://bad"), "")

    @patch("core.knowledge.discovery.searxng_wizard.search_searxng")
    def test_probe_success(self, mock_search) -> None:
        mock_search.return_value = (
            [{"url": "https://example.org", "title": "x", "snippet": "y"}],
            {"response_kind": "serp", "http_status": 200, "parsed_rows": 1},
        )
        result = probe_searxng_base_url("http://127.0.0.1:8080")
        self.assertTrue(result.ok)
        self.assertEqual(result.base_url, "http://127.0.0.1:8080")
        self.assertIn("1 result", result.message)

    @patch("core.knowledge.discovery.searxng_wizard.search_searxng")
    def test_probe_auth_error(self, mock_search) -> None:
        mock_search.return_value = (
            [],
            {"response_kind": "auth_error", "http_status": 403, "parsed_rows": 0},
        )
        result = probe_searxng_base_url("http://127.0.0.1:8080", api_key="bad")
        self.assertFalse(result.ok)
        self.assertIn("Authentication", result.message)

    @patch("core.knowledge.discovery.searxng_wizard.probe_searxng_base_url")
    @patch("core.knowledge.discovery.searxng_wizard._candidate_urls")
    def test_scan_returns_only_ok_hits(self, mock_candidates, mock_probe) -> None:
        from core.knowledge.discovery.searxng_wizard import SearXNGProbeResult

        mock_candidates.return_value = list(LOCAL_SEARXNG_CANDIDATES[:2])

        def _side_effect(url: str, **kwargs: object):
            ok = "8080" in url
            return SearXNGProbeResult(
                base_url=url,
                ok=ok,
                message="ok" if ok else "fail",
            )

        mock_probe.side_effect = _side_effect
        hits = scan_local_searxng_candidates(timeout=0.1)
        self.assertEqual(len(hits), 2)
        self.assertTrue(all("8080" in h.base_url for h in hits))

    @patch("core.knowledge.discovery.searxng_wizard.shutil.which", return_value="/usr/bin/docker")
    @patch("core.knowledge.discovery.searxng_wizard.subprocess.run")
    def test_docker_cli_available(self, mock_run, _mock_which) -> None:
        from core.knowledge.discovery.searxng_wizard import docker_cli_available

        mock_run.return_value = MagicMock(returncode=0)
        self.assertTrue(docker_cli_available())


if __name__ == "__main__":
    unittest.main()
