"""Source digest validation."""
from __future__ import annotations

import unittest
from unittest import mock

from core.source_digest import digest_memory_context
from core.sidecar_types import SidecarResult, SidecarTask


class TestSourceDigest(unittest.TestCase):
    def test_digest_applied_when_citations_preserved(self) -> None:
        sources = [
            {"id": 1, "filename": "Mem", "content": "User likes metric units", "memory_id": "x"},
        ]
        long_context = "- User likes metric units\n" + ("detail line.\n" * 500)
        client = mock.Mock()
        client.complete.return_value = SidecarResult(
            ok=True,
            text="[1] User prefers metric units.",
            task=SidecarTask.source_digest,
        )
        with mock.patch(
            "core.source_digest.get_sidecar_source_digest_enabled",
            return_value=True,
        ), mock.patch(
            "core.source_digest.get_sidecar_foreground_timeout_ms",
            return_value=1500,
        ), mock.patch(
            "core.source_digest.get_sidecar_source_digest_min_chars",
            return_value=100,
        ):
            result = digest_memory_context(long_context, sources, client)
        self.assertTrue(result.applied)
        self.assertIn("[1]", result.text)
        self.assertGreater(result.chars_before, result.chars_after)

    def test_skips_when_below_char_threshold(self) -> None:
        sources = [
            {"id": 1, "filename": "Mem", "content": "Short fact", "memory_id": "x"},
        ]
        raw = "- Short fact"
        client = mock.Mock()
        with mock.patch(
            "core.source_digest.get_sidecar_source_digest_enabled",
            return_value=True,
        ), mock.patch(
            "core.source_digest.get_sidecar_source_digest_min_chars",
            return_value=4096,
        ):
            result = digest_memory_context(raw, sources, client)
        self.assertFalse(result.applied)
        self.assertEqual(result.text, raw)
        self.assertEqual(result.skip_reason, "below_threshold")
        client.complete.assert_not_called()

    def test_fallback_when_disabled(self) -> None:
        sources = [{"id": 1, "filename": "Mem", "content": "fact"}]
        raw = "- fact\n" + ("detail.\n" * 500)
        with mock.patch(
            "core.source_digest.get_sidecar_source_digest_enabled",
            return_value=False,
        ), mock.patch(
            "core.source_digest.get_sidecar_source_digest_min_chars",
            return_value=100,
        ):
            result = digest_memory_context(raw, sources, mock.Mock())
        self.assertEqual(result.text, raw)
        self.assertFalse(result.applied)
        self.assertEqual(result.skip_reason, "disabled")


if __name__ == "__main__":
    unittest.main()
