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
        ):
            out, applied = digest_memory_context(
                "- User likes metric units",
                sources,
                client,
            )
        self.assertTrue(applied)
        self.assertIn("[1]", out)

    def test_fallback_when_disabled(self) -> None:
        with mock.patch(
            "core.source_digest.get_sidecar_source_digest_enabled",
            return_value=False,
        ):
            out, applied = digest_memory_context("raw", [], mock.Mock())
        self.assertEqual(out, "raw")
        self.assertFalse(applied)


if __name__ == "__main__":
    unittest.main()
