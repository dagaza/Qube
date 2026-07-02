"""Tests for evidence retrieval cache (HTTP resilience Slice 5)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge import evidence_cache as ec  # noqa: E402


class TestEvidenceCacheTTL(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("QUBE_EVIDENCE_CACHE", None)
        os.environ.pop("QUBE_EVIDENCE_CACHE_TTL", None)

    def test_default_ttl_is_one_hour(self) -> None:
        os.environ.pop("QUBE_EVIDENCE_CACHE_TTL", None)
        self.assertEqual(ec.evidence_cache_ttl_seconds(), 3600)

    def test_env_ttl_override(self) -> None:
        os.environ["QUBE_EVIDENCE_CACHE_TTL"] = "86400"
        self.assertEqual(ec.evidence_cache_ttl_seconds(), 86400)

    def test_get_cached_rows_uses_env_ttl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            key = "abc123"
            path = cache_dir / f"{key}.json"
            path.write_text(
                json.dumps(
                    {
                        "ts": time.time() - 7200,
                        "rows": [{"title": "cached"}],
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(ec, "_CACHE_DIR", cache_dir):
                os.environ["QUBE_EVIDENCE_CACHE_TTL"] = "86400"
                rows = ec.get_cached_rows(key)
                self.assertEqual(len(rows or []), 1)

                os.environ["QUBE_EVIDENCE_CACHE_TTL"] = "3600"
                rows = ec.get_cached_rows(key)
                self.assertIsNone(rows)

    def test_ttl_zero_disables_reads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            key = "fresh"
            (cache_dir / f"{key}.json").write_text(
                json.dumps({"ts": time.time(), "rows": [{"title": "x"}]}),
                encoding="utf-8",
            )
            with patch.object(ec, "_CACHE_DIR", cache_dir):
                os.environ["QUBE_EVIDENCE_CACHE_TTL"] = "0"
                self.assertIsNone(ec.get_cached_rows(key))


if __name__ == "__main__":
    unittest.main()
