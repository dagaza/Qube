"""Tests for adapter readiness metadata."""

from __future__ import annotations

import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.adapter_readiness import (  # noqa: E402
    adapters_by_readiness,
    get_adapter_readiness_meta,
    implemented_adapter_readiness,
)
from core.knowledge.adapters.catalog import get_adapter_entry, readiness_for_entry  # noqa: E402


class TestAdapterReadiness(unittest.TestCase):
    def test_live_adapters_are_not_stub(self) -> None:
        for adapter_id, meta in implemented_adapter_readiness().items():
            self.assertNotEqual(
                meta.readiness,
                "stub",
                msg=f"{adapter_id} is live but marked stub",
            )

    def test_acm_dl_is_beta_indirect(self) -> None:
        meta = get_adapter_readiness_meta("acm_dl")
        self.assertEqual(meta.readiness, "beta")
        self.assertIn("OpenAlex", meta.production_strategy)

    def test_pubmed_is_production(self) -> None:
        entry = get_adapter_entry("pubmed")
        self.assertIsNotNone(entry)
        assert entry is not None
        meta = readiness_for_entry(entry)
        self.assertEqual(meta.readiness, "production")

    def test_unimplemented_adapter_is_stub(self) -> None:
        meta = get_adapter_readiness_meta(
            "not_a_real_adapter",
            implemented=False,
        )
        self.assertEqual(meta.readiness, "stub")

    def test_production_and_beta_sets_partition_live(self) -> None:
        live = set(implemented_adapter_readiness())
        partitioned = set(adapters_by_readiness("production")) | set(
            adapters_by_readiness("beta")
        )
        self.assertEqual(live, partitioned)
