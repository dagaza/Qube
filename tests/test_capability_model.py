"""T2/T3/T5 — fingerprint stability, NormalizedHit provenance, tier ordering."""

import unittest

from core.integrations.capabilities.model import (
    CapabilityDescriptor,
    CapabilityTier,
    NormalizedHit,
    fingerprint_descriptors,
)
from core.integrations.capabilities.urn import CapabilityURN


def _descriptor(action: str, tier: CapabilityTier, schema=None) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        urn=CapabilityURN.build("mcp", "github", action),
        group="github",
        action=action,
        tier=tier,
        description="cosmetic text that must not affect the fingerprint",
        input_schema=schema or {"type": "object", "properties": {"q": {"type": "string"}}},
    )


class TestFingerprint(unittest.TestCase):
    """T2 — fingerprint_descriptors is stable, order-independent, drift-sensitive."""

    def test_stable_and_order_independent(self):
        a = _descriptor("search-issues", CapabilityTier.READ)
        b = _descriptor("create-issue", CapabilityTier.WRITE)
        fp1 = fingerprint_descriptors([a, b])
        fp2 = fingerprint_descriptors([b, a])
        self.assertEqual(fp1, fp2)
        self.assertEqual(fp1, fingerprint_descriptors([a, b]))

    def test_ignores_cosmetic_fields(self):
        base = _descriptor("search-issues", CapabilityTier.READ)
        cosmetic = CapabilityDescriptor(
            urn=base.urn,
            group=base.group,
            action=base.action,
            tier=base.tier,
            description="a totally different description",
            input_schema=base.input_schema,
            tags=("new", "tags"),
            raw_ref="search_issues_v2",
            needs_review=True,
        )
        self.assertEqual(
            fingerprint_descriptors([base]),
            fingerprint_descriptors([cosmetic]),
        )

    def test_changes_on_tier_drift(self):
        read = _descriptor("search-issues", CapabilityTier.READ)
        write = _descriptor("search-issues", CapabilityTier.WRITE)
        self.assertNotEqual(
            fingerprint_descriptors([read]),
            fingerprint_descriptors([write]),
        )

    def test_changes_on_schema_drift(self):
        a = _descriptor("search-issues", CapabilityTier.READ, schema={"type": "object"})
        b = _descriptor(
            "search-issues",
            CapabilityTier.READ,
            schema={"type": "object", "properties": {"state": {"type": "string"}}},
        )
        self.assertNotEqual(
            fingerprint_descriptors([a]),
            fingerprint_descriptors([b]),
        )

    def test_changes_when_capability_added(self):
        a = _descriptor("search-issues", CapabilityTier.READ)
        b = _descriptor("read-pr", CapabilityTier.READ)
        self.assertNotEqual(
            fingerprint_descriptors([a]),
            fingerprint_descriptors([a, b]),
        )


class TestNormalizedHitProvenance(unittest.TestCase):
    """T3 — NormalizedHit.to_evidence_dict preserves cap: provenance (P8)."""

    def test_preserves_source_cap(self):
        cap = CapabilityURN.parse("cap:mcp:github/search-issues@2")
        hit = NormalizedHit(
            title="Crash on export",
            snippet="Users report a crash when exporting large PDFs.",
            source_cap=cap,
            url="https://example.invalid/issues/4821",
            full_text="full body",
        )
        row = hit.to_evidence_dict()
        # Full URN (incl. version) is preserved end-to-end.
        self.assertEqual(row["_capability"], "cap:mcp:github/search-issues@2")
        # The provider origin survives so the hit is attributable.
        self.assertEqual(row["retrieval_method"], "mcp")
        self.assertEqual(row["_adapter"], "cap:mcp:github/search-issues")
        # The versionless base is recoverable from the row.
        self.assertEqual(
            CapabilityURN.parse(row["_capability"]).base,
            cap.base,
        )
        # Legacy adapter-row shape is intact.
        for key in ("title", "snippet", "full_text", "url"):
            self.assertIn(key, row)

    def test_shape_matches_live_adapter_keys(self):
        cap = CapabilityURN.build("live", "pubmed", "search")
        row = NormalizedHit(title="t", snippet="s", source_cap=cap).to_evidence_dict()
        expected_keys = {
            "title", "snippet", "full_text", "url", "_adapter",
            "retrieval_method", "_capability",
        }
        self.assertTrue(expected_keys.issubset(row.keys()))


class TestCapabilityTier(unittest.TestCase):
    """T5 — tier escalation ordering read < write < destructive."""

    def test_rank_order(self):
        self.assertLess(CapabilityTier.READ.rank, CapabilityTier.WRITE.rank)
        self.assertLess(CapabilityTier.WRITE.rank, CapabilityTier.DESTRUCTIVE.rank)

    def test_escalates_over(self):
        self.assertTrue(CapabilityTier.WRITE.escalates_over(CapabilityTier.READ))
        self.assertTrue(CapabilityTier.DESTRUCTIVE.escalates_over(CapabilityTier.READ))
        self.assertTrue(CapabilityTier.DESTRUCTIVE.escalates_over(CapabilityTier.WRITE))

    def test_does_not_escalate_same_or_lower(self):
        self.assertFalse(CapabilityTier.READ.escalates_over(CapabilityTier.READ))
        self.assertFalse(CapabilityTier.READ.escalates_over(CapabilityTier.WRITE))
        self.assertFalse(CapabilityTier.WRITE.escalates_over(CapabilityTier.DESTRUCTIVE))

    def test_string_enum_values(self):
        self.assertEqual(CapabilityTier.READ.value, "read")
        self.assertEqual(CapabilityTier.WRITE.value, "write")
        self.assertEqual(CapabilityTier.DESTRUCTIVE.value, "destructive")


if __name__ == "__main__":
    unittest.main()
