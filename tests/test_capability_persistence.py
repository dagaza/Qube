"""T9 — capability persistence: descriptor cache + consent store + drift.

Proves the durable state honours least-privilege (P3/P7): descriptor cache and
consent are separate files, discovery never grants, evaluation is default-deny,
and any tier escalation / contract change / needs-review flag blocks use until
the user re-reviews. Uses ``tmp_path`` via monkeypatching ``user_data_root`` so
nothing touches the real user profile.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.integrations.capabilities import persistence as P
from core.integrations.capabilities.mapper import CapabilityMapper, RawTool
from core.integrations.capabilities.model import CapabilityTier
from core.integrations.capabilities.persistence import (
    ConsentStore,
    capability_fingerprint,
    evaluate_access,
    integrations_dir,
    load_descriptor_cache,
    save_descriptor_cache,
)


def _descriptors(tools):
    group = CapabilityMapper().map_tools("mcp", "docs", tools)
    return list(group.capabilities)


_TOOLS = [
    RawTool("search_docs", "Search", {"type": "object"}),
    RawTool("create_doc", "Create", {"type": "object"}),
]


class _TmpRootTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self._root = Path(self._tmp.name)
        self._orig = P.user_data_root
        P.user_data_root = lambda: self._root  # type: ignore[assignment]

    def tearDown(self):
        P.user_data_root = self._orig  # type: ignore[assignment]
        self._tmp.cleanup()


class TestPaths(_TmpRootTestCase):
    def test_integrations_dir_is_under_user_data_root(self):
        path = integrations_dir("mcp")
        self.assertTrue(path.exists())
        self.assertEqual(path, self._root / "integrations" / "mcp")

    def test_provider_id_normalized(self):
        self.assertEqual(integrations_dir(" MCP ").name, "mcp")


class TestDescriptorCache(_TmpRootTestCase):
    def test_save_then_load_roundtrip(self):
        descs = _descriptors(_TOOLS)
        save_descriptor_cache("mcp", descs)
        loaded = load_descriptor_cache("mcp")
        self.assertEqual(loaded["provider_id"], "mcp")
        self.assertEqual(len(loaded["capabilities"]), 2)
        actions = {c["action"] for c in loaded["capabilities"]}
        self.assertEqual(actions, {"search-docs", "create-doc"})

    def test_missing_cache_returns_empty(self):
        self.assertEqual(load_descriptor_cache("mcp"), {})


class TestConsentStore(_TmpRootTestCase):
    def test_grant_and_deny_persist_separately_from_cache(self):
        descs = {d.action: d for d in _descriptors(_TOOLS)}
        store = ConsentStore("mcp")
        # No descriptor cache written by consent operations.
        store.grant(descs["search-docs"])
        self.assertTrue(store.path.exists())
        self.assertFalse((integrations_dir("mcp") / "descriptors.json").exists())

        grant = store.get(descs["search-docs"].urn)
        self.assertIsNotNone(grant)
        self.assertTrue(grant.granted)
        self.assertEqual(grant.tier, CapabilityTier.READ)

        store.deny(descs["create-doc"])
        self.assertFalse(store.get(descs["create-doc"].urn).granted)

    def test_grant_survives_reload(self):
        descs = {d.action: d for d in _descriptors(_TOOLS)}
        ConsentStore("mcp").grant(descs["search-docs"])
        reopened = ConsentStore("mcp")
        self.assertIsNotNone(reopened.get(descs["search-docs"].urn))


class TestAccessEvaluation(_TmpRootTestCase):
    def setUp(self):
        super().setUp()
        self.descs = {d.action: d for d in _descriptors(_TOOLS)}

    def test_default_deny_without_grant(self):
        decision = evaluate_access(self.descs["search-docs"], None)
        self.assertFalse(decision.allowed)

    def test_explicit_deny_blocks(self):
        d = self.descs["search-docs"]
        grant = ConsentStore("mcp").deny(d)
        self.assertFalse(evaluate_access(d, grant).allowed)

    def test_grant_allows_matching_fingerprint(self):
        d = self.descs["search-docs"]
        grant = ConsentStore("mcp").grant(d)
        self.assertTrue(evaluate_access(d, grant).allowed)

    def test_needs_review_always_denied(self):
        # An unknown verb -> DESTRUCTIVE + needs_review (mapper P7 default).
        weird = _descriptors([RawTool("frobnicate_widget", "?", {})])[0]
        self.assertTrue(weird.needs_review)
        store = ConsentStore("mcp")
        grant = store.grant(weird)  # even a stored grant cannot bypass review
        self.assertFalse(evaluate_access(weird, grant).allowed)

    def test_contract_drift_invalidates_grant(self):
        d = self.descs["search-docs"]
        grant = ConsentStore("mcp").grant(d)
        # Same URN re-discovered but the input schema changed -> new fingerprint.
        changed = _descriptors([RawTool("search_docs", "Search", {"type": "object", "x": 1})])[0]
        self.assertNotEqual(capability_fingerprint(d), capability_fingerprint(changed))
        self.assertFalse(evaluate_access(changed, grant).allowed)

    def test_tier_escalation_invalidates_grant(self):
        # Grant a read; the capability later escalates to write -> deny.
        read = self.descs["search-docs"]
        grant = ConsentStore("mcp").grant(read)
        escalated = _descriptors([RawTool("update_docs", "Now writes", {"type": "object"})])[0]
        # Force the same base URN so only the tier differs in the comparison.
        from dataclasses import replace
        escalated = replace(escalated, urn=read.urn, action=read.action, tier=CapabilityTier.WRITE)
        decision = evaluate_access(escalated, grant)
        self.assertFalse(decision.allowed)


if __name__ == "__main__":
    unittest.main()
