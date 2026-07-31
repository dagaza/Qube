"""T12 — Integrations consent controller (P3/P7)."""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from core.integrations.capabilities import persistence as P
from core.integrations.capabilities.mapper import CapabilityMapper, RawTool
from core.integrations.capabilities.model import CapabilityTier
from core.integrations.capabilities.persistence import (
    ConsentStore,
    capability_fingerprint,
    save_descriptor_cache,
)
from core.integrations.consent_controller import (
    ConsentUIState,
    IntegrationsConsentController,
    derive_consent_ui_state,
    load_cached_descriptors,
)
from core.integrations.capabilities.persistence import evaluate_access


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


class TestIntegrationsConsentController(_TmpRootTestCase):
    def setUp(self):
        super().setUp()
        self.descs = {d.action: d for d in _descriptors(_TOOLS)}
        save_descriptor_cache("mcp", list(self.descs.values()))
        self.controller = IntegrationsConsentController("mcp")

    def test_lists_groups_tiers_and_needs_review(self):
        groups = self.controller.list_groups()
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].name, "docs")
        self.assertEqual(len(groups[0].capabilities), 2)

        weird = _descriptors([RawTool("frobnicate_widget", "?", {})])[0]
        self.assertTrue(weird.needs_review)
        ctrl = IntegrationsConsentController("mcp", descriptors=[weird])
        row = ctrl.list_capability_rows()[0]
        self.assertEqual(row.tier, CapabilityTier.DESTRUCTIVE)
        self.assertTrue(row.needs_review)
        self.assertEqual(row.ui_state, ConsentUIState.NEEDS_REVIEW)

    def test_state_derives_from_evaluate_access_not_grant_presence(self):
        read = self.descs["search-docs"]
        rows = self.controller.list_capability_rows()
        read_row = next(r for r in rows if r.descriptor.action == "search-docs")
        self.assertEqual(read_row.ui_state, ConsentUIState.DENIED)

        self.controller.grant_capability(read)
        rows = self.controller.list_capability_rows()
        read_row = next(r for r in rows if r.descriptor.action == "search-docs")
        self.assertEqual(read_row.ui_state, ConsentUIState.ALLOWED)

    def test_grant_and_deny_write_exact_descriptor_and_survive_reload(self):
        read = self.descs["search-docs"]
        write = self.descs["create-doc"]

        self.controller.grant_capability(read)
        self.controller.deny_capability(write)

        reopened = IntegrationsConsentController("mcp")
        rows = {r.descriptor.action: r for r in reopened.list_capability_rows()}
        self.assertEqual(rows["search-docs"].ui_state, ConsentUIState.ALLOWED)
        self.assertEqual(rows["create-doc"].ui_state, ConsentUIState.DENIED)

        grant = ConsentStore("mcp").get(read.urn)
        self.assertIsNotNone(grant)
        self.assertEqual(grant.fingerprint, capability_fingerprint(read))

    def test_needs_review_stays_ungrantable(self):
        weird = _descriptors([RawTool("frobnicate_widget", "?", {})])[0]
        ctrl = IntegrationsConsentController("mcp", descriptors=[weird])
        with self.assertRaises(ValueError):
            ctrl.grant_capability(weird)
        row = ctrl.list_capability_rows()[0]
        self.assertEqual(row.ui_state, ConsentUIState.NEEDS_REVIEW)

    def test_drift_surfaces_rereview_required(self):
        read = self.descs["search-docs"]
        self.controller.grant_capability(read)
        changed = _descriptors(
            [RawTool("search_docs", "Search", {"type": "object", "x": 1})]
        )[0]
        ctrl = IntegrationsConsentController("mcp", descriptors=[changed])
        row = ctrl.list_capability_rows()[0]
        self.assertEqual(row.ui_state, ConsentUIState.REREVIEW_REQUIRED)
        self.assertFalse(row.decision.allowed)

    def test_tier_escalation_surfaces_rereview_required(self):
        read = self.descs["search-docs"]
        self.controller.grant_capability(read)
        escalated = replace(read, tier=CapabilityTier.WRITE)
        ctrl = IntegrationsConsentController("mcp", descriptors=[escalated])
        row = ctrl.list_capability_rows()[0]
        self.assertEqual(row.ui_state, ConsentUIState.REREVIEW_REQUIRED)

    def test_load_cached_descriptors_roundtrip(self):
        loaded = load_cached_descriptors("mcp")
        self.assertEqual(len(loaded), 2)
        actions = {d.action for d in loaded}
        self.assertEqual(actions, {"search-docs", "create-doc"})

    def test_derive_consent_ui_state_rereview_only_when_grant_was_allowed(self):
        read = self.descs["search-docs"]
        decision = evaluate_access(read, None)
        self.assertEqual(
            derive_consent_ui_state(read, decision, grant_granted=None),
            ConsentUIState.DENIED,
        )
        grant = ConsentStore("mcp").grant(read)
        changed = _descriptors(
            [RawTool("search_docs", "Search", {"type": "object", "x": 1})]
        )[0]
        stale_decision = evaluate_access(changed, grant)
        self.assertEqual(
            derive_consent_ui_state(changed, stale_decision, grant_granted=True),
            ConsentUIState.REREVIEW_REQUIRED,
        )


if __name__ == "__main__":
    unittest.main()
