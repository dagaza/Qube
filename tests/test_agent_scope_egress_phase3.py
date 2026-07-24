"""Phase 3 / #61 — agent scope, step approval, session egress (T19–T21)."""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.composer_attachments import ComposerAttachment
from core.integrations.agent_scope import (
    AgentScope,
    agent_scope_store,
    build_agent_scope_from_attachments,
    urn_base_key,
)
from core.integrations.capabilities import persistence as P
from core.integrations.capabilities.mapper import CapabilityMapper, RawTool
from core.integrations.capabilities.model import CapabilityTier, NormalizedHit
from core.integrations.capabilities.persistence import ConsentStore, save_descriptor_cache
from core.integrations.capabilities.urn import CapabilityURN
from core.integrations.capability_invoke import (
    invoke_gated_capability,
    preview_gated_capability,
)
from core.integrations.composer_capability_gate import (
    capabilities_requiring_step_approval,
    format_step_approval_message,
    pending_step_approvals,
)
from core.integrations.egress_summary import (
    format_privacy_report_integrations_section,
    format_session_egress_summary,
)
from core.integrations.registry.provider_registry import (
    register_capability_provider,
    reset_registry_for_tests,
)
from core.integrations.session_egress import session_egress_ledger
from core.integrations.step_approval import (
    requires_step_approval,
    step_approval_store,
)

_P6_PATTERNS = (
    re.compile(r"\bimport\s+mcp\b"),
    re.compile(r"\bfrom\s+mcp\b"),
    re.compile(r"provider\s*==\s*['\"]mcp['\"]"),
)


def _descriptors(tools):
    group = CapabilityMapper().map_tools("fake", "GitHub", tools)
    return list(group.capabilities)


class _FakeProvider:
    provider_id = "fake"

    def __init__(self, **config):
        self.config = config
        self._descriptors = list(config.get("descriptors") or [])
        self.invoke_calls = 0

    async def discover(self):
        return self._descriptors

    async def invoke(self, urn, args, *, ctx):
        if ctx.dry_run:
            return [
                NormalizedHit(
                    title="preview",
                    snippet="dry-run preview",
                    url="",
                    source_cap=urn,
                )
            ]
        self.invoke_calls += 1
        return [
            NormalizedHit(
                title="hit",
                snippet="snippet",
                url="",
                source_cap=urn,
            )
        ]

    async def health(self):
        from core.integrations.capabilities.model import HealthState, HealthStatus

        return HealthStatus(state=HealthState.OK, message="ok")

    def fingerprint(self):
        return "fp"

    def close(self):
        pass


class _TmpRootTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self._root = Path(self._tmp.name)
        self._orig = P.user_data_root
        P.user_data_root = lambda: self._root  # type: ignore[assignment]

    def tearDown(self):
        P.user_data_root = self._orig  # type: ignore[assignment]
        self._tmp.cleanup()


class TestAgentScope(unittest.TestCase):
    def test_scope_allows_attached_urn_only(self):
        attachments = [
            ComposerAttachment(
                kind="capability",
                id="cap:fake:github/search-issues",
                label="Search issues",
            )
        ]
        scope = build_agent_scope_from_attachments("sess-1", attachments)
        allowed, _ = scope.check("cap:fake:github/search-issues")
        self.assertTrue(allowed)
        denied, reason = scope.check("cap:fake:github/create-issue")
        self.assertFalse(denied)
        self.assertIn("scope", reason)

    def test_agent_scope_store_roundtrip(self):
        scope = AgentScope(
            session_id="sess-2",
            allowed_urn_bases=frozenset({urn_base_key("cap:fake:github/search-issues")}),
        )
        agent_scope_store.set_scope(scope)
        loaded = agent_scope_store.get_scope("sess-2")
        self.assertIsNotNone(loaded)
        self.assertTrue(loaded.allows("cap:fake:github/search-issues"))


class TestStepApproval(_TmpRootTestCase):
    def setUp(self):
        super().setUp()
        self.write_desc = _descriptors([RawTool(name="create_issue")])[0]
        save_descriptor_cache("fake", [self.write_desc])

    def test_write_requires_step_approval(self):
        desc = _descriptors([RawTool(name="create_issue")])[0]
        self.assertTrue(requires_step_approval(desc))
        read_desc = _descriptors([RawTool(name="search_issues")])[0]
        self.assertFalse(requires_step_approval(read_desc))

    def test_pending_until_granted(self):
        urn = str(self.write_desc.urn)
        step_approval_store.grant("s1", "t1", urn)
        attachments = [
            ComposerAttachment(
                kind="capability",
                id=urn,
                label="Create issue",
            )
        ]
        pending = pending_step_approvals("s1", "t1", attachments)
        self.assertEqual(pending, [])
        pending2 = pending_step_approvals("s1", "t2", attachments)
        self.assertEqual(len(pending2), 1)
        msg = format_step_approval_message(pending2)
        self.assertIn("modify external data", msg)


class TestSessionEgressAndInvoke(_TmpRootTestCase):
    def setUp(self):
        super().setUp()
        reset_registry_for_tests()
        session_egress_ledger.clear_session("egress-sess")
        step_approval_store.clear_session("egress-sess")
        agent_scope_store.clear_session("egress-sess")

        self.read_desc = _descriptors([RawTool(name="search_issues")])[0]
        self.write_desc = _descriptors([RawTool(name="create_issue")])[0]
        register_capability_provider("fake", _FakeProvider)

        save_descriptor_cache("fake", [self.read_desc, self.write_desc])
        store = ConsentStore("fake")
        for desc in (self.read_desc, self.write_desc):
            store.grant(desc)

    def tearDown(self):
        reset_registry_for_tests()
        super().tearDown()

    def test_invoke_records_egress(self):
        attachments = [
            ComposerAttachment(
                kind="capability",
                id=str(self.read_desc.urn),
                label="Search",
            )
        ]
        scope = build_agent_scope_from_attachments("egress-sess", attachments)
        agent_scope_store.set_scope(scope)
        result = invoke_gated_capability(
            self.read_desc.urn,
            "query",
            session_id="egress-sess",
            turn_id="1",
            agent_scope=scope,
            provider_factory_kwargs={"descriptors": [self.read_desc, self.write_desc]},
        )
        self.assertTrue(result.allowed)
        records = session_egress_ledger.records_for_session("egress-sess")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].capability_group, self.read_desc.group)
        self.assertEqual(records[0].server_id, self.read_desc.urn.namespace)
        summary = format_session_egress_summary("egress-sess")
        self.assertIn("Session integrations", summary)
        privacy = format_privacy_report_integrations_section("egress-sess")
        self.assertIn("Integrations (this session)", privacy)

    def test_write_denied_without_step_approval(self):
        attachments = [
            ComposerAttachment(
                kind="capability",
                id=str(self.write_desc.urn),
                label="Create",
            )
        ]
        scope = build_agent_scope_from_attachments("egress-sess", attachments)
        result = invoke_gated_capability(
            self.write_desc.urn,
            "query",
            session_id="egress-sess",
            turn_id="2",
            agent_scope=scope,
            provider_factory_kwargs={"descriptors": [self.read_desc, self.write_desc]},
        )
        self.assertFalse(result.allowed)
        self.assertIn("step approval", result.reason)
        records = session_egress_ledger.records_for_session("egress-sess")
        self.assertTrue(any(not r.allowed for r in records))

    def test_write_allowed_after_step_approval(self):
        step_approval_store.grant(
            "egress-sess", "3", str(self.write_desc.urn)
        )
        scope = AgentScope(
            session_id="egress-sess",
            allowed_urn_bases=frozenset({urn_base_key(self.write_desc.urn)}),
        )
        result = invoke_gated_capability(
            self.write_desc.urn,
            "query",
            session_id="egress-sess",
            turn_id="3",
            agent_scope=scope,
            provider_factory_kwargs={"descriptors": [self.read_desc, self.write_desc]},
        )
        self.assertTrue(result.allowed)

    def test_out_of_scope_denied(self):
        scope = AgentScope(
            session_id="egress-sess",
            allowed_urn_bases=frozenset({urn_base_key(self.read_desc.urn)}),
        )
        result = invoke_gated_capability(
            self.write_desc.urn,
            "query",
            session_id="egress-sess",
            turn_id="4",
            agent_scope=scope,
            provider_factory_kwargs={"descriptors": [self.read_desc, self.write_desc]},
        )
        self.assertFalse(result.allowed)
        self.assertIn("scope", result.reason)

    def test_preview_is_dry_run(self):
        provider = _FakeProvider(descriptors=[self.write_desc])
        register_capability_provider("fake", lambda **kw: provider)
        preview = preview_gated_capability(
            self.write_desc.urn,
            "query",
            provider_factory_kwargs={"descriptors": [self.write_desc]},
        )
        self.assertTrue(preview.dry_run)
        self.assertEqual(provider.invoke_calls, 0)


class TestPhase3P6Guardrail(unittest.TestCase):
    def test_new_modules_p6_clean(self):
        root = Path(__file__).resolve().parents[1] / "core" / "integrations"
        targets = [
            root / "agent_scope.py",
            root / "step_approval.py",
            root / "session_egress.py",
            root / "egress_summary.py",
            root / "composer_capability_gate.py",
        ]
        for path in targets:
            text = path.read_text(encoding="utf-8")
            for pattern in _P6_PATTERNS:
                self.assertIsNone(
                    pattern.search(text),
                    f"P6 violation in {path.name}: {pattern.pattern}",
                )


class TestComposerGate(_TmpRootTestCase):
    def test_capabilities_requiring_step_approval_from_cache(self):
        write_desc = _descriptors([RawTool(name="create_issue")])[0]
        save_descriptor_cache("fake", [write_desc])
        attachments = [
            ComposerAttachment(
                kind="capability",
                id=str(write_desc.urn),
                label="Create",
            )
        ]
        items = capabilities_requiring_step_approval(attachments)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].tier, CapabilityTier.WRITE.value)


if __name__ == "__main__":
    unittest.main()
