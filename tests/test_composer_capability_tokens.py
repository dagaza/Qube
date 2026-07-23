"""T14 — composer cap token spine + invoke gate (Phase 2 / #60 slice 1)."""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.composer_attachments import (
    ComposerAttachment,
    format_token,
    parse_attachments,
    resolve_attachment_routing,
)
from core.integrations.capabilities import persistence as P
from core.integrations.capabilities.mapper import CapabilityMapper, RawTool
from core.integrations.capabilities.model import CapabilityTier, NormalizedHit
from core.integrations.capabilities.persistence import ConsentStore, evaluate_access, save_descriptor_cache
from core.integrations.capabilities.urn import CapabilityURN
from core.integrations.capability_invoke import (
    evaluate_invoke_access,
    invoke_gated_capability,
    parse_composer_capability_urn,
)
from core.integrations.registry.provider_registry import (
    register_capability_provider,
    reset_registry_for_tests,
)
from core.knowledge.connectors.mcp_connector import McpConnector

_INVOKE_SRC = Path(__file__).resolve().parents[1] / "core" / "integrations" / "capability_invoke.py"
_P6_PATTERNS = (
    re.compile(r"\bimport\s+mcp\b"),
    re.compile(r"\bfrom\s+mcp\b"),
    re.compile(r"provider\s*==\s*['\"]mcp['\"]"),
)


def _descriptors(tools):
    group = CapabilityMapper().map_tools("fake", "docs", tools)
    return list(group.capabilities)


class _FakeProvider:
    provider_id = "fake"

    def __init__(self, **config):
        self.config = config
        self._descriptors = list(config.get("descriptors") or [])

    async def discover(self):
        return self._descriptors

    async def invoke(self, urn, args, *, ctx):
        return [
            NormalizedHit(
                title="Hit",
                snippet=f"answer for {args.get('query', '')}",
                source_cap=urn,
                url="https://example.test/hit",
            )
        ]

    async def health(self):
        from core.integrations.capabilities import HealthState, HealthStatus

        return HealthStatus(state=HealthState.OK)

    def fingerprint(self):
        return "fake-fp"

    def close(self):
        return None


class _TmpRootTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self._root = Path(self._tmp.name)
        self._orig = P.user_data_root
        P.user_data_root = lambda: self._root  # type: ignore[assignment]

    def tearDown(self):
        P.user_data_root = self._orig  # type: ignore[assignment]
        self._tmp.cleanup()


class TestCapTokenParse(unittest.TestCase):
    def test_parse_valid_capability_token(self):
        text = "Find issues @[cap:mcp:github/search-issues] please"
        clean, attachments = parse_attachments(text)
        self.assertEqual(clean, "Find issues please")
        self.assertEqual(len(attachments), 1)
        self.assertEqual(attachments[0].kind, "capability")
        self.assertEqual(
            attachments[0].id, "cap:mcp:github/search-issues"
        )

    def test_parse_versioned_capability_token(self):
        _, attachments = parse_attachments("@[cap:mcp:github/search-issues@2] q")
        self.assertEqual(len(attachments), 1)
        self.assertEqual(
            attachments[0].id, "cap:mcp:github/search-issues@2"
        )

    def test_malformed_capability_token_fail_closed(self):
        clean, attachments = parse_attachments("@[cap:not-a-valid-urn] hello")
        self.assertEqual(clean, "hello")
        self.assertEqual(attachments, [])

    def test_format_capability_token_roundtrip(self):
        urn = CapabilityURN.build("mcp", "github", "search-issues")
        att = ComposerAttachment(kind="capability", id=str(urn), label="github/search-issues")
        token = format_token(att)
        self.assertEqual(token, "@[cap:mcp:github/search-issues]")
        _, attachments = parse_attachments(token)
        self.assertEqual(attachments[0].id, str(urn))

    def test_parse_composer_capability_urn_accepts_bodies(self):
        self.assertEqual(
            str(parse_composer_capability_urn("mcp:github/search-issues")),
            "cap:mcp:github/search-issues",
        )
        self.assertIsNone(parse_composer_capability_urn("bad urn"))


class TestCapTokenRoute(unittest.TestCase):
    def test_resolve_capability_routing(self):
        att = ComposerAttachment(
            kind="capability",
            id="cap:mcp:github/search-issues",
            label="github/search-issues",
        )
        patch = resolve_attachment_routing([att])
        assert patch is not None
        self.assertEqual(patch["route"], "capability")
        self.assertEqual(patch["strategy"], "attachment_capability")
        self.assertEqual(patch["capability_urn"], "cap:mcp:github/search-issues")

    def test_capability_route_not_web(self):
        att = ComposerAttachment(
            kind="capability",
            id="cap:fake:docs/search-docs",
            label="docs/search-docs",
        )
        patch = resolve_attachment_routing([att])
        assert patch is not None
        self.assertNotEqual(patch["route"], "web")


class TestInvokeAccessGate(_TmpRootTestCase):
    def setUp(self):
        super().setUp()
        reset_registry_for_tests()
        self.addCleanup(reset_registry_for_tests)
        self.read = _descriptors([RawTool("search_docs", "Search", {"type": "object"})])[0]
        save_descriptor_cache("fake", [self.read])

    def test_compose_invoke_denies_without_grant(self):
        decision = evaluate_invoke_access(self.read, None)
        self.assertFalse(decision.allowed)

    def test_compose_invoke_allows_with_grant(self):
        grant = ConsentStore("fake").grant(self.read)
        decision = evaluate_invoke_access(self.read, grant)
        self.assertTrue(decision.allowed)

    def test_invoke_gated_denies_without_grant(self):
        result = invoke_gated_capability(
            "cap:fake:docs/search-docs",
            "query",
            live_descriptors=[self.read],
            provider_factory_kwargs={"descriptors": [self.read]},
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.rows, ())

    def test_invoke_gated_runs_with_grant(self):
        register_capability_provider("fake", _FakeProvider)
        ConsentStore("fake").grant(self.read)
        result = invoke_gated_capability(
            "cap:fake:docs/search-docs",
            "reactor safety",
            live_descriptors=[self.read],
            provider_factory_kwargs={"descriptors": [self.read]},
        )
        self.assertTrue(result.allowed)
        self.assertEqual(len(result.rows), 1)
        self.assertEqual(result.rows[0]["_adapter"], "docs")
        self.assertTrue(
            str(result.rows[0]["_capability"]).startswith("cap:fake:docs/search-docs")
        )


class TestMcpConnectorConsentAlignment(_TmpRootTestCase):
    def setUp(self):
        super().setUp()
        self.read = _descriptors([RawTool("search_docs", "Search", {"type": "object"})])[0]
        self.write = _descriptors([RawTool("create_doc", "Create", {"type": "object"})])[0]

    def test_read_uses_ephemeral_grant_when_none_stored(self):
        allowed = McpConnector._is_permitted(
            self.read,
            "mcp",
            evaluate_access,
            ConsentStore,
            CapabilityTier,
        )
        self.assertTrue(allowed)

    def test_explicit_deny_blocks_read(self):
        store = ConsentStore("mcp")
        store.deny(self.read)
        allowed = McpConnector._is_permitted(
            self.read,
            "mcp",
            evaluate_access,
            ConsentStore,
            CapabilityTier,
        )
        self.assertFalse(allowed)

    def test_write_still_default_denied_without_grant(self):
        allowed = McpConnector._is_permitted(
            self.write,
            "mcp",
            evaluate_access,
            ConsentStore,
            CapabilityTier,
        )
        self.assertFalse(allowed)


class TestSlice1P6Guardrail(unittest.TestCase):
    def test_capability_invoke_module_is_p6_clean(self):
        src = _INVOKE_SRC.read_text(encoding="utf-8")
        for pat in _P6_PATTERNS:
            self.assertIsNone(
                pat.search(src),
                f"capability_invoke trips P6 guardrail pattern {pat.pattern!r}",
            )


if __name__ == "__main__":
    unittest.main()
