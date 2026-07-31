"""T8 — the MCP CapabilityProvider client.

Two layers:

* Fast, deterministic contract tests against an in-memory fake transport
  (handshake ordering, tier mapping, provenance, dry-run, foreign/unknown URN,
  timeout, health).
* Real stdio tests that spawn ``tests/fixtures/mock_mcp_server.py`` with
  ``sys.executable`` to prove the hand-rolled NDJSON :class:`StdioTransport`
  actually performs the ``initialize`` -> ``tools/list`` -> ``tools/call``
  lifecycle and honours a per-request timeout.
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from core.integrations.capabilities import (
    CapabilityInvocationError,
    CapabilityProvider,
    CapabilityTier,
    CapabilityURN,
    HealthState,
    InvokeContext,
)
from core.integrations.providers.mcp import McpCapabilityProvider
from core.integrations.providers.mcp.transport import McpTimeoutError

_MOCK_SERVER = Path(__file__).resolve().parent / "fixtures" / "mock_mcp_server.py"

_FAKE_TOOLS = [
    {"name": "search_docs", "description": "Search", "inputSchema": {"type": "object"}},
    {"name": "create_doc", "description": "Create", "inputSchema": {"type": "object"}},
    {"name": "delete_doc", "description": "Delete", "inputSchema": {"type": "object"}},
]


class FakeTransport:
    """In-memory transport that records traffic and returns canned results."""

    def __init__(self, *, tools=None, timeout_on_call=False, call_result=None):
        self._connected = False
        self._tools = tools if tools is not None else _FAKE_TOOLS
        self._timeout_on_call = timeout_on_call
        self._call_result = call_result
        self.requests: list[tuple[str, dict | None]] = []
        self.notifications: list[tuple[str, dict | None]] = []
        self.closed = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._connected = True

    def request(self, method, params=None, *, timeout_s):
        self.requests.append((method, params))
        if method == "initialize":
            return {"protocolVersion": "2024-11-05"}
        if method == "tools/list":
            return {"tools": self._tools}
        if method == "tools/call":
            if self._timeout_on_call:
                raise McpTimeoutError("simulated timeout")
            return self._call_result or {
                "content": [
                    {"type": "text", "title": "T", "text": "hello world", "url": "https://x.test"}
                ]
            }
        return {}

    def notify(self, method, params=None) -> None:
        self.notifications.append((method, params))

    def close(self) -> None:
        self.closed = True
        self._connected = False


class TestProviderContractFake(unittest.TestCase):
    def _provider(self, **kw) -> McpCapabilityProvider:
        transport = kw.pop("transport", None) or FakeTransport(**kw)
        return McpCapabilityProvider(namespace="docs", transport=transport)

    def test_is_capability_provider(self):
        self.assertIsInstance(self._provider(), CapabilityProvider)

    def test_discover_handshake_order_and_tiers(self):
        transport = FakeTransport()
        provider = McpCapabilityProvider(namespace="docs", transport=transport)
        descriptors = asyncio.run(provider.discover())

        methods = [m for m, _ in transport.requests]
        self.assertEqual(methods[0], "initialize")
        self.assertEqual(methods[1], "tools/list")
        self.assertIn(("notifications/initialized", None), transport.notifications)

        by_action = {d.action: d for d in descriptors}
        self.assertEqual(by_action["search-docs"].tier, CapabilityTier.READ)
        self.assertEqual(by_action["create-doc"].tier, CapabilityTier.WRITE)
        self.assertEqual(by_action["delete-doc"].tier, CapabilityTier.DESTRUCTIVE)
        for d in descriptors:
            self.assertEqual(d.urn.provider, "mcp")
            self.assertEqual(d.urn.namespace, "docs")

    def test_invoke_returns_provenance_hits(self):
        provider = self._provider()
        asyncio.run(provider.discover())
        urn = CapabilityURN.build("mcp", "docs", "search-docs")
        hits = asyncio.run(provider.invoke(urn, {"query": "q"}, ctx=InvokeContext(query="q")))
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].source_cap, urn)
        ev = hits[0].to_evidence_dict()
        self.assertEqual(ev["_capability"], str(urn))
        self.assertEqual(ev["retrieval_method"], "mcp")

    def test_invoke_uses_raw_ref_not_slug(self):
        transport = FakeTransport()
        provider = McpCapabilityProvider(namespace="docs", transport=transport)
        asyncio.run(provider.discover())
        urn = CapabilityURN.build("mcp", "docs", "search-docs")
        asyncio.run(provider.invoke(urn, {"query": "q"}, ctx=InvokeContext(query="q")))
        call = [p for m, p in transport.requests if m == "tools/call"][0]
        self.assertEqual(call["name"], "search_docs")  # raw tool name, not "search-docs"

    def test_invoke_rejects_foreign_provider(self):
        provider = self._provider()
        asyncio.run(provider.discover())
        urn = CapabilityURN.build("other", "docs", "search-docs")
        with self.assertRaises(CapabilityInvocationError):
            asyncio.run(provider.invoke(urn, {}, ctx=InvokeContext(query="q")))

    def test_invoke_rejects_unknown_capability(self):
        provider = self._provider()
        asyncio.run(provider.discover())
        urn = CapabilityURN.build("mcp", "docs", "nonexistent")
        with self.assertRaises(CapabilityInvocationError):
            asyncio.run(provider.invoke(urn, {}, ctx=InvokeContext(query="q")))

    def test_dry_run_write_has_no_side_effect(self):
        transport = FakeTransport()
        provider = McpCapabilityProvider(namespace="docs", transport=transport)
        asyncio.run(provider.discover())
        urn = CapabilityURN.build("mcp", "docs", "create-doc")
        hits = asyncio.run(
            provider.invoke(urn, {"body": "x"}, ctx=InvokeContext(query="q", dry_run=True))
        )
        self.assertEqual(len(hits), 1)
        self.assertIn("dry-run", hits[0].title)
        # No tools/call was issued for the write capability under dry_run.
        self.assertNotIn("tools/call", [m for m, _ in transport.requests])

    def test_invoke_timeout_becomes_invocation_error(self):
        transport = FakeTransport(timeout_on_call=True)
        provider = McpCapabilityProvider(namespace="docs", transport=transport)
        asyncio.run(provider.discover())
        urn = CapabilityURN.build("mcp", "docs", "search-docs")
        with self.assertRaises(CapabilityInvocationError):
            asyncio.run(provider.invoke(urn, {}, ctx=InvokeContext(query="q", timeout_s=0.1)))

    def test_health_and_fingerprint(self):
        provider = self._provider()
        asyncio.run(provider.discover())
        health = asyncio.run(provider.health())
        self.assertEqual(health.state, HealthState.OK)
        fp = provider.fingerprint()
        self.assertEqual(len(fp), 64)

    def test_health_down_when_not_connected(self):
        provider = self._provider()  # never discovered/connected
        health = asyncio.run(provider.health())
        self.assertEqual(health.state, HealthState.DOWN)

    def test_normalize_splits_multiline_filesystem_search_paths(self):
        urn = CapabilityURN.build("mcp", "filesystem", "search-files")
        root = "/fixture/workspace"
        result = {
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"{root}/meeting-notes-alpha.txt\n"
                        f"{root}/design-spec-beta.txt\n"
                        f"{root}/summary-alpha.txt"
                    ),
                }
            ]
        }
        hits = McpCapabilityProvider._normalize(
            result,
            source_cap=urn,
            max_results=5,
        )
        self.assertEqual(len(hits), 3)
        self.assertEqual(hits[0].title, "meeting-notes-alpha.txt")
        self.assertEqual(hits[1].snippet, f"{root}/design-spec-beta.txt")

    def test_normalize_keeps_single_block_for_non_search_capabilities(self):
        urn = CapabilityURN.build("mcp", "docs", "read-doc")
        text = "line one\nline two"
        hits = McpCapabilityProvider._normalize(
            {"content": [{"text": text}]},
            source_cap=urn,
            max_results=5,
        )
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].snippet, text[:600])


@unittest.skipUnless(_MOCK_SERVER.exists(), "mock server fixture missing")
class TestProviderRealStdio(unittest.TestCase):
    def _provider(self) -> McpCapabilityProvider:
        return McpCapabilityProvider(
            namespace="docs",
            command=[sys.executable, str(_MOCK_SERVER)],
        )

    def test_discover_over_real_stdio(self):
        provider = self._provider()
        try:
            descriptors = asyncio.run(provider.discover())
        finally:
            provider.close()
        actions = {d.action for d in descriptors}
        self.assertIn("search-docs", actions)
        self.assertIn("create-doc", actions)

    def test_invoke_over_real_stdio(self):
        provider = self._provider()
        try:
            asyncio.run(provider.discover())
            urn = CapabilityURN.build("mcp", "docs", "search-docs")
            hits = asyncio.run(
                provider.invoke(urn, {"query": "reactor"}, ctx=InvokeContext(query="reactor"))
            )
        finally:
            provider.close()
        self.assertEqual(len(hits), 1)
        self.assertIn("reactor", hits[0].snippet)
        self.assertEqual(hits[0].to_evidence_dict()["_capability"], str(urn))

    def test_invoke_timeout_over_real_stdio(self):
        provider = self._provider()
        try:
            asyncio.run(provider.discover())
            urn = CapabilityURN.build("mcp", "docs", "search-slow")
            with self.assertRaises(CapabilityInvocationError):
                asyncio.run(
                    provider.invoke(urn, {"query": "x"}, ctx=InvokeContext(query="x", timeout_s=0.5))
                )
        finally:
            provider.close()


@unittest.skipUnless(_MOCK_SERVER.exists(), "mock server fixture missing")
class TestConnectorDelegation(unittest.TestCase):
    """The legacy McpConnector must delegate to the provider (single path)."""

    def setUp(self):
        from core.integrations.capabilities import persistence as P
        self._tmp = TemporaryDirectory()
        self._orig = P.user_data_root
        P.user_data_root = lambda: Path(self._tmp.name)  # type: ignore[assignment]
        self._P = P

    def tearDown(self):
        self._P.user_data_root = self._orig  # type: ignore[assignment]
        self._tmp.cleanup()

    def _config(self, tool_name):
        return {
            "command": [sys.executable, str(_MOCK_SERVER)],
            "tool_name": tool_name,
            "adapter_id": "my_mcp",
            "namespace": "docs",
        }

    def test_read_search_returns_capability_rows(self):
        from core.knowledge.connectors.mcp_connector import McpConnector
        rows = McpConnector().execute("reactor", config=self._config("search_docs"))
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["_capability"].startswith("cap:mcp:docs/search-docs"))
        self.assertEqual(rows[0]["_adapter"], "my_mcp")  # short id preserved (KI2)
        self.assertEqual(rows[0]["retrieval_method"], "mcp")

    def test_write_tool_is_default_denied(self):
        from core.knowledge.connectors.mcp_connector import McpConnector
        rows = McpConnector().execute("x", config=self._config("create_doc"))
        self.assertEqual(rows, [])  # write tier, no grant -> denied (P7)

    def test_test_connection_ok(self):
        from core.knowledge.connectors.mcp_connector import McpConnector
        ok, msg = McpConnector().test_connection(config=self._config("search_docs"))
        self.assertTrue(ok)
        self.assertIn("capabilities", msg)


if __name__ == "__main__":
    unittest.main()
