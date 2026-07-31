"""T4 — a mock provider satisfies the CapabilityProvider contract.

The mock plays the role of a deterministic, network-free MCP server: it maps a
fixed raw tool surface to capabilities and returns normalized hits. It proves
that the runtime can depend solely on the protocol + value objects, with no
provider-specific import (P5/P6).
"""

import asyncio
import unittest

from core.integrations.capabilities.mapper import CapabilityMapper, RawTool
from core.integrations.capabilities.model import (
    CapabilityDescriptor,
    HealthState,
    HealthStatus,
    NormalizedHit,
    fingerprint_descriptors,
)
from core.integrations.capabilities.protocol import (
    CapabilityInvocationError,
    CapabilityProvider,
    InvokeContext,
)
from core.integrations.capabilities.urn import CapabilityURN

_RAW_TOOLS = [
    RawTool("search_issues", "Search GitHub issues", {"type": "object"}),
    RawTool("create_issue", "Create a GitHub issue", {"type": "object"}),
    RawTool("delete_branch", "Delete a branch", {"type": "object"}),
]


class MockProvider:
    """A minimal, deterministic CapabilityProvider (no network)."""

    provider_id = "mock"

    def __init__(self) -> None:
        self._descriptors: list[CapabilityDescriptor] = []

    async def discover(self) -> list[CapabilityDescriptor]:
        group = CapabilityMapper().map_tools(self.provider_id, "github", _RAW_TOOLS)
        self._descriptors = list(group.capabilities)
        return self._descriptors

    async def invoke(self, urn, args, *, ctx: InvokeContext) -> list[NormalizedHit]:
        if urn.provider != self.provider_id:
            raise CapabilityInvocationError(f"{urn} does not belong to {self.provider_id}")
        return [
            NormalizedHit(
                title=f"result for {urn.action}",
                snippet=ctx.query,
                source_cap=urn,
            )
        ]

    async def health(self) -> HealthStatus:
        return HealthStatus(state=HealthState.OK, latency_ms=1.0)

    def fingerprint(self) -> str:
        return fingerprint_descriptors(self._descriptors)


class TestCapabilityProviderContract(unittest.TestCase):
    def setUp(self):
        self.provider = MockProvider()

    def test_isinstance_runtime_checkable(self):
        self.assertIsInstance(self.provider, CapabilityProvider)

    def test_discover_maps_tiers(self):
        descriptors = asyncio.run(self.provider.discover())
        by_action = {d.action: d for d in descriptors}
        self.assertEqual(by_action["search-issues"].tier.value, "read")
        self.assertEqual(by_action["create-issue"].tier.value, "write")
        self.assertEqual(by_action["delete-branch"].tier.value, "destructive")
        for d in descriptors:
            self.assertEqual(d.urn.provider, "mock")

    def test_invoke_returns_provenance_bearing_hits(self):
        asyncio.run(self.provider.discover())
        urn = CapabilityURN.build("mock", "github", "search-issues")
        ctx = InvokeContext(query="crash on export")
        hits = asyncio.run(self.provider.invoke(urn, {}, ctx=ctx))
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].source_cap, urn)
        self.assertEqual(hits[0].to_evidence_dict()["_capability"], str(urn))

    def test_invoke_rejects_foreign_urn(self):
        urn = CapabilityURN.build("other", "github", "search-issues")
        with self.assertRaises(CapabilityInvocationError):
            asyncio.run(self.provider.invoke(urn, {}, ctx=InvokeContext(query="q")))

    def test_health_and_fingerprint(self):
        asyncio.run(self.provider.discover())
        self.assertEqual(asyncio.run(self.provider.health()).state, HealthState.OK)
        fp = self.provider.fingerprint()
        self.assertIsInstance(fp, str)
        self.assertEqual(len(fp), 64)  # sha256 hex

    def test_incomplete_provider_is_not_instance(self):
        class NotAProvider:
            provider_id = "nope"

        self.assertNotIsInstance(NotAProvider(), CapabilityProvider)


if __name__ == "__main__":
    unittest.main()
