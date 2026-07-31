"""T11 — the provider-agnostic capability provider registry.

Covers register/resolve/create/list by string id, unknown-id handling, and the
load-bearing P5/P6 invariant: the registry core resolves providers **without**
importing a concrete provider. The only module allowed to import a provider is
the composition root (``core/integrations/providers/__init__.py``), and it is
what registers the built-in MCP provider by id.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from core.integrations.capabilities import CapabilityProvider
from core.integrations.registry import (
    UnknownCapabilityProvider,
    create_capability_provider,
    get_capability_provider_factory,
    is_provider_registered,
    list_capability_providers,
    register_capability_provider,
    reset_registry_for_tests,
)
from core.integrations.registry import provider_registry

_REGISTRY_SRC = Path(provider_registry.__file__)
_PROVIDERS_INIT = (
    Path(__file__).resolve().parent.parent
    / "core"
    / "integrations"
    / "providers"
    / "__init__.py"
)


class _FakeProvider:
    """Minimal object standing in for a concrete provider factory result."""

    def __init__(self, *, provider_id: str = "fake", **config) -> None:
        self.provider_id = provider_id
        self.config = config


class TestRegistryContract(unittest.TestCase):
    def setUp(self) -> None:
        reset_registry_for_tests()
        self.addCleanup(reset_registry_for_tests)

    def test_register_get_and_create_by_id(self):
        register_capability_provider("fake", _FakeProvider)
        self.assertTrue(is_provider_registered("fake"))
        self.assertIs(get_capability_provider_factory("fake"), _FakeProvider)

        instance = create_capability_provider("fake", token="abc")
        self.assertIsInstance(instance, _FakeProvider)
        # Per-instance config is passed straight through to the factory.
        self.assertEqual(instance.config, {"token": "abc"})

    def test_provider_id_is_normalized(self):
        register_capability_provider("  Fake  ", _FakeProvider)
        self.assertTrue(is_provider_registered("fake"))
        self.assertTrue(is_provider_registered("FAKE"))
        self.assertIn("fake", list_capability_providers())

    def test_unknown_provider_raises(self):
        with self.assertRaises(UnknownCapabilityProvider) as ctx:
            create_capability_provider("does-not-exist")
        self.assertEqual(ctx.exception.provider_id, "does-not-exist")
        self.assertIsNone(get_capability_provider_factory("nope"))
        self.assertFalse(is_provider_registered("nope"))

    def test_register_rejects_empty_id_and_non_callable(self):
        with self.assertRaises(ValueError):
            register_capability_provider("", _FakeProvider)
        with self.assertRaises(TypeError):
            register_capability_provider("bad", object())  # type: ignore[arg-type]

    def test_register_overwrites_idempotently(self):
        register_capability_provider("fake", _FakeProvider)
        register_capability_provider("fake", _FakeProvider)
        self.assertEqual(
            [p for p in list_capability_providers() if p == "fake"], ["fake"]
        )


class TestBuiltinRegistration(unittest.TestCase):
    """The composition root registers the real MCP provider by id."""

    def setUp(self) -> None:
        reset_registry_for_tests()
        self.addCleanup(reset_registry_for_tests)

    def test_mcp_provider_is_registered_by_id(self):
        # A bare lookup must lazily trigger the composition root.
        self.assertIn("mcp", list_capability_providers())
        self.assertTrue(is_provider_registered("mcp"))

    def test_created_mcp_provider_satisfies_protocol(self):
        provider = create_capability_provider(
            "mcp", namespace="docs", command=["python", "-c", "pass"]
        )
        # Resolved purely by id, yet satisfies the provider-agnostic contract.
        self.assertIsInstance(provider, CapabilityProvider)
        self.assertEqual(provider.provider_id, "mcp")
        # Clean up the (unconnected) transport if the provider exposes close().
        close = getattr(provider, "close", None)
        if callable(close):
            close()

    def test_reset_then_reload_reregisters_builtins(self):
        self.assertIn("mcp", list_capability_providers())
        reset_registry_for_tests()
        # After a reset the next lookup must reload the builtins.
        self.assertIn("mcp", list_capability_providers())


# The exact patterns the Starfall P6 guardrail (.cursor/starfall/verify/mcp.py)
# scans for anywhere under core/integrations/ outside providers/mcp/.
_P6_PATTERNS = (
    re.compile(r"\bimport\s+mcp\b"),
    re.compile(r"\bfrom\s+mcp\b"),
    re.compile(r"provider\s*==\s*['\"]mcp['\"]"),
)


class TestProviderAgnosticInvariant(unittest.TestCase):
    """P5/P6: the registry core never imports a concrete provider."""

    def test_registry_source_is_p6_clean(self):
        # Mirror the real guardrail: the registry module (docstring included)
        # must contain no MCP import and no provider-equality branch.
        src = _REGISTRY_SRC.read_text(encoding="utf-8")
        for pat in _P6_PATTERNS:
            self.assertIsNone(
                pat.search(src),
                f"registry core trips P6 guardrail pattern {pat.pattern!r}",
            )

    def test_only_composition_root_imports_the_provider(self):
        root = _PROVIDERS_INIT.read_text(encoding="utf-8")
        # The composition root is the one place allowed to import a provider.
        self.assertIn("McpCapabilityProvider", root)
        # ...and it does so via the provider subpackage, not the raw MCP SDK, so
        # it must not trip the guardrail's import patterns either.
        for pat in _P6_PATTERNS[:2]:
            self.assertIsNone(
                pat.search(root),
                f"composition root trips P6 guardrail pattern {pat.pattern!r}",
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
