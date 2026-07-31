"""T1 — CapabilityURN parse/build/round-trip and malformed rejection."""

import unittest

from core.integrations.capabilities.urn import CapabilityURN, InvalidCapabilityURN


class TestCapabilityURN(unittest.TestCase):
    def test_parse_round_trip(self):
        raw = "cap:mcp:github/search-issues"
        urn = CapabilityURN.parse(raw)
        self.assertEqual(urn.provider, "mcp")
        self.assertEqual(urn.namespace, "github")
        self.assertEqual(urn.action, "search-issues")
        self.assertIsNone(urn.version)
        self.assertEqual(str(urn), raw)

    def test_parse_with_version_round_trip(self):
        raw = "cap:mcp:github/search-issues@2"
        urn = CapabilityURN.parse(raw)
        self.assertEqual(urn.version, "2")
        self.assertTrue(urn.is_versioned)
        self.assertEqual(str(urn), raw)

    def test_build_matches_parse(self):
        built = CapabilityURN.build("live", "pubmed", "search")
        parsed = CapabilityURN.parse("cap:live:pubmed/search")
        self.assertEqual(built, parsed)
        self.assertEqual(hash(built), hash(parsed))

    def test_base_strips_version(self):
        urn = CapabilityURN.parse("cap:mcp:github/search-issues@2")
        self.assertEqual(str(urn.base), "cap:mcp:github/search-issues")
        self.assertIsNone(urn.base.version)
        self.assertEqual(urn.base.base, urn.base)

    def test_with_version(self):
        urn = CapabilityURN.build("mcp", "github", "search-issues")
        pinned = urn.with_version("2")
        self.assertEqual(str(pinned), "cap:mcp:github/search-issues@2")
        self.assertEqual(str(pinned.with_version(None)), "cap:mcp:github/search-issues")

    def test_value_equality_and_hashable(self):
        a = CapabilityURN.build("mcp", "github", "search-issues")
        b = CapabilityURN.build("mcp", "github", "search-issues")
        self.assertEqual(a, b)
        self.assertEqual(len({a, b}), 1)
        self.assertIn(a, {b: 1})

    def test_try_parse_returns_none_on_bad_input(self):
        self.assertIsNone(CapabilityURN.try_parse("not-a-urn"))
        self.assertIsNotNone(CapabilityURN.try_parse("cap:mcp:github/search"))

    def test_malformed_rejected(self):
        bad_values = [
            "",
            "cap:",
            "cap:mcp",
            "cap:mcp:github",
            "cap:mcp:github/",
            "cap:mcp:/search",
            "cap::github/search",
            "github/search",
            "cap:MCP:github/search",          # uppercase provider
            "cap:mcp:github/search-issues@",   # empty version
            "cap:mcp:git hub/search",          # space in namespace
            "cap:mcp:github/search/extra",     # too many path segments
        ]
        for value in bad_values:
            with self.subTest(value=value):
                with self.assertRaises(InvalidCapabilityURN):
                    CapabilityURN.parse(value)

    def test_build_rejects_invalid_parts(self):
        with self.assertRaises(InvalidCapabilityURN):
            CapabilityURN.build("mcp", "github", "Search Issues")
        with self.assertRaises(InvalidCapabilityURN):
            CapabilityURN.build("", "github", "search")

    def test_parse_rejects_non_string(self):
        with self.assertRaises(InvalidCapabilityURN):
            CapabilityURN.parse(None)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
