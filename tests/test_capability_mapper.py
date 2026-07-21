"""T6 — CapabilityMapper: tier heuristics, least-privilege default, URN validity."""

import unittest

from core.integrations.capabilities.mapper import (
    CapabilityMapper,
    CapabilityMappingError,
    RawTool,
)
from core.integrations.capabilities.model import CapabilityTier
from core.integrations.capabilities.urn import CapabilityURN


class TestClassifyTier(unittest.TestCase):
    def test_read_verbs(self):
        for name in ("search_issues", "read_file", "list-repos", "getUser", "query.rows"):
            tier, needs_review = CapabilityMapper.classify_tier(name)
            with self.subTest(name=name):
                self.assertEqual(tier, CapabilityTier.READ)
                self.assertFalse(needs_review)

    def test_write_verbs(self):
        for name in ("create_issue", "update_record", "post_comment", "uploadFile"):
            tier, needs_review = CapabilityMapper.classify_tier(name)
            with self.subTest(name=name):
                self.assertEqual(tier, CapabilityTier.WRITE)
                self.assertFalse(needs_review)

    def test_destructive_verbs(self):
        for name in ("delete_branch", "merge_pull_request", "exec_command", "drop_table"):
            tier, needs_review = CapabilityMapper.classify_tier(name)
            with self.subTest(name=name):
                self.assertEqual(tier, CapabilityTier.DESTRUCTIVE)
                self.assertFalse(needs_review)

    def test_highest_privilege_verb_wins(self):
        tier, needs_review = CapabilityMapper.classify_tier("get_and_delete_item")
        self.assertEqual(tier, CapabilityTier.DESTRUCTIVE)
        self.assertFalse(needs_review)

    def test_unknown_verb_defaults_to_destructive_and_needs_review(self):
        # P7: an unrecognised action must NOT be silently classified as read.
        for name in ("frobnicate_widget", "xyzzy", "quux2000"):
            tier, needs_review = CapabilityMapper.classify_tier(name)
            with self.subTest(name=name):
                self.assertEqual(tier, CapabilityTier.DESTRUCTIVE)
                self.assertTrue(needs_review)


class TestMapTools(unittest.TestCase):
    def setUp(self):
        self.mapper = CapabilityMapper()
        self.tools = [
            RawTool("search_issues", "Search issues", {"type": "object"}),
            RawTool("create_issue", "Create an issue"),
            RawTool("delete_branch", "Delete a branch"),
            RawTool("frobnicate", "Mystery tool"),
        ]

    def test_group_shape(self):
        group = self.mapper.map_tools("mcp", "github", self.tools, group_label="GitHub")
        self.assertEqual(group.provider_id, "mcp")
        self.assertEqual(group.name, "GitHub")
        self.assertEqual(len(group.capabilities), 4)

    def test_descriptors_valid_urns_and_raw_ref(self):
        group = self.mapper.map_tools("mcp", "github", self.tools)
        by_ref = {d.raw_ref: d for d in group.capabilities}
        self.assertEqual(str(by_ref["search_issues"].urn), "cap:mcp:github/search-issues")
        # raw tool id is retained for the Advanced view only.
        self.assertEqual(by_ref["create_issue"].raw_ref, "create_issue")
        # Every emitted URN round-trips through the grammar.
        for d in group.capabilities:
            self.assertEqual(CapabilityURN.parse(str(d.urn)), d.urn)

    def test_unknown_tool_flagged(self):
        group = self.mapper.map_tools("mcp", "github", self.tools)
        frob = next(d for d in group.capabilities if d.raw_ref == "frobnicate")
        self.assertEqual(frob.tier, CapabilityTier.DESTRUCTIVE)
        self.assertTrue(frob.needs_review)

    def test_manifest_override_wins(self):
        group = self.mapper.map_tools(
            "mcp",
            "github",
            [RawTool("frobnicate", "Mystery tool")],
            tier_overrides={"frobnicate": CapabilityTier.READ},
        )
        cap = group.capabilities[0]
        self.assertEqual(cap.tier, CapabilityTier.READ)
        self.assertFalse(cap.needs_review)

    def test_messy_tool_name_slugified_to_valid_urn(self):
        group = self.mapper.map_tools("mcp", "File System", [RawTool("Read.File v2")])
        cap = group.capabilities[0]
        self.assertEqual(CapabilityURN.parse(str(cap.urn)), cap.urn)
        self.assertEqual(cap.tier, CapabilityTier.READ)


class TestSlugConsistency(unittest.TestCase):
    """L1 — snake/kebab/dotted/camelCase names normalise to the same action id."""

    def test_camelcase_matches_snake_and_dotted(self):
        mapper = CapabilityMapper()
        for name in ("search_issues", "searchIssues", "search.issues", "search-issues"):
            group = mapper.map_tools("mcp", "github", [RawTool(name)])
            with self.subTest(name=name):
                self.assertEqual(str(group.capabilities[0].urn), "cap:mcp:github/search-issues")


class TestUrnCollision(unittest.TestCase):
    """M1 — tools that normalise to the same action must not shadow each other."""

    def test_collision_disambiguated_and_flagged(self):
        mapper = CapabilityMapper()
        group = mapper.map_tools(
            "mcp",
            "github",
            [RawTool("search_issues"), RawTool("searchIssues"), RawTool("search.issues")],
        )
        urns = [str(d.urn) for d in group.capabilities]
        # All three survive with distinct URNs (no silent shadowing).
        self.assertEqual(len(set(urns)), 3)
        self.assertEqual(urns[0], "cap:mcp:github/search-issues")
        self.assertEqual(urns[1], "cap:mcp:github/search-issues-2")
        self.assertEqual(urns[2], "cap:mcp:github/search-issues-3")
        # Each keeps its own raw tool id so invocation still routes correctly.
        self.assertEqual([d.raw_ref for d in group.capabilities],
                         ["search_issues", "searchIssues", "search.issues"])
        # The disambiguated ones are flagged for human review.
        self.assertFalse(group.capabilities[0].needs_review)
        self.assertTrue(group.capabilities[1].needs_review)
        self.assertTrue(group.capabilities[2].needs_review)

    def test_disambiguation_skips_existing_suffix(self):
        mapper = CapabilityMapper()
        group = mapper.map_tools(
            "mcp", "github",
            [RawTool("get_x"), RawTool("get.x"), RawTool("get-x-2")],
        )
        urns = [str(d.urn) for d in group.capabilities]
        # All three get unique ids; the literal "get-x-2" (whose natural slug was
        # already claimed by disambiguating "get.x") is pushed to "get-x-2-2"
        # rather than clobbering it.
        self.assertEqual(len(set(urns)), 3)
        self.assertEqual([u.split("/")[-1] for u in urns],
                         ["get-x", "get-x-2", "get-x-2-2"])


class TestNamespaceValidation(unittest.TestCase):
    """L2 — an un-sluggable namespace fails fast with a clear error."""

    def test_empty_namespace_raises_mapping_error(self):
        mapper = CapabilityMapper()
        with self.assertRaises(CapabilityMappingError):
            mapper.map_tools("mcp", "???", [RawTool("read_x")])
        with self.assertRaises(CapabilityMappingError):
            mapper.map_tool("mcp", "   ", RawTool("read_x"))

    def test_valid_namespace_slugified(self):
        mapper = CapabilityMapper()
        group = mapper.map_tools("mcp", "File System", [RawTool("read_file")])
        self.assertEqual(str(group.capabilities[0].urn), "cap:mcp:file-system/read-file")


if __name__ == "__main__":
    unittest.main()
