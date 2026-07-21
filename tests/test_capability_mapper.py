"""T6 — CapabilityMapper: tier heuristics, least-privilege default, URN validity."""

import unittest

from core.integrations.capabilities.mapper import CapabilityMapper, RawTool
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


if __name__ == "__main__":
    unittest.main()
