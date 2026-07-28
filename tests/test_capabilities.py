"""Tests for core.capabilities (Phase 1.1–1.2)."""

from __future__ import annotations

import unittest

from core.capabilities import (
    ALL_CAPABILITY_IDS,
    CAPABILITY_SPECS_BY_ID,
    CapabilityRequiredError,
    EditionTier,
    FEATURE_CAPABILITY_REGISTRY,
    capabilities_for_tier,
    get_resolved_capabilities,
    has_capability,
    has_feature,
    invalidate_capabilities_cache,
    require_capability,
    require_feature,
    resolve_capabilities,
    tier_includes,
)


class TestEditionCapabilities(unittest.TestCase):
    def setUp(self) -> None:
        invalidate_capabilities_cache()

    def tearDown(self) -> None:
        invalidate_capabilities_cache()

    def test_registry_covers_all_specs(self) -> None:
        self.assertEqual(set(CAPABILITY_SPECS_BY_ID.keys()), set(ALL_CAPABILITY_IDS))

    def test_feature_registry_points_at_known_capabilities(self) -> None:
        for feature_id, cap_id in FEATURE_CAPABILITY_REGISTRY.items():
            self.assertIn(
                cap_id,
                ALL_CAPABILITY_IDS,
                msg=f"feature {feature_id} -> unknown capability {cap_id}",
            )

    def test_mit_launch_grants_all_capabilities(self) -> None:
        caps = resolve_capabilities()
        self.assertEqual(caps.tier, EditionTier.HOME)
        self.assertEqual(caps.source, "mit_launch")
        self.assertTrue(all(caps.flags.values()))
        self.assertEqual(len(caps.granted_capability_ids()), len(ALL_CAPABILITY_IDS))

    def test_require_capability_does_not_raise_under_mit_launch(self) -> None:
        require_capability("team.policy")
        require_feature("policy.org_profile_enforce")

    def test_has_helpers_under_mit_launch(self) -> None:
        self.assertTrue(has_capability("pro.theme_packs"))
        self.assertTrue(has_feature("theme_pack.import_official"))

    def test_unknown_capability_and_feature_ids_raise(self) -> None:
        with self.assertRaises(KeyError):
            has_capability("pro.unknown_thing")
        with self.assertRaises(KeyError):
            has_feature("feature.does_not_exist")

    def test_tier_includes_ordering(self) -> None:
        self.assertTrue(tier_includes(EditionTier.PRO, EditionTier.TEAM))
        self.assertFalse(tier_includes(EditionTier.TEAM, EditionTier.PRO))
        self.assertTrue(tier_includes(EditionTier.HOME, EditionTier.ENTERPRISE))

    def test_capabilities_for_tier_home_has_no_paid_flags(self) -> None:
        flags = capabilities_for_tier(EditionTier.HOME)
        self.assertFalse(any(flags.values()))

    def test_capabilities_for_tier_pro_includes_pro_not_team(self) -> None:
        flags = capabilities_for_tier(EditionTier.PRO)
        self.assertTrue(flags["pro.theme_packs"])
        self.assertFalse(flags["team.policy"])

    def test_capabilities_for_tier_team_includes_pro_and_team(self) -> None:
        flags = capabilities_for_tier(EditionTier.TEAM)
        self.assertTrue(flags["pro.theme_packs"])
        self.assertTrue(flags["team.policy"])
        self.assertFalse(flags["enterprise.sso"])

    def test_entitlement_overrides(self) -> None:
        caps = resolve_capabilities(
            tier=EditionTier.HOME,
            entitlement_overrides={"pro.theme_packs": True, "team.policy": False},
            source="test",
        )
        self.assertTrue(caps.has("pro.theme_packs"))
        self.assertFalse(caps.has("team.policy"))
        self.assertTrue(caps.has("pro.knowledge_packs_official"))

    def test_get_resolved_capabilities_is_cached(self) -> None:
        first = get_resolved_capabilities()
        second = get_resolved_capabilities()
        self.assertIs(first, second)
        invalidate_capabilities_cache()
        third = get_resolved_capabilities()
        self.assertIsNot(first, third)

    def test_require_capability_raises_when_denied(self) -> None:
        caps = resolve_capabilities(
            tier=EditionTier.HOME,
            entitlement_overrides={"team.policy": False},
            source="test",
        )
        invalidate_capabilities_cache()
        from core import capabilities as mod

        original = mod._MIT_LAUNCH_GRANTS_ALL
        mod._MIT_LAUNCH_GRANTS_ALL = False
        mod._resolved_cache = caps
        try:
            with self.assertRaises(CapabilityRequiredError) as ctx:
                require_capability("team.policy", feature_id="policy.org_profile_enforce")
            self.assertEqual(ctx.exception.capability_id, "team.policy")
            self.assertEqual(ctx.exception.feature_id, "policy.org_profile_enforce")
        finally:
            mod._MIT_LAUNCH_GRANTS_ALL = original
            invalidate_capabilities_cache()


if __name__ == "__main__":
    unittest.main()
