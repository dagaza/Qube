"""Tests for GGUF variant (community quantizer) branding."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from core.hf_publisher_branding import (
    COMMUNITY_PUBLISHER_LOGOS,
    HuggingFaceBrandingResolver,
    _community_logo_asset_path,
    _humanize_owner,
    _owner_from_repo_id,
)


class TestVariantBranding(unittest.TestCase):
    def test_owner_from_repo_id(self) -> None:
        self.assertEqual(_owner_from_repo_id("unsloth/foo-GGUF"), "unsloth")
        self.assertEqual(_owner_from_repo_id("no-slash"), "")

    def test_humanize_owner(self) -> None:
        self.assertEqual(_humanize_owner("unsloth"), "Unsloth")
        self.assertEqual(_humanize_owner("davidau"), "DavidAU")

    def test_community_logo_unsloth(self) -> None:
        self.assertIn("unsloth", COMMUNITY_PUBLISHER_LOGOS)
        path = _community_logo_asset_path("unsloth")
        self.assertIsNotNone(path)
        assert path is not None
        self.assertIn("unsloth.png", path)

    def test_resolve_variant_bundled_community_skips_hf_api(self) -> None:
        resolver = HuggingFaceBrandingResolver(timeout_s=1.0)
        with (
            mock.patch.object(resolver, "get_org_metadata") as org,
            mock.patch.object(resolver, "get_user_metadata") as user,
        ):
            out = resolver.resolve_variant_branding("unsloth/gemma-GGUF")
        org.assert_not_called()
        user.assert_not_called()
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out["name"], "Unsloth")
        self.assertEqual(out["owner"], "unsloth")
        self.assertIn("unsloth.png", out["logo"])

    def test_resolve_variant_uses_org_name_and_hf_fallback_logo(self) -> None:
        resolver = HuggingFaceBrandingResolver(timeout_s=1.0)
        with (
            mock.patch.object(resolver, "get_org_metadata", return_value={"fullname": "Bartowski"}),
            mock.patch.object(resolver, "get_user_metadata"),
            mock.patch.object(resolver, "_cache_avatar_file", return_value=None),
        ):
            out = resolver.resolve_variant_branding("bartowski/llama-GGUF")
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out["name"], "Bartowski")
        self.assertEqual(out["owner"], "bartowski")
        self.assertTrue(out["logo"].endswith("hf-logo.svg") or "/hf-logo.svg" in out["logo"])

    def test_resolve_variant_cached_avatar_path(self) -> None:
        resolver = HuggingFaceBrandingResolver(timeout_s=1.0)
        avatar = Path("/tmp/qube-test-avatar.png")
        with (
            mock.patch.object(resolver, "get_org_metadata", return_value={"avatarUrl": "https://hf.co/x.png"}),
            mock.patch.object(resolver, "_cache_avatar_file", return_value=avatar),
        ):
            out = resolver.resolve_variant_branding("bartowski/llama-GGUF")
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out["logo"], str(avatar))

    def test_variant_row_predicate_catalog_vs_gguf_owner(self) -> None:
        catalog_publisher = "google"
        gguf_repo = "unsloth/gemma-GGUF"
        gguf_owner = _owner_from_repo_id(gguf_repo)
        show = bool(
            catalog_publisher
            and gguf_repo
            and gguf_owner
            and catalog_publisher != gguf_owner
        )
        self.assertTrue(show)
        self.assertFalse(
            _owner_from_repo_id("google/gemma-GGUF") != "google"
            if catalog_publisher == "google"
            else False
        )


if __name__ == "__main__":
    unittest.main()
