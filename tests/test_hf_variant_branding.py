"""Tests for GGUF variant (community quantizer) branding."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from core.hf_publisher_branding import (
    HuggingFaceBrandingResolver,
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

    def test_resolve_variant_uses_org_name_and_hf_fallback_logo(self) -> None:
        resolver = HuggingFaceBrandingResolver(timeout_s=1.0)
        with (
            mock.patch.object(resolver, "get_org_metadata", return_value={"fullname": "Unsloth AI"}),
            mock.patch.object(resolver, "get_user_metadata"),
            mock.patch.object(resolver, "_cache_avatar_file", return_value=None),
        ):
            out = resolver.resolve_variant_branding("unsloth/gemma-GGUF")
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out["name"], "Unsloth AI")
        self.assertEqual(out["owner"], "unsloth")
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
