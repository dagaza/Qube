"""Tests for Pro Share themes license helpers."""

from __future__ import annotations

import pytest

from core.capabilities import CapabilityRequiredError, invalidate_capabilities_cache
from core import capabilities as capabilities_mod
from core.theme_pro_features import (
    LICENSE_REQUIRED_MESSAGE,
    PRO_SHARE_THEMES_FEATURE,
    PRO_THEME_PACKS_CAPABILITY,
    require_pro_share_themes,
    sync_share_themes_pro_features,
    user_has_pro_share_themes,
)


def test_share_themes_capability_and_feature_ids():
    assert PRO_THEME_PACKS_CAPABILITY == "pro.theme_packs"
    assert PRO_SHARE_THEMES_FEATURE == "theme_pack.import_official"


def test_require_pro_share_themes_without_license_raises():
    original = capabilities_mod._GRANT_ALL_CAPABILITIES_OVERRIDE
    capabilities_mod._GRANT_ALL_CAPABILITIES_OVERRIDE = False
    invalidate_capabilities_cache()
    try:
        with pytest.raises(CapabilityRequiredError):
            require_pro_share_themes()
        assert not user_has_pro_share_themes()
    finally:
        capabilities_mod._GRANT_ALL_CAPABILITIES_OVERRIDE = original
        invalidate_capabilities_cache()


def test_require_pro_share_themes_with_grant_all_override():
    original = capabilities_mod._GRANT_ALL_CAPABILITIES_OVERRIDE
    capabilities_mod._GRANT_ALL_CAPABILITIES_OVERRIDE = True
    invalidate_capabilities_cache()
    try:
        require_pro_share_themes()
        assert user_has_pro_share_themes()
    finally:
        capabilities_mod._GRANT_ALL_CAPABILITIES_OVERRIDE = original
        invalidate_capabilities_cache()


def test_license_required_message_mentions_share_themes():
    assert "Share themes" in LICENSE_REQUIRED_MESSAGE
    assert "Settings → License" in LICENSE_REQUIRED_MESSAGE


def test_sync_share_themes_pro_features_updates_hint_and_buttons():
    from unittest.mock import MagicMock

    host = MagicMock()
    host.themes_save_as_btn = MagicMock()
    host.themes_import_btn = MagicMock()
    host.themes_share_hint = MagicMock()

    original = capabilities_mod._GRANT_ALL_CAPABILITIES_OVERRIDE
    capabilities_mod._GRANT_ALL_CAPABILITIES_OVERRIDE = False
    invalidate_capabilities_cache()
    try:
        sync_share_themes_pro_features(host)
        host.themes_save_as_btn.setEnabled.assert_called_with(True)
        assert "Import a Pro license" in host.themes_share_hint.setText.call_args.args[0]

        capabilities_mod._GRANT_ALL_CAPABILITIES_OVERRIDE = True
        invalidate_capabilities_cache()
        sync_share_themes_pro_features(host)
        assert "Pro license active" in host.themes_share_hint.setText.call_args.args[0]
    finally:
        capabilities_mod._GRANT_ALL_CAPABILITIES_OVERRIDE = original
        invalidate_capabilities_cache()
