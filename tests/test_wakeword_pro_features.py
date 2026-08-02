"""Tests for Pro alternate wakeword library gating."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataclasses import dataclass

from core.capabilities import EditionTier, invalidate_capabilities_cache, resolve_capabilities
from core.wakeword_pro_features import (
    PREFERRED_DEFAULT_WAKEWORD_ID,
    build_wakeword_menu_items,
    is_alternate_wakeword,
    resolve_default_free_wakeword_spec,
    selectable_wakeword_specs,
    wakeword_selection_allowed,
)


@dataclass
class _TestSpec:
    wakeword_id: str
    display_name: str
    source: str = "local"
    path: str = ""
    default_threshold: float = 0.5
    recommended: bool = True


def _spec(wakeword_id: str, *, recommended: bool = True) -> _TestSpec:
    return _TestSpec(
        wakeword_id=wakeword_id,
        display_name=wakeword_id.replace("_", " ").title(),
        source="local",
        path=f"/tmp/{wakeword_id}.onnx",
        default_threshold=0.5,
        recommended=recommended,
    )


class _FakeManager:
    def __init__(self, specs: list[_TestSpec]) -> None:
        self._catalog = {spec.wakeword_id: spec for spec in specs}

    def get_by_id(self, wakeword_id: str) -> _TestSpec | None:
        return self._catalog.get(wakeword_id)

    def list_recommended(self) -> list[_TestSpec]:
        return [spec for spec in self._catalog.values() if spec.recommended]

    def list_community(self) -> list[_TestSpec]:
        return [spec for spec in self._catalog.values() if not spec.recommended]


class WakewordProFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        invalidate_capabilities_cache()

    def tearDown(self) -> None:
        invalidate_capabilities_cache()

    def test_home_tier_denies_wakeword_library(self) -> None:
        caps = resolve_capabilities(tier=EditionTier.HOME, source="test")
        self.assertFalse(caps.has("pro.wakeword_library"))

    def test_pro_tier_grants_wakeword_library(self) -> None:
        caps = resolve_capabilities(tier=EditionTier.PRO, source="test")
        self.assertTrue(caps.has("pro.wakeword_library"))

    def test_resolve_default_prefers_hey_qube(self) -> None:
        manager = _FakeManager(
            [
                _spec("hey_jarvis"),
                _spec(PREFERRED_DEFAULT_WAKEWORD_ID),
                _spec("hey_rhasspy"),
            ]
        )
        default = resolve_default_free_wakeword_spec(manager)
        assert default is not None
        self.assertEqual(default.wakeword_id, PREFERRED_DEFAULT_WAKEWORD_ID)

    def test_resolve_default_falls_back_to_jarvis(self) -> None:
        manager = _FakeManager([_spec("hey_jarvis"), _spec("hey_rhasspy")])
        default = resolve_default_free_wakeword_spec(manager)
        assert default is not None
        self.assertEqual(default.wakeword_id, "hey_jarvis")

    def test_free_tier_catalog_is_default_only(self) -> None:
        manager = _FakeManager(
            [
                _spec("hey_jarvis"),
                _spec("alexa", recommended=False),
            ]
        )
        with patch(
            "core.wakeword_pro_features.user_has_pro_wakeword_library",
            return_value=False,
        ):
            specs = selectable_wakeword_specs(manager)
            self.assertEqual(len(specs), 1)
            self.assertEqual(specs[0].wakeword_id, "hey_jarvis")
            items = build_wakeword_menu_items(manager)
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0][1], "Hey Jarvis")

    def test_pro_tier_catalog_includes_community(self) -> None:
        manager = _FakeManager(
            [
                _spec("hey_jarvis"),
                _spec("alexa", recommended=False),
            ]
        )
        with patch(
            "core.wakeword_pro_features.user_has_pro_wakeword_library",
            return_value=True,
        ):
            specs = selectable_wakeword_specs(manager)
            self.assertEqual(len(specs), 2)

    def test_alternate_requires_license(self) -> None:
        manager = _FakeManager([_spec("hey_jarvis"), _spec("hey_rhasspy")])
        default = _spec("hey_jarvis")
        alternate = _spec("hey_rhasspy")
        self.assertFalse(is_alternate_wakeword(default, manager))
        self.assertTrue(is_alternate_wakeword(alternate, manager))
        with patch(
            "core.wakeword_pro_features.user_has_pro_wakeword_library",
            return_value=False,
        ):
            self.assertTrue(wakeword_selection_allowed(default, manager))
            self.assertFalse(wakeword_selection_allowed(alternate, manager))

    def test_revoke_resets_alternate_selection(self) -> None:
        from core.wakeword_pro_features import revoke_unlicensed_wakeword_selection

        manager = _FakeManager([_spec("hey_jarvis"), _spec("hey_rhasspy")])
        worker = MagicMock()
        worker.wakeword_manager = manager
        worker.active_wakeword_id = "hey_rhasspy"
        worker.catalog_by_ui_name = {
            spec.display_name: spec for spec in manager._catalog.values()
        }
        worker.set_wakeword = MagicMock()

        with patch(
            "core.wakeword_pro_features.user_has_pro_wakeword_library",
            return_value=False,
        ), patch(
            "core.app_settings.get_active_wakeword_id",
            return_value="hey_rhasspy",
        ):
            self.assertTrue(revoke_unlicensed_wakeword_selection(worker))
            worker.set_wakeword.assert_called_once_with("Hey Jarvis")


if __name__ == "__main__":
    unittest.main()
