"""Tests for Pro custom STT / TTS / embedding model path gating."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core import capabilities as capabilities_mod
from core import embedding_models as em
from core import stt_models as sm
from core import tts_models as tm
from core.capabilities import EditionTier, invalidate_capabilities_cache, resolve_capabilities
from core.model_paths_pro_features import (
    PRO_CUSTOM_MODEL_PATHS_FEATURE,
    effective_advanced_embedding_unlocked,
    revoke_unlicensed_custom_model_paths,
    sync_custom_model_paths_pro_features,
    user_has_pro_custom_model_paths,
)


class ModelPathsProFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        invalidate_capabilities_cache()

    def tearDown(self) -> None:
        invalidate_capabilities_cache()

    def test_home_tier_denies_custom_model_paths(self) -> None:
        caps = resolve_capabilities(tier=EditionTier.HOME, source="test")
        self.assertFalse(caps.has("pro.custom_model_paths"))
        self.assertFalse(user_has_pro_custom_model_paths())

    def test_pro_tier_grants_custom_model_paths(self) -> None:
        caps = resolve_capabilities(tier=EditionTier.PRO, source="test")
        self.assertTrue(caps.has("pro.custom_model_paths"))

    def test_effective_unlock_requires_license_and_flag(self) -> None:
        with patch(
            "core.model_paths_pro_features.user_has_pro_custom_model_paths",
            return_value=True,
        ), patch(
            "core.app_settings.get_advanced_embedding_unlocked",
            return_value=True,
        ):
            self.assertTrue(effective_advanced_embedding_unlocked())

        with patch(
            "core.model_paths_pro_features.user_has_pro_custom_model_paths",
            return_value=False,
        ), patch(
            "core.app_settings.get_advanced_embedding_unlocked",
            return_value=True,
        ):
            self.assertFalse(effective_advanced_embedding_unlocked())

    def test_revoke_clears_unlock_flags_and_paths(self) -> None:
        with patch(
            "core.model_paths_pro_features.user_has_pro_custom_model_paths",
            return_value=False,
        ), patch(
            "core.app_settings.get_advanced_stt_unlocked",
            return_value=True,
        ), patch(
            "core.app_settings.get_advanced_tts_unlocked",
            return_value=True,
        ), patch(
            "core.app_settings.get_advanced_embedding_unlocked",
            return_value=True,
        ), patch(
            "core.app_settings.get_stt_model_path",
            return_value="/tmp/custom-stt",
        ), patch(
            "core.app_settings.get_tts_model_path",
            return_value="/tmp/custom.onnx",
        ), patch(
            "core.app_settings.get_embedding_model_path",
            return_value="/tmp/custom.gguf",
        ), patch(
            "core.tts_models.is_protected_tts_model",
            return_value=False,
        ), patch(
            "core.app_settings.set_advanced_stt_unlocked"
        ) as set_stt_unlock, patch(
            "core.app_settings.set_advanced_tts_unlocked"
        ) as set_tts_unlock, patch(
            "core.app_settings.set_advanced_embedding_unlocked"
        ) as set_emb_unlock, patch(
            "core.app_settings.set_stt_model_path"
        ) as set_stt_path, patch(
            "core.app_settings.set_tts_model_path"
        ) as set_tts_path, patch(
            "core.app_settings.set_embedding_model_path"
        ) as set_emb_path:
            self.assertTrue(revoke_unlicensed_custom_model_paths())
            set_stt_unlock.assert_called_once_with(False)
            set_tts_unlock.assert_called_once_with(False)
            set_emb_unlock.assert_called_once_with(False)
            set_stt_path.assert_called_once_with("")
            set_tts_path.assert_called_once_with("")
            set_emb_path.assert_called_once_with("")

    def test_sync_reconciles_toggle_without_license(self) -> None:
        class FakeToggle:
            def __init__(self) -> None:
                self._checked = True
                self._blocked = False

            def blockSignals(self, blocked: bool) -> None:
                self._blocked = blocked

            def setChecked(self, checked: bool) -> None:
                self._checked = checked

            def isChecked(self) -> bool:
                return self._checked

            def setEnabled(self, _enabled: bool) -> None:
                return None

        class FakeHost:
            def __init__(self) -> None:
                self.advanced_stt_toggle = FakeToggle()
                self._apply_advanced_stt_panel_visibility_called = False

            def _apply_advanced_stt_panel_visibility(self) -> None:
                self._apply_advanced_stt_panel_visibility_called = True

        host = FakeHost()
        with patch(
            "core.model_paths_pro_features.user_has_pro_custom_model_paths",
            return_value=False,
        ), patch(
            "core.model_paths_pro_features.revoke_unlicensed_custom_model_paths",
            return_value=False,
        ), patch(
            "core.app_settings.get_advanced_stt_unlocked",
            return_value=True,
        ), patch(
            "core.app_settings.get_advanced_tts_unlocked",
            return_value=False,
        ), patch(
            "core.app_settings.get_advanced_embedding_unlocked",
            return_value=False,
        ):
            sync_custom_model_paths_pro_features(host)
        self.assertFalse(host.advanced_stt_toggle.isChecked())
        self.assertTrue(host._apply_advanced_stt_panel_visibility_called)


class ModelPathsRuntimeResolutionTests(unittest.TestCase):
    def test_stt_custom_override_ignored_without_pro(self) -> None:
        def body(root: Path) -> None:
            stt_dir = Path(sm.get_stt_models_dir())
            custom = stt_dir / "my-whisper"
            custom.mkdir(parents=True)
            (custom / "model.bin").write_bytes(b"x")
            with patch(
                "core.stt_models.get_stt_model_path",
                return_value=str(custom.resolve()),
            ), patch(
                "core.model_paths_pro_features.custom_stt_override_allowed",
                return_value=False,
            ):
                self.assertEqual(sm.resolve_active_stt_model_spec(), sm.BUNDLED_STT_MODEL_ID)

        self._run_in_tmp(body)

    def test_embedding_custom_override_ignored_without_pro(self) -> None:
        def body(root: Path) -> None:
            emb_dir = Path(em.get_embedding_models_dir())
            custom = emb_dir / "custom-embed.gguf"
            custom.write_bytes(b"z" * 100)
            with patch(
                "core.embedding_models.get_embedding_model_path",
                return_value=str(custom.resolve()),
            ), patch(
                "core.model_paths_pro_features.custom_embedding_override_allowed",
                return_value=False,
            ):
                self.assertEqual(em.resolve_active_gguf_path(), "")

        self._run_in_tmp(body)

    def _run_in_tmp(self, fn) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            prev = os.getcwd()
            os.chdir(tmp)
            try:
                with patch(
                    "core.stt_models.models_root",
                    return_value=tmp_path / "models",
                ), patch(
                    "core.embedding_models.models_root",
                    return_value=tmp_path / "models",
                ):
                    fn(tmp_path)
            finally:
                os.chdir(prev)


if __name__ == "__main__":
    unittest.main()
