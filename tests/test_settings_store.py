import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.settings_store import (
    SettingsStore,
    _LEGACY_TO_DOTTED,
    reset_settings_store_for_tests,
)
from core import app_settings


class TestSettingsStore(unittest.TestCase):
    def setUp(self) -> None:
        reset_settings_store_for_tests()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.user_path = Path(self._tmpdir.name) / "settings.json"
        self.schema_path = Path(self._tmpdir.name) / "schema.json"
        self.schema_path.write_text(
            json.dumps(
                {
                    "qube.engine.mode": {
                        "type": "string",
                        "enum": ["internal", "external"],
                        "default": "internal",
                    },
                    "qube.memory.enrichment": {"type": "boolean", "default": True},
                    "qube.native.gpuLayers": {
                        "type": ["integer", "null"],
                        "minimum": 0,
                        "maximum": 200,
                        "default": None,
                    },
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        reset_settings_store_for_tests()
        self._tmpdir.cleanup()

    def _store(self) -> SettingsStore:
        return SettingsStore(user_path=self.user_path, schema_path=self.schema_path)

    def test_defaults_when_file_missing_and_no_qsettings(self) -> None:
        with patch.object(SettingsStore, "_migrate_from_qsettings", return_value=False):
            store = self._store()
        self.assertEqual(store.get("qube.engine.mode"), "internal")
        self.assertTrue(store.get("qube.memory.enrichment"))
        self.assertFalse(store.contains("qube.engine.mode"))

    def test_set_persists_override_and_omits_schema_default(self) -> None:
        with patch.object(SettingsStore, "_migrate_from_qsettings", return_value=False):
            store = self._store()
        store.set("qube.engine.mode", "external")
        self.assertTrue(self.user_path.is_file())
        data = json.loads(self.user_path.read_text(encoding="utf-8"))
        self.assertEqual(data["qube.engine.mode"], "external")
        store.set("qube.engine.mode", "internal")
        data = json.loads(self.user_path.read_text(encoding="utf-8"))
        self.assertNotIn("qube.engine.mode", data)

    def test_migrate_from_qsettings(self) -> None:
        mock_qs = MagicMock()
        mock_qs.contains.side_effect = lambda key: key == "engine_mode"
        mock_qs.value.side_effect = lambda key: "external" if key == "engine_mode" else None

        with patch("PyQt6.QtCore.QSettings", return_value=mock_qs):
            store = SettingsStore(user_path=self.user_path, schema_path=self.schema_path)
        self.assertEqual(store.get("qube.engine.mode"), "external")
        self.assertTrue(self.user_path.is_file())

    def test_reload_from_disk_detects_changes(self) -> None:
        with patch.object(SettingsStore, "_migrate_from_qsettings", return_value=False):
            store = self._store()
        store.set("qube.engine.mode", "external")
        self.user_path.write_text(
            json.dumps({"qube.engine.mode": "internal"}, indent=2) + "\n",
            encoding="utf-8",
        )
        store._refresh_disk_mtime()
        result = store.reload_from_disk()
        self.assertTrue(result.ok)
        self.assertIn("qube.engine.mode", result.changed_keys)
        self.assertEqual(store.get("qube.engine.mode"), "internal")

    def test_reload_invalid_json_keeps_prior_values(self) -> None:
        with patch.object(SettingsStore, "_migrate_from_qsettings", return_value=False):
            store = self._store()
        store.set("qube.engine.mode", "external")
        self.user_path.write_text("{ not json", encoding="utf-8")
        store._refresh_disk_mtime()
        result = store.reload_from_disk()
        self.assertFalse(result.ok)
        self.assertEqual(store.get("qube.engine.mode"), "external")

    def test_validate_json_text(self) -> None:
        with patch.object(SettingsStore, "_migrate_from_qsettings", return_value=False):
            store = self._store()
        ok = store.validate_json_text('{"qube.engine.mode": "external"}')
        self.assertTrue(ok.ok)
        self.assertEqual(ok.overrides["qube.engine.mode"], "external")
        bad = store.validate_json_text('{"qube.native.gpuLayers": "lots"}')
        self.assertFalse(bad.ok)

    def test_format_json_text(self) -> None:
        with patch.object(SettingsStore, "_migrate_from_qsettings", return_value=False):
            store = self._store()
        formatted, err = store.format_json_text('{"b":1,"a":2}')
        self.assertIsNone(err)
        self.assertIn('"a": 2', formatted)

    def test_save_from_json_text(self) -> None:
        with patch.object(SettingsStore, "_migrate_from_qsettings", return_value=False):
            store = self._store()
        result = store.save_from_json_text('{"qube.engine.mode": "external"}')
        self.assertTrue(result.ok)
        self.assertEqual(store.get("qube.engine.mode"), "external")
        self.assertIn("qube.engine.mode", result.changed_keys)

    def test_legacy_map_covers_all_app_settings_keys(self) -> None:
        from core import app_settings as mod

        dotted_keys = {
            mod.KEY_MEMORY_ENRICHMENT,
            mod.KEY_ENGINE_MODE,
            mod.KEY_NATIVE_MODEL_PATH,
            mod.KEY_NATIVE_GPU_LAYERS,
            mod.KEY_NATIVE_CPU_THREADS,
            mod.KEY_NATIVE_CHAT_FORMAT,
            mod.KEY_NATIVE_PROMPT_LAYOUT,
            mod.KEY_NATIVE_AUTO_LOAD_ON_STARTUP,
            mod.KEY_ONBOARDING_LOCAL_LLM_TOUR,
            mod.KEY_MODEL_MANAGER_HARDWARE_SUGGESTIONS,
            mod.KEY_MODELS_DIRECTORY,
            mod.KEY_NATIVE_REASONING_DISPLAY,
            mod.KEY_WAKEWORD_ACTIVE_ID,
            mod.KEY_WAKEWORD_THRESHOLDS,
            mod.KEY_AUDIO_INPUT_DEVICE,
            mod.KEY_AUDIO_OUTPUT_DEVICE,
        }
        self.assertEqual(set(_LEGACY_TO_DOTTED.values()), dotted_keys)


class TestAppSettingsWithJsonStore(unittest.TestCase):
    def setUp(self) -> None:
        reset_settings_store_for_tests()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.user_path = Path(self._tmpdir.name) / "settings.json"
        schema_src = Path(__file__).resolve().parents[1] / "assets" / "config" / "settings.schema.json"
        patcher = patch("core.settings_store.default_user_settings_path", return_value=self.user_path)
        patcher2 = patch("core.settings_store.bundled_settings_schema_path", return_value=schema_src)
        self.addCleanup(patcher.stop)
        self.addCleanup(patcher2.stop)
        patcher.start()
        patcher2.start()

    def tearDown(self) -> None:
        reset_settings_store_for_tests()
        self._tmpdir.cleanup()

    def test_default_engine_mode_is_internal(self) -> None:
        self.assertEqual(app_settings.DEFAULT_ENGINE_MODE, "internal")
        self.assertEqual(app_settings.get_engine_mode(), "internal")

    def test_invalid_engine_mode_falls_back_to_internal(self) -> None:
        store = SettingsStore(user_path=self.user_path)
        store.set("qube.engine.mode", "not-a-mode")  # coerce falls back to default
        reset_settings_store_for_tests()
        self.assertEqual(app_settings.get_engine_mode(), "internal")

    def test_ensure_engine_mode_initialized_persists_internal_on_first_launch(self) -> None:
        self.assertEqual(app_settings.ensure_engine_mode_initialized(), "internal")
        data = json.loads(self.user_path.read_text(encoding="utf-8"))
        self.assertEqual(data.get("qube.engine.mode"), "internal")

    def test_ensure_engine_mode_initialized_preserves_existing_choice(self) -> None:
        self.user_path.write_text(
            json.dumps({"qube.engine.mode": "external"}, indent=2) + "\n",
            encoding="utf-8",
        )
        reset_settings_store_for_tests()
        self.assertEqual(app_settings.ensure_engine_mode_initialized(), "external")

    def test_reset_help_guidance_settings(self) -> None:
        with patch.object(SettingsStore, "_migrate_from_qsettings", return_value=False):
            SettingsStore(user_path=self.user_path)
        app_settings.set_onboarding_local_llm_tour_completed(True)
        app_settings.set_model_manager_hardware_suggestions(True)
        app_settings.reset_help_guidance_settings()
        self.assertFalse(app_settings.get_onboarding_local_llm_tour_completed())
        self.assertFalse(app_settings.get_model_manager_hardware_suggestions())


if __name__ == "__main__":
    unittest.main()
