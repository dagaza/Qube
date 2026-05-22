import unittest
from unittest.mock import patch

from core.app_settings import (
    DEFAULT_ENGINE_MODE,
    ensure_engine_mode_initialized,
    get_engine_mode,
    get_model_manager_hardware_suggestions,
    get_onboarding_local_llm_tour_completed,
    reset_help_guidance_settings,
)


class TestEngineModeDefault(unittest.TestCase):
    def test_default_engine_mode_constant_is_internal(self) -> None:
        self.assertEqual(DEFAULT_ENGINE_MODE, "internal")

    def test_default_engine_mode_is_internal_without_setting(self) -> None:
        with patch("core.app_settings._settings") as mock_settings:
            mock_settings.return_value.value.side_effect = (
                lambda key, default, type=str: default
            )
            self.assertEqual(get_engine_mode(), "internal")

    def test_invalid_engine_mode_falls_back_to_internal(self) -> None:
        with patch("core.app_settings._settings") as mock_settings:
            mock_settings.return_value.value.return_value = "invalid"
            self.assertEqual(get_engine_mode(), "internal")

    def test_ensure_engine_mode_initialized_persists_internal_on_first_launch(self) -> None:
        with patch("core.app_settings._settings") as mock_settings:
            store: dict[str, object] = {}

            def _set_value(key: str, value: object) -> None:
                store[key] = value

            mock_settings.return_value.contains.return_value = False
            mock_settings.return_value.setValue.side_effect = _set_value
            mock_settings.return_value.sync.return_value = None
            mock_settings.return_value.value.side_effect = (
                lambda key, default, type=str: store.get(key, default)
            )

            self.assertEqual(ensure_engine_mode_initialized(), "internal")
            self.assertEqual(store.get("engine_mode"), "internal")

    def test_ensure_engine_mode_initialized_preserves_existing_choice(self) -> None:
        with patch("core.app_settings._settings") as mock_settings:
            store: dict[str, object] = {"engine_mode": "external"}

            mock_settings.return_value.contains.return_value = True
            mock_settings.return_value.value.side_effect = (
                lambda key, default, type=str: store.get(key, default)
            )

            self.assertEqual(ensure_engine_mode_initialized(), "external")
            self.assertEqual(store.get("engine_mode"), "external")


class TestModelManagerHardwareSuggestionsDefault(unittest.TestCase):
    def test_default_is_disabled(self) -> None:
        with patch("core.app_settings._settings") as mock_settings:
            mock_settings.return_value.value.side_effect = (
                lambda key, default, type=bool: default
            )
            self.assertFalse(get_model_manager_hardware_suggestions())


class TestResetHelpGuidanceSettings(unittest.TestCase):
    def test_reset_clears_tour_and_hardware_suggestions(self) -> None:
        with patch("core.app_settings._settings") as mock_settings:
            store: dict[str, object] = {
                "onboarding_local_llm_tour_completed": True,
                "model_manager_hardware_suggestions": True,
            }

            def _set_value(key: str, value: object) -> None:
                store[key] = value

            mock_settings.return_value.value.side_effect = (
                lambda key, default, type=bool: store.get(key, default)
            )
            mock_settings.return_value.setValue.side_effect = _set_value
            mock_settings.return_value.sync.return_value = None

            reset_help_guidance_settings()

            self.assertFalse(get_onboarding_local_llm_tour_completed())
            self.assertFalse(get_model_manager_hardware_suggestions())


if __name__ == "__main__":
    unittest.main()
