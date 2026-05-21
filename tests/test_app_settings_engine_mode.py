import unittest
from unittest.mock import patch

from core.app_settings import get_engine_mode


class TestEngineModeDefault(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
