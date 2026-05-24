"""Engine-mode constant (behavior tests live in tests/test_settings_store.py)."""

import unittest

from core.app_settings import DEFAULT_ENGINE_MODE


class TestEngineModeDefault(unittest.TestCase):
    def test_default_engine_mode_constant_is_internal(self) -> None:
        self.assertEqual(DEFAULT_ENGINE_MODE, "internal")


if __name__ == "__main__":
    unittest.main()
