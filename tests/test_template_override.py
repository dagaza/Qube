from __future__ import annotations

import unittest

from core.template_override import detect_template_override


class TestTemplateOverride(unittest.TestCase):
    def test_gpt_oss_includes_return_stop_only(self) -> None:
        o = detect_template_override("openai_gpt-oss-20b", {})
        self.assertIsNotNone(o)
        assert o is not None
        self.assertIn("<|return|>", o.extra_stops)
        self.assertNotIn("<|end|>", o.extra_stops)
        self.assertEqual(o.template_type, "oss_harmony")

    def test_gpt_plus_oss_in_name(self) -> None:
        o = detect_template_override("My-org-gpt-something-oss-gguf", {})
        self.assertIsNotNone(o)
        assert o is not None
        self.assertIn("<|return|>", o.extra_stops)


if __name__ == "__main__":
    unittest.main()
