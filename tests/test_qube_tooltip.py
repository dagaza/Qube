import unittest

from core.qube_tooltip import _TOOLTIP_MAX_WIDTH_PX, _tooltip_label_width_px


class TestTooltipLabelWidth(unittest.TestCase):
    def test_short_text_shrink_wraps(self) -> None:
        self.assertEqual(_tooltip_label_width_px(72), 72)

    def test_long_text_caps_at_max(self) -> None:
        self.assertEqual(_tooltip_label_width_px(900), _TOOLTIP_MAX_WIDTH_PX)

    def test_zero_width_clamps_to_one(self) -> None:
        self.assertEqual(_tooltip_label_width_px(0), 1)


if __name__ == "__main__":
    unittest.main()
