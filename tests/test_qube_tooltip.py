import unittest
from unittest.mock import patch

from core.qube_tooltip import (
    _clamp_tip_position,
    _TOOLTIP_MAX_WIDTH_PX,
    _tooltip_clip_rect,
    _tooltip_label_width_px,
    _tooltip_text_height,
    _tooltip_widget_and_text,
    QubeToolTipController,
)
from PyQt6.QtCore import QPoint, QRect, QSize
from PyQt6.QtWidgets import QApplication, QLabel, QSpinBox, QWidget


class TestTooltipLabelWidth(unittest.TestCase):
    def test_short_text_shrink_wraps(self) -> None:
        self.assertEqual(_tooltip_label_width_px(72), 72)

    def test_long_text_caps_at_max(self) -> None:
        self.assertEqual(_tooltip_label_width_px(900), _TOOLTIP_MAX_WIDTH_PX)

    def test_zero_width_clamps_to_one(self) -> None:
        self.assertEqual(_tooltip_label_width_px(0), 1)


class TestTooltipContentSize(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PyQt6.QtWidgets import QApplication

        cls._app = QApplication.instance() or QApplication([])

    def test_long_paragraph_height_exceeds_single_line(self) -> None:
        ctrl = QubeToolTipController.instance()
        ctrl._ensure_popup()
        assert ctrl._label is not None
        long_text = " ".join(["word"] * 80)
        size = ctrl._label_content_size(long_text)
        self.assertEqual(size.width(), _TOOLTIP_MAX_WIDTH_PX)
        self.assertGreater(size.height(), ctrl._label.fontMetrics().height() * 2)

    def test_multiline_text_height_grows(self) -> None:
        ctrl = QubeToolTipController.instance()
        ctrl._ensure_popup()
        assert ctrl._label is not None
        one_line = ctrl._label_content_size("Short tip.")
        many_lines = ctrl._label_content_size("\n".join(["Line of text"] * 6))
        self.assertGreater(many_lines.height(), one_line.height())

    def test_spinbox_line_edit_inherits_parent_tooltip(self) -> None:
        spin = QSpinBox()
        tip = (
            "Acts as a background noise filter which controls when normal speech is "
            "considered loud enough to keep recording/transcription active."
        )
        spin.setToolTip(tip)
        line_edit = spin.findChild(type(spin.lineEdit())) if spin.lineEdit() else None
        self.assertIsNotNone(line_edit)
        anchor, resolved = _tooltip_widget_and_text(line_edit)  # type: ignore[arg-type]
        self.assertEqual(resolved, tip)
        self.assertIs(anchor, spin)

    def test_vad_tip_height_fits_wrapped_body(self) -> None:
        ctrl = QubeToolTipController.instance()
        ctrl._ensure_popup()
        assert ctrl._label is not None
        tip = (
            "Acts as a background noise filter which controls when normal speech is considered "
            "loud enough to keep recording/transcription active. Lower values protect against "
            "false positives."
        )
        size = ctrl._label_content_size(tip)
        self.assertEqual(size.width(), _TOOLTIP_MAX_WIDTH_PX)
        self.assertGreaterEqual(
            size.height(),
            _tooltip_text_height(tip, ctrl._label, _TOOLTIP_MAX_WIDTH_PX),
        )
        self.assertGreater(size.height(), ctrl._label.fontMetrics().height() * 2)


class TestTooltipClip(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_clip_rect_from_ancestor_property(self) -> None:
        parent = QWidget()
        parent.setProperty("qube_tooltip_clip", True)
        parent.resize(400, 300)
        child = QLabel(parent)
        child.move(50, 50)
        parent.show()
        clip = _tooltip_clip_rect(child)
        self.assertIsNotNone(clip)
        assert clip is not None
        self.assertGreaterEqual(clip.width(), 350)
        self.assertGreaterEqual(clip.height(), 250)

    def test_place_tip_stays_inside_clip_rect(self) -> None:
        ctrl = QubeToolTipController.instance()
        parent = QWidget()
        parent.setProperty("qube_tooltip_clip", True)
        parent.setGeometry(100, 100, 320, 240)
        parent.show()
        anchor = QLabel(parent)
        anchor.setGeometry(250, 200, 40, 16)
        anchor.show()
        anchor_pos = anchor.mapToGlobal(QPoint(anchor.width(), anchor.height()))
        sz = QSize(200, 120)
        placed = ctrl._place_tip(anchor_pos, sz, anchor=anchor)
        clip = _tooltip_clip_rect(anchor)
        assert clip is not None
        self.assertGreaterEqual(placed.x(), clip.left())
        self.assertGreaterEqual(placed.y(), clip.top())
        self.assertLessEqual(placed.x() + sz.width(), clip.right() + 1)
        self.assertLessEqual(placed.y() + sz.height(), clip.bottom() + 1)

    def test_clamp_tip_position(self) -> None:
        bounds = QRect(0, 0, 100, 80)
        pos = _clamp_tip_position(QPoint(90, 70), QSize(40, 30), bounds)
        self.assertEqual(pos, QPoint(60, 50))

    def test_leave_does_not_hide_when_cursor_still_on_anchor(self) -> None:
        ctrl = QubeToolTipController.instance()
        parent = QWidget()
        parent.show()
        anchor = QLabel(parent)
        anchor.setGeometry(20, 20, 72, 22)
        anchor.setToolTip("Verified size chip tooltip")
        anchor.show()
        anchor_pos = anchor.mapToGlobal(QPoint(36, 11))
        ctrl.show_tip(anchor, anchor_pos, "Verified size chip tooltip")
        self.assertTrue(ctrl._popup is not None and ctrl._popup.isVisible())
        # Simulate a spurious Leave while the cursor remains over the anchor.
        with patch.object(anchor, "underMouse", return_value=True):
            ctrl.hide_if_cursor_left_anchor()
        self.assertTrue(ctrl._popup.isVisible())
        ctrl.hide_tip()


if __name__ == "__main__":
    unittest.main()
