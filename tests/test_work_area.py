"""Work-area bounds for maximized frameless windows."""

import unittest

from PyQt6.QtCore import QRect

from core.platform.work_area import parse_net_workarea_line, _clamp_rect_to_monitor


class WorkAreaTests(unittest.TestCase):
    def test_parse_net_workarea_line(self) -> None:
        rect = parse_net_workarea_line("_NET_WORKAREA(CARDINAL) = 0, 32, 1920, 1080")
        self.assertIsNotNone(rect)
        self.assertEqual(rect.x(), 0)
        self.assertEqual(rect.y(), 32)
        self.assertEqual(rect.width(), 1920)
        self.assertEqual(rect.height(), 1048)

    def test_clamp_rect_to_monitor(self) -> None:
        monitor = QRect(1920, 0, 1920, 1080)
        work = QRect(0, 32, 3840, 1048)
        clamped = _clamp_rect_to_monitor(work, monitor)
        self.assertEqual(clamped.x(), 1920)
        self.assertEqual(clamped.y(), 32)
        self.assertEqual(clamped.width(), 1920)
        self.assertEqual(clamped.height(), 1048)


if __name__ == "__main__":
    unittest.main()
