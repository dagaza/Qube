"""Tests for transcript timeline rail helpers."""

from __future__ import annotations

import unittest

from ui.components.transcript_timeline_rail import (
    compute_active_waypoint_index,
    compute_scroll_target_for_waypoint_y,
    compute_stacked_marker_centers,
    format_waypoint_tooltip,
    nearest_waypoint_index_for_y,
    transcript_timeline_should_show,
    truncate_waypoint_label,
)


class TestTranscriptTimelineRailHelpers(unittest.TestCase):
    def test_truncate_waypoint_label_collapses_whitespace(self) -> None:
        self.assertEqual(
            truncate_waypoint_label("  hello   world  "),
            "hello world",
        )

    def test_truncate_waypoint_label_ellipsis(self) -> None:
        long_text = "a" * 100
        result = truncate_waypoint_label(long_text, max_len=20)
        self.assertTrue(result.endswith("…"))
        self.assertLessEqual(len(result), 20)

    def test_format_waypoint_tooltip(self) -> None:
        self.assertEqual(
            format_waypoint_tooltip(2, 5, "Explain quantum tunneling"),
            "Turn 3 of 5\nExplain quantum tunneling",
        )
        self.assertEqual(format_waypoint_tooltip(0, 3, ""), "Turn 1 of 3")

    def test_compute_scroll_target_for_waypoint_y(self) -> None:
        self.assertEqual(
            compute_scroll_target_for_waypoint_y(
                200, margin=24, scroll_min=0, scroll_max=1000
            ),
            176,
        )
        self.assertEqual(
            compute_scroll_target_for_waypoint_y(
                10, margin=24, scroll_min=0, scroll_max=1000
            ),
            0,
        )
        self.assertEqual(
            compute_scroll_target_for_waypoint_y(
                2000, margin=24, scroll_min=0, scroll_max=500
            ),
            500,
        )

    def test_transcript_timeline_should_show_requires_overflow(self) -> None:
        self.assertFalse(
            transcript_timeline_should_show(400, 800, waypoint_count=3),
        )
        self.assertTrue(
            transcript_timeline_should_show(1200, 800, waypoint_count=2),
        )
        self.assertFalse(
            transcript_timeline_should_show(1200, 800, waypoint_count=0),
        )

    def test_compute_active_waypoint_index(self) -> None:
        ys = [0, 200, 400, 600]
        self.assertEqual(compute_active_waypoint_index(0, ys), 0)
        # Default viewport_margin=24: a turn activates once scroll_top + margin reaches its y.
        self.assertEqual(compute_active_waypoint_index(175, ys), 0)
        self.assertEqual(compute_active_waypoint_index(176, ys), 1)
        self.assertEqual(compute_active_waypoint_index(220, ys), 1)
        self.assertEqual(compute_active_waypoint_index(900, ys), 3)

    def test_nearest_waypoint_index_for_y(self) -> None:
        centers = [20.0, 60.0, 100.0]
        self.assertEqual(nearest_waypoint_index_for_y(58.0, centers), 1)
        self.assertEqual(nearest_waypoint_index_for_y(200.0, centers), -1)

    def test_compute_stacked_marker_centers_single(self) -> None:
        self.assertEqual(compute_stacked_marker_centers(1, 400), [200.0])

    def test_compute_stacked_marker_centers_even_stack(self) -> None:
        centers = compute_stacked_marker_centers(4, 400)
        self.assertEqual(len(centers), 4)
        self.assertLess(centers[0], centers[-1])
        gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
        self.assertAlmostEqual(gaps[0], gaps[1], places=5)
        self.assertAlmostEqual(gaps[1], gaps[2], places=5)


class TestTranscriptTimelineRailWidget(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PyQt6.QtWidgets import QApplication

        cls._app = QApplication.instance() or QApplication([])

    def test_rail_hides_without_overflow(self) -> None:
        from ui.components.transcript_timeline_rail import (
            TranscriptTimelineRail,
            TranscriptWaypointEntry,
        )

        rail = TranscriptTimelineRail()
        rail.resize(22, 400)
        rail.set_geometry_from_container(
            [TranscriptWaypointEntry(y=0, label="One")],
            container_height=200,
            show=False,
        )
        self.assertFalse(rail.isVisible())

    def test_rail_emits_click_for_nearest_marker(self) -> None:
        from ui.components.transcript_timeline_rail import (
            TranscriptTimelineRail,
            TranscriptWaypointEntry,
        )
        from PyQt6.QtCore import QPointF, Qt
        from PyQt6.QtGui import QMouseEvent

        rail = TranscriptTimelineRail()
        rail.resize(22, 400)
        rail.set_geometry_from_container(
            [
                TranscriptWaypointEntry(y=0, label="First"),
                TranscriptWaypointEntry(y=400, label="Second"),
            ],
            container_height=400,
            show=True,
        )
        clicked: list[int] = []
        rail.waypoint_clicked.connect(clicked.append)

        centers = rail._marker_center_ys()
        center_y = centers[1]
        event = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(rail.width() / 2, center_y),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        rail.mousePressEvent(event)
        self.assertEqual(clicked, [1])


if __name__ == "__main__":
    unittest.main()
