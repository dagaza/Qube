"""Settings section header chrome for page guided tours (Phase 6a)."""

from __future__ import annotations

import unittest

from ui.onboarding.tour_registry import (
    get_tour_builder,
    settings_section_tour_id,
    tour_display_name,
)
from ui.views.settings.registry import SETTINGS_SECTIONS, get_section


class TestSettingsSectionHeaderRow(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from PyQt6.QtWidgets import QApplication

        cls._app = QApplication.instance() or QApplication([])

    def test_header_row_places_help_button_after_title(self) -> None:
        from PyQt6.QtWidgets import QHBoxLayout, QWidget

        from ui.views.settings.widgets import make_settings_section_header_row

        host = QWidget()
        host.resize(640, 120)
        host.show()

        title, btn, icon_lbl, row = make_settings_section_header_row(
            host,
            initial_tour_id="settings.voice_audio",
            initial_area_display_name="Voice & Audio settings",
        )
        outer = QHBoxLayout(host)
        outer.addWidget(row)
        self._app.processEvents()

        self.assertEqual(title.text(), "")
        self.assertEqual(btn.tour_id, "settings.voice_audio")
        self.assertLess(icon_lbl.geometry().x(), title.geometry().x())
        self.assertLess(title.geometry().x(), btn.geometry().x())

    def test_every_settings_section_has_registered_tour(self) -> None:
        for sec in SETTINGS_SECTIONS:
            tour_id = settings_section_tour_id(sec.id)
            self.assertIsNotNone(
                get_tour_builder(tour_id),
                msg=f"missing tour builder for {sec.id} -> {tour_id}",
            )

    def test_sync_header_updates_title_tour_id_and_enabled_state(self) -> None:
        from PyQt6.QtWidgets import QLabel, QWidget

        from ui.components.page_tour_help_button import PageTourHelpButton
        from ui.onboarding.settings_tour_header import sync_settings_section_tour_header

        host = QWidget()
        title = QLabel(host)
        icon_lbl = QLabel(host)
        btn = PageTourHelpButton("settings.voice_audio", parent=host)

        for sec in SETTINGS_SECTIONS:
            sync_settings_section_tour_header(title, btn, sec.id, icon_lbl=icon_lbl)
            tour_id = settings_section_tour_id(sec.id)
            self.assertEqual(title.text(), get_section(sec.id).title)
            self.assertEqual(icon_lbl.property("icon_name"), get_section(sec.id).icon)
            self.assertEqual(btn.tour_id, tour_id)
            self.assertEqual(btn.area_display_name, tour_display_name(tour_id))
            self.assertTrue(btn.isEnabled())


if __name__ == "__main__":
    unittest.main()
