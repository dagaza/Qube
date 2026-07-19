"""Settings right-pane section header sync for page guided tours."""

from __future__ import annotations

from PyQt6.QtWidgets import QLabel

from core.paths import resource_path
from ui.components.page_tour_help_button import PageTourHelpButton
from ui.onboarding.tour_registry import (
    get_tour_builder,
    settings_section_tour_id,
    tour_display_name,
)
from ui.views.settings.registry import get_section


def configure_settings_section_icon_label(
    icon_lbl: QLabel,
    section_id: str,
    *,
    icon_size: int = 20,
) -> None:
    """Store icon metadata on the primary settings header icon label."""
    sec = get_section(str(section_id))
    if sec is None:
        return
    icon_lbl.setProperty("icon_name", sec.icon)
    icon_lbl.setProperty(
        "svg_path",
        str(resource_path(*sec.svg_icon)) if sec.svg_icon is not None else "",
    )
    icon_lbl.setProperty("icon_size", icon_size)


def sync_settings_section_tour_header(
    title_lbl: QLabel,
    tour_btn: PageTourHelpButton,
    section_id: str | None,
    *,
    icon_lbl: QLabel | None = None,
) -> None:
    """Update the section title, icon, and ? button for the active settings section."""
    if not section_id:
        return
    sec = get_section(str(section_id))
    title = sec.title if sec is not None else str(section_id)
    title_lbl.setText(title)
    tour_id = settings_section_tour_id(str(section_id))
    tour_btn.set_tour(tour_id, area_display_name=tour_display_name(tour_id))
    tour_btn.setEnabled(get_tour_builder(tour_id) is not None)
    if icon_lbl is not None:
        configure_settings_section_icon_label(icon_lbl, str(section_id))
