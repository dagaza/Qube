"""Build and start page guided tours."""

from __future__ import annotations

from PyQt6.QtWidgets import QWidget

from ui.components.onboarding_tour import OnboardingTour
from ui.onboarding.tour_registry import build_tour


def build_page_tour(parent: QWidget, tour_id: str) -> OnboardingTour | None:
    host = parent.window() if parent is not None else parent
    if host is None:
        return None
    return build_tour(tour_id, host)


def start_page_tour_on_window(
    host: QWidget,
    tour_id: str,
    *,
    area_display_name: str | None = None,
) -> bool:
    """Build and start via MainWindow tour manager."""
    if not hasattr(host, "request_page_tour"):
        return False
    host.request_page_tour(tour_id, area_display_name=area_display_name)
    return True
