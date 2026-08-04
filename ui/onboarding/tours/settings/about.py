"""Guided tour: Settings → About."""

from __future__ import annotations

from ui.components.onboarding_tour import OnboardingStep, OnboardingTour
from ui.onboarding.tour_helpers import open_settings_section
from ui.onboarding.tours.settings._common import make_settings_tour_finish_step


def _sv(host):
    return host.settings_view


def _open(host) -> None:
    open_settings_section(host, "about", anchor="about-qube")


def build_settings_about_tour(host) -> OnboardingTour:
    steps = [
        OnboardingStep(
            step_id="welcome",
            title="About Qube",
            body=(
                "See your installed version, open the Qube website, and check GitHub "
                "Releases for newer builds."
            ),
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="updates",
            title="Software updates",
            body=(
                "Click Check for updates to compare your version with the latest GitHub "
                "Release. Qube opens the matching download when a newer build is available."
            ),
            target_getter=lambda h: _sv(h).check_for_updates_btn,
            on_enter=lambda h: open_settings_section(h, "about", anchor="software-updates"),
        ),
        make_settings_tour_finish_step("About settings", _open),
    ]
    return OnboardingTour(host, steps)
