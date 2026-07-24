"""Guided tour: Settings → General."""

from __future__ import annotations

from ui.components.onboarding_tour import OnboardingStep, OnboardingTour
from ui.onboarding.tour_helpers import open_settings_section
from ui.onboarding.tours.settings._common import make_settings_tour_finish_step


def _sv(host):
    return host.settings_view


def _open(host) -> None:
    open_settings_section(host, "general")


def build_settings_general_tour(host) -> OnboardingTour:
    steps = [
        OnboardingStep(
            step_id="welcome",
            title="General settings",
            body=(
                "Application-wide preferences such as on-screen language. In Settings → "
                "Themes you can choose Dark, Light, or Follow system appearance; the "
                "moon/sun icon in the left navigation rail switches variants within "
                "your theme family when a matching option exists."
            ),
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="language",
            title="Application language",
            body=(
                "Switch between British and American English spelling for labels, tooltips, "
                "and other UI copy."
            ),
            target_getter=lambda h: _sv(h).general_language_card,
            on_enter=_open,
        ),
        make_settings_tour_finish_step("General settings", _open),
    ]
    return OnboardingTour(host, steps)
