"""Guided tour: Settings → Advanced."""

from __future__ import annotations

from ui.components.onboarding_tour import OnboardingStep, OnboardingTour
from ui.onboarding.tour_helpers import open_settings_section
from ui.onboarding.tours.settings._common import make_settings_tour_finish_step


def _sv(host):
    return host.settings_view


def _open(host) -> None:
    open_settings_section(host, "advanced", anchor="json")


def build_settings_advanced_tour(host) -> OnboardingTour:
    steps = [
        OnboardingStep(
            step_id="welcome",
            title="Advanced settings",
            body=(
                "Edit raw JSON preferences when a key is not exposed elsewhere. "
                "Diagnostic logs and licenses moved to Diagnostics, Privacy & data, "
                "and License — proceed carefully here."
            ),
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="json",
            title="JSON settings editor",
            body=(
                "Inspect or override keys not exposed in the UI. Invalid JSON is rejected "
                "before save; reload when settings.json changes on disk."
            ),
            target_getter=lambda h: _sv(h).open_settings_json_btn,
            on_enter=_open,
        ),
        make_settings_tour_finish_step("Advanced settings", _open),
    ]
    return OnboardingTour(host, steps)
