"""Guided tour: Settings → Diagnostics."""

from __future__ import annotations

from core.diagnostic_logs import iter_diagnostic_logs_by_category
from ui.components.onboarding_tour import OnboardingStep, OnboardingTour
from ui.onboarding.tour_helpers import open_settings_section
from ui.onboarding.tours.settings._common import make_settings_tour_finish_step
from ui.onboarding.tours.settings._diagnostic_logs import diagnostic_log_tour_steps


def _sv(host):
    return host.settings_view


def _open(host) -> None:
    open_settings_section(host, "diagnostics", anchor="logs")


def build_settings_diagnostics_tour(host) -> OnboardingTour:
    steps = [
        OnboardingStep(
            step_id="welcome",
            title="Diagnostics",
            body=(
                "Technical troubleshooting logs for application runtime events and skill "
                "activation. Privacy-sensitive audit logs live under Privacy & data."
            ),
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="logs_folder",
            title="Open logs folder",
            body=(
                "Reveal the logs directory in your file manager for all rotating debug "
                "files Qube writes under ~/.qube/logs."
            ),
            target_getter=lambda h: _sv(h).open_logs_folder_btn,
            on_enter=_open,
        ),
        *diagnostic_log_tour_steps(
            "diagnostics",
            iter_diagnostic_logs_by_category("technical"),
        ),
        make_settings_tour_finish_step("Diagnostics", _open),
    ]
    return OnboardingTour(host, steps)
