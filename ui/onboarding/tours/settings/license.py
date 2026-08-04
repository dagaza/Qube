"""Guided tour: Settings → License."""

from __future__ import annotations

from ui.components.onboarding_tour import OnboardingStep, OnboardingTour
from ui.onboarding.tour_helpers import open_settings_section
from ui.onboarding.tours.settings._common import make_settings_tour_finish_step


def _sv(host):
    return host.settings_view


def _open(host) -> None:
    open_settings_section(host, "license", anchor="license")


def build_settings_license_tour(host) -> OnboardingTour:
    steps = [
        OnboardingStep(
            step_id="welcome",
            title="License",
            body=(
                "Activate a QUBE1 license key from your purchase email or import a signed "
                ".qube-license file to unlock Pro capabilities such as Library Pro depth. "
                "Nothing prompts you on startup — this section is optional."
            ),
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="activate_key",
            title="Activate license key",
            body=(
                "Paste the QUBE1 license key from your email and click Activate license key. "
                "Qube validates the signature offline and caches the license locally."
            ),
            target_getter=lambda h: _sv(h).activate_license_key_btn,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="import",
            title="Import license file",
            body=(
                "Alternatively, select a signed .qube-license or JSON license file. Qube "
                "validates and caches it locally."
            ),
            target_getter=lambda h: _sv(h).import_license_btn,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="remove",
            title="Remove cached license",
            body=(
                "Delete the locally cached license. Pro toggles turn off on the next sync."
            ),
            target_getter=lambda h: _sv(h).remove_license_btn,
            on_enter=_open,
        ),
        make_settings_tour_finish_step("License", _open),
    ]
    return OnboardingTour(host, steps)
