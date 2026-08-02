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
                "Application-wide preferences: language, personalization, composer "
                "behavior, and Model Manager discovery hints."
            ),
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="language",
            title="Language",
            body=(
                "Switch between British and American English spelling for labels, tooltips, "
                "and other UI copy."
            ),
            target_getter=lambda h: _sv(h).general_language_card,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="units",
            title="Personalization",
            body=(
                "Choose default measurement units for weather and other numeric answers, "
                "or let Qube infer units from conversation."
            ),
            target_getter=lambda h: _sv(h).profile_units_selector,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="composer",
            title="Composer @ mentions",
            body=(
                "Optionally treat typed @tool shorthands (e.g. @research) as routing, "
                "or keep using the @ picker and chips above the composer."
            ),
            target_getter=lambda h: _sv(h).composer_bare_mention_routing_cb,
            on_enter=lambda h: open_settings_section(h, "general", anchor="composer"),
        ),
        OnboardingStep(
            step_id="hardware_hints",
            title="Hardware-aware Model Manager",
            body=(
                "When enabled, Model Manager ranks verified models and shows Good fit "
                "badges from detected RAM and VRAM."
            ),
            target_getter=lambda h: _sv(h).model_manager_hardware_suggestions_cb,
            on_enter=lambda h: open_settings_section(h, "general", anchor="discovery"),
        ),
        make_settings_tour_finish_step("General settings", _open),
    ]
    return OnboardingTour(host, steps)
