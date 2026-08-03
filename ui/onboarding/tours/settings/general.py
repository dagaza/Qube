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
                "Pick **Use inferred units** (default), **Metric**, or **Imperial** for weather "
                "and other numeric answers."
            ),
            target_getter=lambda h: _sv(h).profile_units_selector,
            on_enter=_open,
        ),
        OnboardingStep(
            step_id="composer",
            title="Composer @ mentions",
            body=(
                "Optional: typed @tool shorthands (e.g. @research) route like the @ palette. "
                "Off by default—recommended to use Attach (@) or chips above the composer."
            ),
            target_getter=lambda h: _sv(h).composer_bare_mention_routing_cb,
            on_enter=lambda h: open_settings_section(h, "general", anchor="composer"),
        ),
        OnboardingStep(
            step_id="hardware_hints",
            title="Hardware-aware Model Manager",
            body=(
                "Rank verified models and show Good fit badges from detected RAM and VRAM. "
                "May be less reliable on integrated GPUs or APUs."
            ),
            target_getter=lambda h: _sv(h).model_manager_hardware_suggestions_cb,
            on_enter=lambda h: open_settings_section(h, "general", anchor="discovery"),
        ),
        make_settings_tour_finish_step("General settings", _open),
    ]
    return OnboardingTour(host, steps)
